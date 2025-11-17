from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import s3fs
import boto3
import os
import mlflow
import logging
from botocore.client import Config

# Khởi tạo Logger
log = logging.getLogger(__name__)

# -------------------------------
# Configuration
# -------------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio-service.nexusml.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "") 
MINIO_BUCKET = "nexusml"

# MLflow setup
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 16),
    'email_on_failure': True,
    'email': ['ml-alert@yourcompany.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_classification_minio_optimized',
    default_args=default_args,
    description='Train & Deploy Sentiment Model using Boto3 for Bucket Ops',
    schedule='0 2 * * *', 
    catchup=False,
)

# -------------------------------
# Helper Function: Get Boto3 Client
# -------------------------------
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )

# -------------------------------
# Tasks 
# -------------------------------

@task(dag=dag)
def ensure_bucket_exists():
    """Dùng Boto3 để tạo bucket và folder ảo."""
    s3 = get_s3_client()
    
    # 1. Kiểm tra và tạo Bucket
    try:
        s3.head_bucket(Bucket=MINIO_BUCKET)
        log.info(f"Bucket '{MINIO_BUCKET}' đã tồn tại.")
    except Exception:
        log.info(f"Bucket '{MINIO_BUCKET}' chưa tồn tại. Đang tạo...")
        try:
            s3.create_bucket(Bucket=MINIO_BUCKET)
        except Exception as e:
            if "BucketAlreadyOwnedByYou" not in str(e):
                raise e

    # 2. Tạo folder ảo (raw, processed, models) bằng file .keep
    folders = ["raw", "processed", "models"]
    for folder in folders:
        key = f"{folder}/.keep"
        try:
            s3.put_object(Bucket=MINIO_BUCKET, Key=key, Body="")
            log.info(f"Đã tạo placeholder: {key}")
        except Exception as e:
            log.warning(f"Không thể tạo file {key}: {e}")

@task(dag=dag)
def extract_and_combine_data():
    """Dùng Boto3 để liệt kê file, sau đó dùng Pandas+s3fs để đọc."""
    s3 = get_s3_client()
    
    # 1. Liệt kê tất cả file .csv trong thư mục raw/
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix="raw/")
    
    csv_files = []
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                csv_files.append(f"s3://{MINIO_BUCKET}/{obj['Key']}")
    
    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {MINIO_BUCKET}/raw/")
    
    log.info(f"Tìm thấy {len(csv_files)} file CSV: {csv_files}")

    # 2. Đọc và gộp dữ liệu
    df_list = []
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

    for file_path in csv_files:
        log.info(f"Đang đọc: {file_path}")
        try:
            df_temp = pd.read_csv(file_path, storage_options=storage_options)
            df_list.append(df_temp)
        except Exception as e:
            log.error(f"Lỗi đọc file {file_path}: {e}")

    if not df_list:
        raise ValueError("Không đọc được dữ liệu nào.")

    combined_df = pd.concat(df_list, ignore_index=True)
    log.info(f"Tổng số dòng dữ liệu: {len(combined_df)}")

    # 3. Lưu file gộp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_path_temp = f"{MINIO_BUCKET}/processed/combined_raw_{timestamp}.csv"
    
    combined_df.to_csv(f"s3://{s3_path_temp}", index=False, storage_options=storage_options)
    return s3_path_temp 

@task(dag=dag)
def preprocess_data(s3_raw_temp: str):
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Tiền xử lý file: {s3_raw_temp}")
    df = pd.read_csv(f"s3://{s3_raw_temp}", storage_options=storage_options)
    
    # Làm sạch text
    df['text'] = df['text'].str.lower().str.replace(
        r'[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
        '', regex=True
    )
    
    s3_clean = s3_raw_temp.replace("combined_raw", "clean_data")
    df.to_csv(f"s3://{s3_clean}", index=False, storage_options=storage_options)
    return s3_clean

@task(dag=dag)
def train_model(s3_clean: str):
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Huấn luyện trên file: {s3_clean}")
    df = pd.read_csv(f"s3://{s3_clean}", storage_options=storage_options)
    
    if df.empty or len(df) < 5:
        raise ValueError("Dữ liệu quá ít để huấn luyện.")

    # Map label & xử lý dữ liệu
    label_map = {"positive": 1, "negative": 0, "neutral": 2}
    df['label_id'] = df['label'].map(label_map).fillna(2)
    
    # Đảm bảo không có NaN trong text
    df['text'] = df['text'].fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].astype(str), df['label_id'],
        test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Lưu artifacts cục bộ
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    # Log lên MLflow
    mlflow.set_experiment("sentiment_classification")
    with mlflow.start_run(run_name=f"minio_run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
        
    return {"accuracy": acc, "run_id": run.info.run_id}

@task(dag=dag)
def evaluate_model(train_output: dict):
    acc = train_output.get("accuracy")
    log.info(f"Độ chính xác mô hình: {acc}")
    return "register_model" if acc >= 0.6 else "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    log.info(f"Đã đăng ký model phiên bản: {registered.version}")
    
    # Upload bản sao model sang MinIO dùng Boto3
    s3 = get_s3_client()
    version_path = f"models/sentiment/v{registered.version}"
    
    s3.upload_file("/tmp/models/model.pkl", MINIO_BUCKET, f"{version_path}/model.pkl")
    s3.upload_file("/tmp/models/vectorizer.pkl", MINIO_BUCKET, f"{version_path}/vectorizer.pkl")
    
    log.info(f"Đã upload model backup tới: {MINIO_BUCKET}/{version_path}")

# -------------------------------
# Dependencies
# -------------------------------
notify_success = BashOperator(task_id='notify_success', bash_command='echo "Success!"', dag=dag)
notify_fail = BashOperator(task_id='notify_failure', bash_command='echo "Failed!"', trigger_rule=TriggerRule.ONE_FAILED, dag=dag)

bucket = ensure_bucket_exists()
raw = extract_and_combine_data()
clean = preprocess_data(raw)
train = train_model(clean)
branch = evaluate_model(train)
register = register_model(train)

bucket >> raw >> clean >> train >> branch
branch >> register >> notify_success
branch >> notify_fail