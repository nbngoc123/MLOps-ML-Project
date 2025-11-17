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
import boto3 # Dùng boto3 cho thao tác bucket ổn định hơn
import os
import mlflow
import logging

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
    description='Train & Deploy Sentiment Model',
    schedule='0 2 * * *', 
    catchup=False,
)

# -------------------------------
# Tasks (ĐÃ SỬA: KHỞI TẠO KẾT NỐI TRONG HÀM)
# -------------------------------

@task(dag=dag)
def ensure_bucket_exists():
    """Dùng Boto3 để tạo bucket (ổn định hơn s3fs cho việc này)."""
    log.info(f"Đang kết nối tới MinIO: {MINIO_ENDPOINT}")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    
    try:
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
        log.info(f"Bucket {MINIO_BUCKET} đã tồn tại.")
    except Exception:
        log.info(f"Bucket {MINIO_BUCKET} chưa tồn tại. Đang tạo...")
        s3_client.create_bucket(Bucket=MINIO_BUCKET)
    
    # Tạo các "thư mục" giả lập (S3 không có thư mục thật, chỉ là prefix)
    # Ta tạo file .keep để giữ chỗ
    folders = ["raw", "processed", "models"]
    for folder in folders:
        try:
            s3_client.put_object(Bucket=MINIO_BUCKET, Key=f"{folder}/.keep")
        except Exception as e:
            log.warning(f"Không thể tạo placeholder cho {folder}: {e}")

    log.info("Cấu trúc thư mục đã sẵn sàng.")

@task(dag=dag)
def extract_and_combine_data():
    # KHỞI TẠO s3fs BÊN TRONG HÀM
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
    
    raw_path_prefix = f"{MINIO_BUCKET}/raw/"
    all_csv_files = fs.glob(f"{raw_path_prefix}*.csv")
    
    if not all_csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {raw_path_prefix}")
    
    log.info(f"Tìm thấy {len(all_csv_files)} file: {all_csv_files}")

    df_list = []
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

    for file_path in all_csv_files:
        log.info(f"Đọc file: {file_path}")
        df_list.append(pd.read_csv(f"s3://{file_path}", storage_options=storage_options))
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
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
    
    df = pd.read_csv(f"s3://{s3_raw_temp}", storage_options=storage_options)
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
    df = pd.read_csv(f"s3://{s3_clean}", storage_options=storage_options)
    
    if df.empty or len(df) < 5:
        raise ValueError("Dữ liệu quá ít.")

    # Map label
    label_map = {"positive": 1, "negative": 0, "neutral": 2}
    # Xử lý trường hợp label trong CSV không khớp
    df['label_id'] = df['label'].map(label_map).fillna(2) # Mặc định neutral nếu lỗi

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].astype(str), df['label_id'],
        test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Lưu artifacts
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

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
    return "register_model" if train_output.get("accuracy") >= 0.6 else "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    # KHỞI TẠO s3fs TRONG HÀM
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    
    version_path = f"{MINIO_BUCKET}/models/sentiment/v{registered.version}"
    fs.makedirs(version_path, exist_ok=True)
    fs.put("/tmp/models/model.pkl", f"{version_path}/model.pkl")
    fs.put("/tmp/models/vectorizer.pkl", f"{version_path}/vectorizer.pkl")

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