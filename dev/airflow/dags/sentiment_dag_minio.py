from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from mlflow.tracking import MlflowClient
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
import gc 

log = logging.getLogger(__name__)

# -------------------------------
# Configuration
# -------------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin") 
MINIO_BUCKET = "nexusml" 

# MLflow setup
# MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 16),
    'email_on_failure': True, # Lưu ý: Cần cấu hình SMTP trong Airflow nếu muốn gửi mail thật
    'email': ['ml-alert@yourcompany.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_classification_minio_optimized',
    default_args=default_args,
    description='Train & Deploy Sentiment Model (Memory Optimized)',
    schedule='0 2 * * *', 
    catchup=False,
)

# -------------------------------
# Utility Functions
# -------------------------------
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

# -------------------------------
# Tasks
# -------------------------------

@task(dag=dag)
def ensure_bucket_exists():
    s3_client = get_s3_client()
    
    # 🛠️ FIX 1: Tạo cả bucket 'nexusml' VÀ 'mlflow' để tránh lỗi permission
    buckets_to_check = [MINIO_BUCKET, "mlflow"]
    
    for bucket in buckets_to_check:
        try:
            s3_client.head_bucket(Bucket=bucket)
            log.info(f"Bucket {bucket} đã tồn tại.")
        except Exception:
            log.info(f"Bucket {bucket} chưa tồn tại. Đang tạo...")
            s3_client.create_bucket(Bucket=bucket)
    
    # Tạo folder placeholder cho bucket dữ liệu
    folders = ["raw", "processed", "models"]
    for folder in folders:
        try:
            s3_client.put_object(Bucket=MINIO_BUCKET, Key=f"{folder}/.keep", Body=b'')
        except Exception as e:
            log.warning(f"Lỗi tạo placeholder {folder}: {e}")

@task(dag=dag)
def extract_and_combine_data():
    """
    Phiên bản Memory-Safe: Đọc từng file và ghi append vào đĩa cứng (/tmp).
    """
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
    
    raw_path_prefix = f"{MINIO_BUCKET}/raw/"
    all_csv_files = fs.glob(f"{raw_path_prefix}*.csv")
    
    if not all_csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {raw_path_prefix}")
    
    log.info(f"Tìm thấy {len(all_csv_files)} files. Bắt đầu xử lý Stream...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_temp_file = f"/tmp/combined_{timestamp}.csv"
    
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

    first_file = True
    
    for file_path in all_csv_files:
        log.info(f"Đang gộp file: {file_path}")
        try:
            with pd.read_csv(f"s3://{file_path}", storage_options=storage_options, chunksize=10000) as reader:
                for chunk in reader:
                    mode = 'w' if first_file else 'a'
                    header = first_file
                    chunk.to_csv(local_temp_file, mode=mode, header=header, index=False)
                    first_file = False
        except Exception as e:
            log.error(f"Lỗi khi đọc file {file_path}: {e}")
            continue

    s3_path_dest = f"processed/combined_raw_{timestamp}.csv"
    s3_client = get_s3_client()
    s3_client.upload_file(local_temp_file, MINIO_BUCKET, s3_path_dest)
    
    if os.path.exists(local_temp_file):
        os.remove(local_temp_file)
        
    log.info(f"Đã upload file gộp: {s3_path_dest}")
    return f"{MINIO_BUCKET}/{s3_path_dest}"

@task(dag=dag)
def preprocess_data(s3_raw_path_no_bucket: str):
    """
    🛠️ FIX 2: Xử lý trường hợp file rỗng hoặc lỗi đọc để tránh FileNotFoundError
    """
    s3_path_full = f"{s3_raw_path_no_bucket}"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_clean_file = f"/tmp/clean_{timestamp}.csv"
    
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Bắt đầu làm sạch dữ liệu từ: {s3_path_full}")
    
    first_chunk = True
    has_data = False # Cờ kiểm tra dữ liệu
    
    try:
        with pd.read_csv(f"s3://{s3_path_full}", storage_options=storage_options, chunksize=10000) as reader:
            for chunk in reader:
                has_data = True
                if 'text' in chunk.columns:
                    chunk['text'] = chunk['text'].astype(str).str.lower().str.replace(
                        r'[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
                        '', regex=True
                    )
                
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk.to_csv(local_clean_file, mode=mode, header=header, index=False)
                first_chunk = False
    except Exception as e:
        log.warning(f"Cảnh báo: Lỗi đọc file hoặc file rỗng. Chi tiết: {e}")

    # Kiểm tra file có tồn tại không trước khi upload
    if not has_data or not os.path.exists(local_clean_file):
        log.warning("Không có dữ liệu sạch được tạo ra. Bỏ qua upload.")
        return None

    s3_clean_key = f"processed/clean_data_{timestamp}.csv"
    s3_client = get_s3_client()
    s3_client.upload_file(local_clean_file, MINIO_BUCKET, s3_clean_key)
    
    if os.path.exists(local_clean_file):
        os.remove(local_clean_file)
        
    return f"{MINIO_BUCKET}/{s3_clean_key}"

@task(dag=dag)
def train_model(s3_clean_path: str):
    # Kiểm tra đầu vào từ task trước
    if not s3_clean_path:
        raise ValueError("Task train_model thất bại vì không có dữ liệu đầu vào (preprocess trả về None).")

    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Đang tải dữ liệu train từ: {s3_clean_path}")
    df = pd.read_csv(f"s3://{s3_clean_path}", storage_options=storage_options)
    
    if df.empty or len(df) < 5:
        raise ValueError("Dữ liệu quá ít để train.")

    label_map = {"positive": 1, "negative": 0, "neutral": 2}
    df['label_id'] = df['label'].map(label_map).fillna(2).astype('int8') 
    
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].astype(str), df['label_id'],
        test_size=0.2, random_state=42
    )
    
    del df
    gc.collect()

    log.info("Bắt đầu Vectorization...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    log.info("Bắt đầu Training...")
    model = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # Lưu artifacts cục bộ
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    log.info("Logging to MLflow...")

    # 🛠️ FIX 3: Đổi tên experiment và chỉ định rõ artifact_location là nexusml
    experiment_name = "sentiment_classification_nexus" 
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        log.info(f"Creating experiment '{experiment_name}' at s3://nexusml/models")
        mlflow.create_experiment(
            experiment_name,
            artifact_location="s3://nexusml/models" # <-- Quan trọng: Lưu đúng vào bucket nexusml
        )
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "LogisticRegression")
        
        # Log model (MLflow sẽ tự upload lên MinIO nexusml/models)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
        
    return {"accuracy": acc, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate_model(train_output: dict):
    acc = train_output.get("accuracy", 0)
    log.info(f"Model accuracy: {acc}")
    if acc >= 0.6:
        return "register_model"
    return "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    # 1. Khởi tạo kết nối S3 (Để backup thủ công)
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
    
    # 2. Lấy thông tin Run và Đăng ký Model
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    
    # Đăng ký vào MLflow Registry
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    log.info(f"Đã đăng ký model version: {registered.version}")

    # =========================================================
    # 3. [QUAN TRỌNG] Gán nhãn @production (Champion)
    # =========================================================
    # Bước này giúp Backend tìm được model mà không cần biết Run ID
    client = MlflowClient(MLFLOW_TRACKING_URI)
    client.set_registered_model_alias(
        name="SentimentClassifier",
        alias="production", 
        version=registered.version
    )
    log.info(f"Đã gán alias '@production' cho version {registered.version}")

    # =========================================================
    # 4. Backup thủ công sang folder version (Tùy chọn)
    # =========================================================
    # Lưu ý: MLflow đã tự lưu artifact rồi, bước này chỉ để bạn dễ nhìn trong MinIO thôi
    version_path = f"{MINIO_BUCKET}/models/sentiment/v{registered.version}"
    try:
        fs.touch(f"{version_path}/.keep")
    except:
        pass
        
    # Kiểm tra file local có tồn tại không trước khi put (tránh lỗi nếu chạy khác worker)
    if os.path.exists("/tmp/models/model.pkl"):
        fs.put("/tmp/models/model.pkl", f"{version_path}/model.pkl")
        fs.put("/tmp/models/vectorizer.pkl", f"{version_path}/vectorizer.pkl")
        log.info(f"Đã backup file thủ công vào: {version_path}")
    else:
        log.warning("Không tìm thấy file tại /tmp/models để backup (có thể do khác Worker).")

# -------------------------------
# Dependencies & Trigger Rules
# -------------------------------
notify_success = BashOperator(
    task_id='notify_success', 
    bash_command='echo "Deploy Success!"', 
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, 
    dag=dag
)

notify_fail = BashOperator(
    task_id='notify_failure', 
    bash_command='echo "Model accuracy too low!"', 
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, 
    dag=dag
)

# Flow
bucket = ensure_bucket_exists()
raw = extract_and_combine_data()
clean = preprocess_data(raw)
train = train_model(clean)
branch = evaluate_model(train)
register = register_model(train)

bucket >> raw >> clean >> train >> branch
branch >> register >> notify_success
branch >> notify_fail