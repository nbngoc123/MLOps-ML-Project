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
import gc 

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
    try:
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
        log.info(f"Bucket {MINIO_BUCKET} đã tồn tại.")
    except Exception:
        log.info(f"Bucket {MINIO_BUCKET} chưa tồn tại. Đang tạo...")
        s3_client.create_bucket(Bucket=MINIO_BUCKET)
    
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
    Sau đó upload file kết quả lên MinIO.
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
    
    # Cấu hình đọc S3
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

    first_file = True
    
    for file_path in all_csv_files:
        log.info(f"Đang gộp file: {file_path}")
        # Đọc theo chunk (mỗi lần 10k dòng) để tránh tràn RAM
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

    # Upload kết quả lên MinIO
    s3_path_dest = f"processed/combined_raw_{timestamp}.csv"
    s3_client = get_s3_client()
    s3_client.upload_file(local_temp_file, MINIO_BUCKET, s3_path_dest)
    
    # Dọn dẹp file tạm
    if os.path.exists(local_temp_file):
        os.remove(local_temp_file)
        
    log.info(f"Đã upload file gộp: {s3_path_dest}")
    return f"{MINIO_BUCKET}/{s3_path_dest}"

@task(dag=dag)
def preprocess_data(s3_raw_path_no_bucket: str):
    """
    Phiên bản Memory-Safe: Đọc file gộp theo Chunk, làm sạch và ghi lại vào file mới.
    """
    s3_path_full = f"{s3_raw_path_no_bucket}" # Task trước trả về dạng bucket/path
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_clean_file = f"/tmp/clean_{timestamp}.csv"
    
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Bắt đầu làm sạch dữ liệu từ: {s3_path_full}")
    
    first_chunk = True
    
    # Đọc file lớn theo chunk (ví dụ 10,000 dòng 1 lần)
    with pd.read_csv(f"s3://{s3_path_full}", storage_options=storage_options, chunksize=10000) as reader:
        for chunk in reader:
            # Logic làm sạch
            if 'text' in chunk.columns:
                chunk['text'] = chunk['text'].astype(str).str.lower().str.replace(
                    r'[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
                    '', regex=True
                )
            
            # Ghi ra file tạm
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(local_clean_file, mode=mode, header=header, index=False)
            first_chunk = False
            
    # Upload file sạch lên MinIO
    s3_clean_key = f"processed/clean_data_{timestamp}.csv"
    s3_client = get_s3_client()
    s3_client.upload_file(local_clean_file, MINIO_BUCKET, s3_clean_key)
    
    if os.path.exists(local_clean_file):
        os.remove(local_clean_file)
        
    return f"{MINIO_BUCKET}/{s3_clean_key}"

@task(dag=dag)
def train_model(s3_clean_path: str):
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info("Đang tải dữ liệu train...")
    # Đọc toàn bộ file clean (nếu file quá lớn > RAM, cần chuyển sang dùng SGDClassifier + partial_fit)
    # Ở đây ta giả định sau khi clean, dữ liệu đủ vừa RAM, nhưng ta dùng dtype tối ưu
    df = pd.read_csv(f"s3://{s3_clean_path}", storage_options=storage_options)
    
    if df.empty or len(df) < 5:
        raise ValueError("Dữ liệu quá ít để train.")

    # Map label và tối ưu bộ nhớ
    label_map = {"positive": 1, "negative": 0, "neutral": 2}
    df['label_id'] = df['label'].map(label_map).fillna(2).astype('int8') 
    
    # Drop cột không cần thiết để giải phóng RAM
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].astype(str), df['label_id'],
        test_size=0.2, random_state=42
    )
    
    # Xóa df gốc để giải phóng RAM cho quá trình vectorization
    del df
    gc.collect()

    log.info("Bắt đầu Vectorization...")
    vectorizer = TfidfVectorizer(max_features=3000) # Giới hạn feature để tiết kiệm RAM
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    log.info("Bắt đầu Training...")
    model = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1) # n_jobs=-1 dùng full CPU core
    model.fit(X_train_vec, y_train)

    # Lưu artifacts
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    log.info("Logging to MLflow...")
    mlflow.set_experiment("sentiment_classification")
    with mlflow.start_run(run_name=f"minio_run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "LogisticRegression")
        
        # Log model trực tiếp
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
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    
    version_path = f"{MINIO_BUCKET}/models/sentiment/v{registered.version}"
    # Tạo thư mục đích (nếu cần thiết với s3fs)
    try:
        fs.touch(f"{version_path}/.keep")
    except:
        pass
        
    fs.put("/tmp/models/model.pkl", f"{version_path}/model.pkl")
    fs.put("/tmp/models/vectorizer.pkl", f"{version_path}/vectorizer.pkl")
    log.info(f"Đã lưu model version {registered.version} vào MinIO")

# -------------------------------
# Dependencies & Trigger Rules
# -------------------------------
notify_success = BashOperator(
    task_id='notify_success', 
    bash_command='echo "Deploy Success!"', 
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, # Chạy nếu register thành công
    dag=dag
)

notify_fail = BashOperator(
    task_id='notify_failure', 
    bash_command='echo "Model accuracy too low or Task Failed!"', 
    trigger_rule=TriggerRule.ONE_FAILED, # Chạy nếu có bất kỳ task nào fail hoặc branch trả về fail
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