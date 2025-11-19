from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import s3fs
import boto3
import os
import mlflow
import logging
import shutil

log = logging.getLogger(__name__)
# --- CONFIG ---
DAG_ID = "topic_classifier_dag"
MINIO_BUCKET = "nexusml"
USE_CASE_PREFIX = "topic_classifier"
# (Giữ nguyên các biến môi trường MinIO/MLflow như file trên)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

default_args = {'owner': 'ml-team', 'start_date': datetime(2025, 11, 16), 'retries': 1}
dag = DAG(DAG_ID, default_args=default_args, schedule='0 9 * * *', catchup=False, tags=['nlp', 'topic', 'nexusml'])

def get_storage_options():
    return {"key": MINIO_ACCESS_KEY, "secret": MINIO_SECRET_KEY, "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}}

@task(dag=dag)
def ensure_bucket_exists():
    s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
    try: s3.create_bucket(Bucket=MINIO_BUCKET)
    except: pass
    for folder in ["raw", "processed", "models"]:
        try: s3.put_object(Bucket=MINIO_BUCKET, Key=f"{USE_CASE_PREFIX}/{folder}/.keep", Body=b'')
        except: pass

@task(dag=dag)
def extract_data():
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    files = fs.glob(f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/*.csv")
    
    if not files:
        # Mock Data cho Topic
        df = pd.DataFrame({
            "text": ["Stock market crashed", "New iPhone released", "Football world cup", "Inflation rate up"],
            "label": ["business", "tech", "sports", "business"]
        })
        path = "/tmp/demo_topic.csv"
        df.to_csv(path, index=False)
        boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)\
             .upload_file(path, MINIO_BUCKET, f"{USE_CASE_PREFIX}/raw/demo_topic.csv")
        files = [f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/demo_topic.csv"]

    dfs = [pd.read_csv(f"s3://{f}", storage_options=get_storage_options()) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)
    save_key = f"{USE_CASE_PREFIX}/processed/combined.csv"
    full_df.to_csv(f"s3://{MINIO_BUCKET}/{save_key}", index=False, storage_options=get_storage_options())
    return save_key

# ----------------------------
# Task 1: Preprocess data
# ----------------------------
@task(dag=dag)
def preprocess_topic(file_key: str):
    # đọc CSV từ S3
    df = pd.read_csv(f"s3://{MINIO_BUCKET}/{file_key}", storage_options=get_storage_options())

    # làm sạch text: lowercase + strip khoảng trắng
    df['clean_text'] = df['text'].astype(str).str.lower().str.strip()

    # lưu lại cột cần thiết
    clean_key = f"{USE_CASE_PREFIX}/processed/clean.csv"
    df[['clean_text', 'topic']].to_csv(
        f"s3://{MINIO_BUCKET}/{clean_key}",
        index=False,
        storage_options=get_storage_options()
    )
    return clean_key

# ----------------------------
# Task 2: Train topic model
# ----------------------------
@task(dag=dag)
def train_topic_model(clean_key: str):
    # 1. Đọc Parquet
    path = f"s3://{MINIO_BUCKET}/{clean_key}"
    df = pd.read_parquet(path, storage_options=get_storage_options())

    # 2. Safety Check (Copy từ hàm trên)
    if len(df) < 5 or df['topic'].nunique() < 2:
        log.warning("Not enough data/classes for Topic model.")
        return {"accuracy": 0.0, "run_id": None}

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'].astype(str), df['topic'], 
        test_size=0.2, random_state=42, stratify=df['topic']
    )

    # 4. Pipeline (SGDClassifier thường nhanh và tốt cho text classification)
    model = make_pipeline(
        # max_features=5000 là đủ cho các bài toán topic đơn giản
        TfidfVectorizer(max_features=5000), 
        SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)

    # 5. Log MLflow
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.set_experiment(f"nexus_{USE_CASE_PREFIX}")
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        # Register model
        mlflow.register_model(f"runs:/{run.info.run_id}/model", TOPIC_MODEL_NAME)

    return {"accuracy": acc, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate(metrics: dict):
    return "register_model" if metrics['accuracy'] >= 0.6 else "notify_fail"

@task(dag=dag)
def register_model(metrics: dict):
    """Đăng ký mô hình và FIX lỗi backup thủ công bằng cách tải Artifact về trước."""
    
    run_id = metrics.get('run_id')
    if not run_id:
        log.warning("No run_id provided; skipping register_model")
        return

    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    model_name = "Nexus_Topic_Classifier"
    
    # 1. Đăng ký mô hình và gán alias (Giữ nguyên)
    reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "production", reg.version)
    log.info("Registered and aliased version %s", reg.version)

    # 2. Bắt đầu FIX LỖI BACKUP
    
    # Tạo thư mục tạm thời để chứa file tải về
    temp_download_dir = f"/tmp/download_artifacts_v{reg.version}"
    os.makedirs(temp_download_dir, exist_ok=True)
    
    # Tải Artifacts từ MLflow Run về worker hiện tại
    try:
        # Tải folder 'model' (chứa model.pkl)
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=temp_download_dir)
        # Tải folder 'artifacts' (chứa vectorizer.pkl) - Giả định vectorizer nằm ở artifact path này
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts", dst_path=temp_download_dir)
    except Exception as e:
        log.warning(f"Could not download artifacts from MLflow Run: {e}")
        shutil.rmtree(temp_download_dir)
        return

    # Đường dẫn file cục bộ sau khi tải
    local_model_path = f"{temp_download_dir}/model/model.pkl"
    local_vectorizer_path = f"{temp_download_dir}/artifacts/vectorizer.pkl" 
    
    # 3. Backup thủ công lên MinIO (Sử dụng file vừa tải về)
    version_path = f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/models/v{reg.version}"
    
    # Đảm bảo thư mục tồn tại
    try: fs.touch(f"{version_path}/.keep") 
    except: pass
        
    if os.path.exists(local_model_path):
        fs.put(local_model_path, f"{version_path}/model.pkl")
        fs.put(local_vectorizer_path, f"{version_path}/vectorizer.pkl")
        log.info(f"Đã backup model thủ công vào: {version_path}")
        
    # 4. Dọn dẹp
    if os.path.exists(temp_download_dir):
        shutil.rmtree(temp_download_dir)

# Flow
init = ensure_bucket_exists()
raw = extract_data()
clean = preprocess_topic(raw)
trained = train_topic_model(clean)
check = evaluate(trained)
reg = register_model(trained)
fail = BashOperator(task_id='notify_fail', bash_command='echo "Topic Model Failed"', dag=dag)
success = BashOperator(task_id='notify_success', bash_command='echo "Topic Model Deployed"', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, dag=dag)

init >> raw >> clean >> trained >> check
check >> reg >> success
check >> fail