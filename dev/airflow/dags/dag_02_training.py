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
import os
import mlflow
import logging
import gc
from minio import Minio

log = logging.getLogger(__name__)

# --- CONFIG ---
MINIO_ENDPOINT_HOST = os.environ.get("MINIO_ENDPOINT", "minio:9000").replace("http://", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

# MLflow Config
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_ENDPOINT_HOST}"
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

def get_minio_client():
    return Minio(MINIO_ENDPOINT_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

dag = DAG(
    '02_model_training_minio',
    default_args={'owner': 'ml-team', 'start_date': datetime(2025, 11, 16)},
    schedule=None, # Chạy khi được trigger
    catchup=False,
    description="ML Pipeline: Train -> Evaluate -> Register"
)

@task(dag=dag)
def train_model(**context):
    # Lấy đường dẫn file từ DAG 1 gửi sang thông qua dag_run.conf
    dag_run_conf = context['dag_run'].conf or {}
    s3_clean_key = dag_run_conf.get('s3_clean_key')
    
    if not s3_clean_key:
        raise ValueError("Không nhận được s3_clean_key từ Trigger!")

    log.info(f"Nhận tín hiệu train từ file: {s3_clean_key}")
    
    client = get_minio_client()
    response = None
    try:
        response = client.get_object(MINIO_BUCKET, s3_clean_key)
        df = pd.read_csv(response)
    finally:
        if response: response.close()

    # Xử lý dữ liệu và Train (Logic giữ nguyên)
    df['label_id'] = df['label'].map({"positive": 1, "negative": 0, "neutral": 2}).fillna(2).astype('int8')
    df = df.drop(columns=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'].astype(str), df['label_id'], test_size=0.2)
    del df
    gc.collect()

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # Lưu artifacts
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    # MLflow Tracking
    mlflow.set_experiment("sentiment_classification")
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
        
    return {"accuracy": acc, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate_model(train_output: dict):
    if train_output['accuracy'] >= 0.6:
        return "register_model"
    return "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    client = get_minio_client()
    run_id = train_output["run_id"]
    
    # Register MLflow
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    
    # Backup to MinIO
    version_path = f"models/sentiment/v{registered.version}"
    client.fput_object(MINIO_BUCKET, f"{version_path}/model.pkl", "/tmp/models/model.pkl")
    client.fput_object(MINIO_BUCKET, f"{version_path}/vectorizer.pkl", "/tmp/models/vectorizer.pkl")

# --- FLOW ---
train = train_model()
branch = evaluate_model(train)
register = register_model(train)

success = BashOperator(task_id='notify_success', bash_command='echo "Deployed!"', trigger_rule='none_failed_min_one_success', dag=dag)
fail = BashOperator(task_id='notify_failure', bash_command='echo "Failed!"', trigger_rule='one_failed', dag=dag)

train >> branch
branch >> register >> success
branch >> fail