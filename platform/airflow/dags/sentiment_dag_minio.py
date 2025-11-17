from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import s3fs
import os
import mlflow

# -------------------------------
# Configuration
# -------------------------------

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio-service.nexusml.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "") 
MINIO_BUCKET = "nexusml"

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': MINIO_ENDPOINT},
    key=MINIO_ACCESS_KEY,
    secret=MINIO_SECRET_KEY
)

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

# -------------------------------
# DAG definition
# -------------------------------
dag = DAG(
    'sentiment_classification_minio_optimized',
    default_args=default_args,
    description='Train & Deploy Sentiment Model with MinIO and MLflow',
    schedule_interval='0 2 * * *',
    catchup=False,
)

# -------------------------------
# Tasks
# -------------------------------
@task(dag=dag)
def ensure_bucket_exists():
    fs.makedirs(f"{MINIO_BUCKET}/raw", exist_ok=True)
    fs.makedirs(f"{MINIO_BUCKET}/processed", exist_ok=True)
    fs.makedirs(f"{MINIO_BUCKET}/models", exist_ok=True)
    print(f"Buckets ensured under {MINIO_BUCKET}")

@task(dag=dag)
def extract_comments():
    data = [
        {"text": "Sản phẩm rất tốt, giao nhanh", "label": "positive"},
        {"text": "Hàng lỗi, không hài lòng", "label": "negative"},
        {"text": "Ổn, giá hợp lý", "label": "neutral"},
    ]
    df = pd.DataFrame(data * 500)
    s3_path = f"{MINIO_BUCKET}/raw/comments_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(f"s3://{s3_path}", index=False, storage_options={
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    })
    return s3_path

@task(dag=dag)
def preprocess_data(s3_raw: str):
    df = pd.read_csv(f"s3://{s3_raw}", storage_options={
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    })
    df['text'] = df['text'].str.lower().str.replace(
        r'[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
        '', regex=True
    )
    s3_clean = s3_raw.replace("raw", "processed")
    df.to_csv(f"s3://{s3_clean}", index=False, storage_options={
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    })
    return s3_clean

@task(dag=dag)
def train_model(s3_clean: str):
    df = pd.read_csv(f"s3://{s3_clean}", storage_options={
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    })
    if df.empty:
        raise ValueError("No data to train!")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'].map({"positive": 1, "negative": 0, "neutral": 2}),
        test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    mlflow.set_experiment("sentiment_classification")
    with mlflow.start_run(run_name=f"minio_run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
    return {"accuracy": acc, "run_id": run.info.run_id}

@task(dag=dag)
def evaluate_model(train_output: dict):
    acc = train_output.get("accuracy")
    return "register_model" if acc >= 0.85 else "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, "SentimentClassifier")

    version_path = f"{MINIO_BUCKET}/models/sentiment/v{registered.version}"
    fs.makedirs(version_path, exist_ok=True)
    fs.put("/tmp/models/model.pkl", f"{version_path}/model.pkl")
    fs.put("/tmp/models/vectorizer.pkl", f"{version_path}/vectorizer.pkl")
    print(f"Model v{registered.version} uploaded to MinIO: {version_path}")

# -------------------------------
# Bash notifications
# -------------------------------
notify_success = BashOperator(
    task_id='notify_success',
    bash_command='echo "Sentiment model deployed!"',
    dag=dag
)
notify_fail = BashOperator(
    task_id='notify_failure',
    bash_command='echo "Model failed threshold!"',
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag
)

# -------------------------------
# DAG dependencies
# -------------------------------
bucket = ensure_bucket_exists()
raw = extract_comments()
clean = preprocess_data(raw)
train = train_model(clean)
branch = evaluate_model(train)
register = register_model(train)

bucket >> raw >> clean >> train >> branch
branch >> register >> notify_success
branch >> notify_fail
