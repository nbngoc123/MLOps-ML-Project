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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib
import s3fs
import boto3
import os
import mlflow
import logging
import shutil
import gc

log = logging.getLogger(__name__)

# ======================
# CONFIG
# ======================
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "nexusml")

USE_CASE_PREFIX = "sentiment_classifier"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 16),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    f"{USE_CASE_PREFIX}_dag",
    default_args=default_args,
    description=f"Train & Deploy {USE_CASE_PREFIX} model",
    schedule="0 2 * * *",
    catchup=False,
    tags=["nlp", USE_CASE_PREFIX, 'nexusml'],
)

def p(path: str) -> str:
    """prefix helper: returns <use_case>/<path>"""
    return f"{USE_CASE_PREFIX}/{path}"

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

def get_storage_options():
    return {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

# ======================
# TASKS
# ======================

@task(dag=dag)
def ensure_bucket_exists():
    s3_client = get_s3_client()
    # create the bucket(s) if missing
    for bucket in [MINIO_BUCKET, "mlflow"]:
        try:
            s3_client.head_bucket(Bucket=bucket)
            log.info("Bucket %s exists", bucket)
        except Exception:
            log.info("Creating bucket %s", bucket)
            try:
                s3_client.create_bucket(Bucket=bucket)
            except Exception as e:
                log.warning("Could not create bucket %s: %s", bucket, e)
    # create placeholder keys for the use-case prefix
    for folder in ["raw", "processed", "models"]:
        key = f"{p(folder)}/.keep"
        try:
            s3_client.put_object(Bucket=MINIO_BUCKET, Key=key, Body=b"")
        except Exception as e:
            log.warning("Unable to create placeholder %s: %s", key, e)

@task(dag=dag)
def extract_and_combine_data():
    """
    Read all CSVs under nexusml/<usecase>/raw/*.csv, append to a local temp file
    and upload combined file to nexusml/<usecase>/processed/combined_{ts}.csv

    Note: returns a 'key' (without bucket) like:
      sentiment_classifier/processed/combined_2025...
    """
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )

    raw_prefix = f"{MINIO_BUCKET}/{p('raw')}"
    files = fs.glob(f"{raw_prefix}/*.csv")
    if not files:
        # create a small mock example to avoid pipeline failing silently
        tmp = "/tmp/demo_sentiment.csv"
        pd.DataFrame({
            "text": ["good job", "bad luck", "excellent work", "terrible experience"],
            "label": ["positive", "negative", "positive", "negative"]
        }).to_csv(tmp, index=False)
        s3 = get_s3_client()
        dest_key = f"{p('raw')}/demo_sentiment.csv"
        s3.upload_file(tmp, MINIO_BUCKET, dest_key)
        files = [f"{MINIO_BUCKET}/{dest_key}"]
        os.remove(tmp)
        log.info("Uploaded demo file to %s/%s", MINIO_BUCKET, dest_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_temp = f"/tmp/combined_{timestamp}.csv"
    storage_options = get_storage_options()

    first = True
    for s3_file in files:
        # s3_file looks like "nexusml/sentiment_classifier/raw/xxx.csv" (because we used fs.glob)
        # convert to s3://{MINIO_BUCKET}/{path_after_bucket}
        # ensure we use path part after first slash (bucket/...)
        try:
            # build read path for pandas
            read_path = f"s3://{s3_file}"
            # use iterator when file large
            reader = pd.read_csv(read_path, storage_options=storage_options, chunksize=10000)
            for chunk in reader:
                mode = "w" if first else "a"
                header = first
                chunk.to_csv(local_temp, mode=mode, header=header, index=False)
                first = False
        except Exception as e:
            log.error("Error reading %s: %s", s3_file, e)
            continue

    if not os.path.exists(local_temp):
        raise ValueError("No combined file produced (no input files or read failed).")

    combined_key = p(f"processed/combined_raw_{timestamp}.csv")
    s3 = get_s3_client()
    s3.upload_file(local_temp, MINIO_BUCKET, combined_key)
    os.remove(local_temp)
    log.info("Uploaded combined file to %s/%s", MINIO_BUCKET, combined_key)
    # return key only (not including bucket)
    return combined_key

@task(dag=dag)
def preprocess_data(s3_combined_key: str):
    """
    Read combined file s3://{MINIO_BUCKET}/{s3_combined_key} in chunks, clean text,
    write cleaned CSV to s3://{MINIO_BUCKET}/{USE_CASE_PREFIX}/processed/clean_data_{ts}.csv
    Returns clean_key (without bucket).
    """
    if not s3_combined_key:
        log.warning("preprocess_data: no input key received")
        return None

    storage_options = get_storage_options()
    read_path = f"s3://{MINIO_BUCKET}/{s3_combined_key}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_clean = f"/tmp/clean_{timestamp}.csv"

    has_data = False
    first = True
    try:
        reader = pd.read_csv(read_path, storage_options=storage_options, chunksize=10000)
        for chunk in reader:
            has_data = True
            if "text" in chunk.columns:
                chunk["text"] = chunk["text"].astype(str).str.lower().str.replace(
                    r"[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]",
                    "",
                    regex=True,
                )
            mode = "w" if first else "a"
            header = first
            chunk.to_csv(local_clean, mode=mode, header=header, index=False)
            first = False
    except Exception as e:
        log.warning("Warning reading combined file: %s", e)

    if not has_data or not os.path.exists(local_clean):
        log.warning("No cleaned data produced.")
        return None

    clean_key = p(f"processed/clean_data_{timestamp}.csv")
    s3 = get_s3_client()
    s3.upload_file(local_clean, MINIO_BUCKET, clean_key)
    os.remove(local_clean)
    log.info("Uploaded clean file to %s/%s", MINIO_BUCKET, clean_key)
    return clean_key

@task(dag=dag)
def train_model(s3_clean_key: str):
    """
    Train model; returns dict {"accuracy": float, "run_id": run_id or None}
    If training can't run (not enough classes), returns accuracy 0 and run_id=None
    """
    if not s3_clean_key:
        raise ValueError("No cleaned input provided to train_model")

    storage_options = get_storage_options()
    read_path = f"s3://{MINIO_BUCKET}/{s3_clean_key}"
    log.info("Loading training data from %s", read_path)

    df = pd.read_csv(read_path, storage_options=storage_options)
    if df is None or df.empty or len(df) < 3:
        log.warning("Not enough data to train (rows=%s).", 0 if df is None else len(df))
        return {"accuracy": 0.0, "run_id": None}

    # Ensure columns exist and use the clean text column if present
    text_col = "text" if "text" in df.columns else ("clean_text" if "clean_text" in df.columns else None)
    if text_col is None:
        raise ValueError("No text column found in training data")

    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        raise ValueError("No label column found in training data")

    # quick label sanity
    label_counts = df[label_col].value_counts()
    log.info("Label distribution before mapping: %s", label_counts.to_dict())

    # Map labels to ints if needed — we'll keep string labels for sklearn too
    # but keep a label_map for later if needed
    label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    # If you expect many label names, adjust mapping logic — here we keep original labels.

    # Check number of classes
    n_classes = df[label_col].nunique()
    if n_classes < 2:
        log.warning("Only %s unique class(es) found: %s. Skipping training.", n_classes, df[label_col].unique())
        return {"accuracy": 0.0, "run_id": None}

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str), df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    # free memory
    del df
    gc.collect()

    # build pipeline and train
    pipeline = make_pipeline(TfidfVectorizer(max_features=3000), LogisticRegression(max_iter=1000))
    log.info("Starting fit on %d samples", len(X_train))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="weighted"))

    # save local artifacts (optional)
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(pipeline.named_steps["tfidfvectorizer"], "/tmp/models/vectorizer.pkl")
    joblib.dump(pipeline.named_steps["logisticregression"], "/tmp/models/model.pkl")
    # Also save whole pipeline for convenience
    joblib.dump(pipeline, "/tmp/models/pipeline.pkl")

    # MLflow logging: ensure experiment points to use-case models folder
    experiment_name = f"nexus_{USE_CASE_PREFIX}"
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"s3://{MINIO_BUCKET}/{p('models')}"
        )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
    run_id = run.info.run_id
    log.info("Finished MLflow run %s (acc=%s)", run_id, acc)

    return {"accuracy": acc, "run_id": run_id}

@task.branch(dag=dag)
def evaluate_model(metrics: dict):
    acc = metrics.get("accuracy", 0.0)
    log.info("Model accuracy: %s", acc)
    return "register_model" if acc >= 0.6 and metrics.get("run_id") else "notify_failure"

@task(dag=dag)
def register_model(metrics: dict):
    """
    Đăng ký mô hình vào MLflow Model Registry và gán alias 'production'.
    Loại bỏ logic backup thủ công lên MinIO/S3.
    """
    log = logging.getLogger(__name__)

    run_id = metrics.get("run_id")
    if not run_id:
        log.warning("No run_id provided; skipping register_model")
        return

    # 1. Khởi tạo Client và xác định URI mô hình
    client = MlflowClient(MLFLOW_TRACKING_URI)
    model_name = f"{USE_CASE_PREFIX}_Model"
    model_uri = f"runs:/{run_id}/model"

    # 2. Đăng ký Mô hình vào MLflow Registry
    try:
        registered = mlflow.register_model(model_uri, model_name)
        log.info("Registered model name=%s version=%s", model_name, registered.version)
    except Exception as e:
        log.error(f"Error registering model: {e}")
        return

    # 3. Gán alias 'production' cho phiên bản vừa đăng ký
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=registered.version
        )
        log.info("Set alias 'production' for version %s", registered.version)
    except Exception as e:
        log.error(f"Error setting 'production' alias: {e}")




# ======================
# FINAL FLOW wiring
# ======================
init = ensure_bucket_exists()
raw = extract_and_combine_data()
clean = preprocess_data(raw)
trained = train_model(clean)
check = evaluate_model(trained)
reg = register_model(trained)

notify_success = BashOperator(
    task_id="notify_success",
    bash_command='echo "Deploy Success!"',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

notify_fail = BashOperator(
    task_id="notify_failure",
    bash_command='echo "Model accuracy too low or training skipped!"',
    dag=dag,
)

init >> raw >> clean >> trained >> check
check >> reg >> notify_success
check >> notify_fail
