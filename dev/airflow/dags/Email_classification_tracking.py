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
from sklearn.metrics import accuracy_score, f1_score
import joblib
import s3fs
import boto3
import os
import mlflow
import logging
import shutil

log = logging.getLogger(__name__)

# --- CONFIG ---
DAG_ID = "email_classifier_dag"
MINIO_BUCKET = "nexusml"
USE_CASE_PREFIX = "email_classifier"
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2025, 11, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(DAG_ID, default_args=default_args, schedule='0 8 * * *', catchup=False, tags=['nlp', 'email', 'nexusml'])

# --- UTILS ---
def get_storage_options():
    return {"key": MINIO_ACCESS_KEY, "secret": MINIO_SECRET_KEY, "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}}

@task(dag=dag)
def ensure_bucket_exists():
    s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
    for bucket in [MINIO_BUCKET, "mlflow"]:
        try: s3.head_bucket(Bucket=bucket)
        except: s3.create_bucket(Bucket=bucket)
    
    # Tạo folder riêng cho email
    for folder in ["raw", "processed", "models"]:
        try: s3.put_object(Bucket=MINIO_BUCKET, Key=f"{USE_CASE_PREFIX}/{folder}/.keep", Body=b'')
        except: pass

@task(dag=dag)
def extract_data():
    """Đọc dữ liệu Email từ Raw"""
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    files = fs.glob(f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/*.csv")
    
    if not files:
        log.warning("Không có data, tạo dữ liệu giả lập...")
        df = pd.DataFrame({
            "subject": ["Win lottery now", "Help me login", "Order #123 status", "Cheap meds", "Hi team meeting today", "Free gift card inside", "Check your invoice", "New password required"],
            "text": ["Click here to win big money. Limited time offer.", "I forgot my password, please help me reset it.", "Where is my item? Tracking says shipped.", "Best pills online, guaranteed results.", "Remember to bring your reports for the discussion.", "Congratulations! You won a $100 Amazon gift card.", "Please review the attached invoice for Q3.", "We detected unusual activity. Click to change your password."],
            "label": ["spam", "support", "order", "spam", "support", "spam", "order", "support"]
        })
        path = f"/tmp/demo_email.csv"
        df.to_csv(path, index=False)
        boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)\
             .upload_file(path, MINIO_BUCKET, f"{USE_CASE_PREFIX}/raw/demo_email.csv")
        files = [f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/demo_email.csv"]

    dfs = [pd.read_csv(f"s3://{f}", storage_options=get_storage_options()) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)
    
    save_key = f"{USE_CASE_PREFIX}/processed/combined.csv"
    full_df.to_csv(f"s3://{MINIO_BUCKET}/{save_key}", index=False, storage_options=get_storage_options())
    return save_key

@task(dag=dag)
def preprocess_email(file_key: str):
    df = pd.read_csv(
        f"s3://{MINIO_BUCKET}/{file_key}", 
        storage_options=get_storage_options(),
        usecols=['text', 'label']
    )
    
    # Xử lý Data:
    df['clean_text'] = df['text'].fillna('').astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    # Map nhãn: 1 cho 'spam', 0 cho các nhãn khác (order, support)
    df['label_target'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    clean_key = f"{USE_CASE_PREFIX}/processed/clean.csv"
    df[['clean_text', 'label_target']].dropna().to_csv(
        f"s3://{MINIO_BUCKET}/{clean_key}", 
        index=False, 
        storage_options=get_storage_options()
    )
    return clean_key

@task(dag=dag)
def train_email_model(clean_key: str):
    """
    Huấn luyện mô hình Email Classifier và sử dụng partial_fit để log metric qua từng epoch (step).
    """
    df = pd.read_csv(f"s3://{MINIO_BUCKET}/{clean_key}", storage_options=get_storage_options())
    
    X = df['clean_text'].astype(str)
    y = df['label_target']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Khai báo các thành phần
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=1, # Giảm min_df vì dữ liệu giả lập nhỏ
        max_features=5000 
    )
    # Đặt max_iter=1 và tắt early_stopping để tự điều khiển vòng lặp bằng partial_fit
    classifier = SGDClassifier(
        loss='modified_huber', 
        penalty='l2', 
        alpha=1e-4, 
        max_iter=1,  
        early_stopping=False, 
        random_state=42
    )

    # 2. Fit Vectorizer và Transform data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # MLflow Tracking
    mlflow.set_experiment(f"nexus_{USE_CASE_PREFIX}")
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_param("vectorizer_ngram", "(1, 2)")
        mlflow.log_param("model_type", "SGDClassifier_ModifiedHuber_Iterative")
        
        # 3. Vòng lặp huấn luyện từng epoch và LOG MÉT-RIC VỚI STEP
        num_epochs = 10 # Số epoch muốn theo dõi
        classes = df['label_target'].unique() # Cần khai báo lớp cho partial_fit
        log.info(f"Starting iterative training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Huấn luyện cho 1 epoch
            classifier.partial_fit(X_train_vec, y_train, classes=classes)
            
            # Đánh giá trên tập huấn luyện sau mỗi epoch
            train_predictions = classifier.predict(X_train_vec)
            # LOG F1-SCORE TRÊN TẬP TRAIN VỚI STEP LÀ SỐ EPOCH
            train_f1 = f1_score(y_train, train_predictions, average='weighted', zero_division=0) 
            
            # <<< Dòng quan trọng: Log metric với step >>>
            mlflow.log_metric("train_f1_per_epoch", train_f1, step=epoch)
            # Bạn có thể đổi tên metric thành "train_loss_per_epoch" nếu bạn có thể tính được giá trị loss thực
            log.info(f"Epoch {epoch}: Train F1-Score: {train_f1:.4f}")
            
        # 4. Tạo lại Pipeline và Đánh giá cuối cùng
        final_model = make_pipeline(vectorizer, classifier)
        
        # Đánh giá trên tập test cuối cùng
        predictions = final_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Log metrics cuối cùng
        mlflow.log_metric("final_test_accuracy", acc)
        mlflow.log_metric("final_test_f1_score", f1)
        
        # 5. Log Model và Artifacts
        
        # Log Vectorizer riêng (cần cho logic backup thủ công)
        temp_dir = "/tmp/mlflow_artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        vectorizer_path = os.path.join(temp_dir, "vectorizer.pkl")
        joblib.dump(vectorizer, vectorizer_path)
        # Log dưới đường dẫn 'artifacts' để task register_model tìm thấy
        mlflow.log_artifact(vectorizer_path, "artifacts") 

        # Log toàn bộ Pipeline dưới đường dẫn 'model'
        mlflow.sklearn.log_model(final_model, "model")
        
        print(f"Training complete. Final Test Accuracy: {acc}")
        
    # Xóa folder tạm
    shutil.rmtree(temp_dir)
    return {"accuracy": acc, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate(metrics: dict):
    # Cập nhật điều kiện check accuracy cho dữ liệu giả lập (ví dụ > 0.6)
    return "register_model" if metrics['accuracy'] >= 0.6 else "notify_fail"

@task(dag=dag)
def register_model(metrics: dict):
    """
    Đăng ký MLflow và Backup an toàn.
    """
    run_id = metrics.get('run_id')
    if not run_id:
        log.warning("No run_id provided; skipping register_model")
        return

    # 1. Đăng ký và gán alias (Giữ nguyên)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    model_name = "Nexus_Email_Classifier"
    reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "production", reg.version)
    log.info("Registered and aliased version %s", reg.version)

    # 2. Thiết lập kết nối S3/MinIO và thư mục tạm
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    version_path = f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/models/v{reg.version}"
    temp_download_dir = f"/tmp/download_artifacts_v{reg.version}"
    os.makedirs(temp_download_dir, exist_ok=True)
    
    # 3. TẢI ARTIFACT TỪ MLFLOW VỀ LOCAL
    try:
        log.info(f"Downloading artifacts from run {run_id}...")
        # Tải Pipeline
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=temp_download_dir)
        # Tải Vectorizer đã log riêng
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts", dst_path=temp_download_dir)
    except Exception as e:
        log.warning(f"Could not download artifacts: {e}")
        shutil.rmtree(temp_download_dir)
        return

    # 4. BACKUP thủ công
    # Cấu trúc file: /tmp/download_artifacts_vX/model/model.pkl (Pipeline)
    local_model_path = f"{temp_download_dir}/model/model.pkl"
    # Cấu trúc file: /tmp/download_artifacts_vX/artifacts/vectorizer.pkl (Vectorizer)
    local_vectorizer_path = f"{temp_download_dir}/artifacts/vectorizer.pkl"
    
    try: fs.touch(f"{version_path}/.keep") 
    except: pass
        
    if os.path.exists(local_model_path) and os.path.exists(local_vectorizer_path):
        fs.put(local_model_path, f"{version_path}/model.pkl")
        fs.put(local_vectorizer_path, f"{version_path}/vectorizer.pkl")
        log.info(f"Đã backup model thủ công vào: {version_path}")
        
    # 5. Dọn dẹp
    if os.path.exists(temp_download_dir):
        shutil.rmtree(temp_download_dir)


# Flow
init = ensure_bucket_exists()
raw = extract_data()
clean = preprocess_email(raw)
trained = train_email_model(clean)
check = evaluate(trained)
reg = register_model(trained)
fail = BashOperator(task_id='notify_fail', bash_command='echo "Email Model Low Accuracy"', dag=dag)
success = BashOperator(task_id='notify_success', bash_command='echo "Email Model Deployed"', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, dag=dag)

init >> raw >> clean >> trained >> check
check >> reg >> success
check >> fail