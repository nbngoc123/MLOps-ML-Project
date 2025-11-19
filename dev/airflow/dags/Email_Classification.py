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
            "subject": ["Win lottery now", "Help me login", "Order #123 status", "Cheap meds"],
            "text": ["Click here to win big money", "I forgot my password", "Where is my item?", "Best pills online"],
            "label": ["spam", "support", "order", "spam"]
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
    # Chỉ đọc các cột cần thiết để tiết kiệm bộ nhớ
    # Giả sử CSV có header là: index, label, text, label_num
    df = pd.read_csv(
        f"s3://{MINIO_BUCKET}/{file_key}", 
        storage_options=get_storage_options(),
        usecols=['text', 'label'] # Chỉ lấy 2 cột này theo yêu cầu
    )
    
    # Xử lý Data:
    # 1. Điền giá trị rỗng
    # 2. Chuyển về chữ thường
    # 3. Loại bỏ ký tự đặc biệt (giữ lại từ và khoảng trắng)
    df['clean_text'] = df['text'].fillna('').astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    # Map nhãn label từ chữ sang số nếu cần (ví dụ ham=0, spam=1)
    # Nếu cột label đã là số hoặc bạn muốn giữ nguyên string thì bỏ qua dòng này
    df['label_target'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Lưu file đã xử lý
    clean_key = f"{USE_CASE_PREFIX}/processed/clean.csv"
    
    # Chỉ lưu text đã sạch và nhãn mục tiêu
    df[['clean_text', 'label_target']].dropna().to_csv(
        f"s3://{MINIO_BUCKET}/{clean_key}", 
        index=False, 
        storage_options=get_storage_options()
    )
    return clean_key

@task(dag=dag)
def train_email_model(clean_key: str):
    df = pd.read_csv(f"s3://{MINIO_BUCKET}/{clean_key}", storage_options=get_storage_options())
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'].astype(str), 
        df['label_target'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_target'] # Đảm bảo tỉ lệ spam/ham đều nhau giữa tập train và test
    )

    # Cấu hình Model (Đã tinh chỉnh)
    model = make_pipeline(
        # 1. Vectorizer: Nâng cấp khả năng đọc hiểu văn bản
        TfidfVectorizer(
            stop_words='english',     # Loại bỏ từ vô nghĩa tiếng Anh (the, a, an...)
            ngram_range=(1, 2),       # Học cả từ đơn và từ ghép (ví dụ: "free money" sẽ bị bắt thay vì chỉ "money")
            min_df=5,                 # Bỏ qua các từ xuất hiện quá ít (nhiễu)
            max_features=5000         # Giới hạn 5000 đặc trưng quan trọng nhất để model nhẹ
        ),
        # 2. Classifier: SGD (Linear SVM) tối ưu cho dữ liệu lớn
        SGDClassifier(
            loss='modified_huber',    # Loss function này chịu nhiễu tốt và hỗ trợ predict_proba
            penalty='l2',             # Regularization để tránh Overfitting
            alpha=1e-4,               # Hệ số phạt
            max_iter=1000,            # Số vòng lặp tối đa
            early_stopping=True,      # Dừng sớm nếu model không học thêm được gì (tiết kiệm resource)
            random_state=42
        )
    )
    
    # Training
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # MLflow Tracking
    mlflow.set_experiment(f"nexus_{USE_CASE_PREFIX}")
    with mlflow.start_run() as run:
        # Log params để sau này biết mình đã config gì
        mlflow.log_param("vectorizer_ngram", "(1, 2)")
        mlflow.log_param("model_type", "SGDClassifier_ModifiedHuber")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training complete. Accuracy: {acc}")
        
    return {"accuracy": acc, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate(metrics: dict):
    return "register_model" if metrics['accuracy'] >= 0.7 else "notify_fail"

@task(dag=dag)
def register_model(metrics: dict):
    """
    Code gốc của bạn, đã được tối ưu để Đăng ký MLflow và Backup an toàn.
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
    
    # 3. TẢI ARTIFACT TỪ MLFLOW VỀ LOCAL (Đảm bảo file tồn tại)
    try:
        log.info(f"Downloading artifacts from run {run_id}...")
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=temp_download_dir)
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts", dst_path=temp_download_dir)
    except Exception as e:
        log.warning(f"Could not download artifacts: {e}")
        shutil.rmtree(temp_download_dir)
        return

    # 4. BACKUP thủ công (Dùng file vừa tải về)
    local_model_path = f"{temp_download_dir}/model/model.pkl"
    local_vectorizer_path = f"{temp_download_dir}/artifacts/vectorizer.pkl"
    
    try: fs.touch(f"{version_path}/.keep") 
    except: pass
        
    if os.path.exists(local_model_path):
        fs.put(local_model_path, f"{version_path}/model.pkl")
        fs.put(local_vectorizer_path, f"{version_path}/vectorizer.pkl")
        log.info(f"Đã backup model thủ công vào: {version_path}")
        
    # 5. Dọn dẹp
    if os.path.exists(temp_download_dir):
        shutil.rmtree(temp_download_dir)


# def register_model_simplified(metrics: dict):
#     """
#     Chỉ đăng ký và gán alias trong MLflow. Loại bỏ logic backup thủ công.
#     """
#     run_id = metrics.get('run_id')
#     if not run_id:
#         log.warning("No run_id provided; skipping register_model")
#         return

#     client = MlflowClient(MLFLOW_TRACKING_URI)
#     model_name = "Nexus_Email_Classifier"
    
#     # 1. Đăng ký mô hình
#     reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
#     log.info("Registered model name=%s version=%s", model_name, reg.version)

#     # 2. Gán alias 'production'
#     client.set_registered_model_alias(
#         name=model_name, 
#         alias="production", 
#         version=reg.version
#     )
#     log.info("Set alias 'production' for version %s", reg.version)

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