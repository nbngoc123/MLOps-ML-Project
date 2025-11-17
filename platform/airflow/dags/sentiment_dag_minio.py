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
import logging

# Khởi tạo Logger
log = logging.getLogger(__name__)

# -------------------------------
# Configuration
# -------------------------------

# Lấy biến môi trường (Đã sửa để khớp với Secret Key trong K8s)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio-service.nexusml.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "") 
MINIO_BUCKET = "nexusml"

# Cấu hình MLflow & AWS Boto3 (cho MinIO)
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

# Khởi tạo filesystem object
try:
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )
except Exception as e:
    log.error(f"Không thể khởi tạo s3fs: {e}")

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
    description='Train & Deploy Sentiment Model reading ALL CSVs from MinIO',
    schedule='0 2 * * *', 
    catchup=False,
)

# -------------------------------
# Tasks
# -------------------------------
@task(dag=dag)
def ensure_bucket_exists():
    """Đảm bảo cấu trúc thư mục trên MinIO tồn tại."""
    fs.makedirs(f"{MINIO_BUCKET}/raw", exist_ok=True)
    fs.makedirs(f"{MINIO_BUCKET}/processed", exist_ok=True)
    fs.makedirs(f"{MINIO_BUCKET}/models", exist_ok=True)
    log.info(f"Đã kiểm tra/tạo các bucket trong {MINIO_BUCKET}")

@task(dag=dag)
def extract_and_combine_data():
    """
    Quét thư mục raw/, đọc TẤT CẢ file csv và gộp thành một DataFrame.
    """
    raw_path_prefix = f"{MINIO_BUCKET}/raw/"
    
    # 1. Tìm tất cả file CSV trong nexusml/raw/
    # fs.glob trả về list dạng ['nexusml/raw/file1.csv', 'nexusml/raw/file2.csv']
    all_csv_files = fs.glob(f"{raw_path_prefix}*.csv")
    
    if not all_csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {raw_path_prefix}. Hãy upload dữ liệu vào MinIO.")
    
    log.info(f"Tìm thấy {len(all_csv_files)} file CSV: {all_csv_files}")

    # 2. Đọc từng file và gộp lại
    df_list = []
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }

    for file_path in all_csv_files:
        # file_path từ glob không có 's3://', ta cần thêm vào khi dùng pandas
        log.info(f"Đang đọc file: {file_path}")
        try:
            df_temp = pd.read_csv(f"s3://{file_path}", storage_options=storage_options)
            df_list.append(df_temp)
        except Exception as e:
            log.error(f"Lỗi khi đọc file {file_path}: {e}")
            continue # Bỏ qua file lỗi

    if not df_list:
        raise ValueError("Không đọc được dữ liệu từ bất kỳ file nào.")

    # Gộp tất cả DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    log.info(f"Tổng số dòng dữ liệu sau khi gộp: {len(combined_df)}")

    # 3. Lưu file gộp tạm thời để các bước sau xử lý
    # Đặt tên file theo timestamp để tránh trùng lặp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_combined_path = f"{MINIO_BUCKET}/processed/combined_raw_{timestamp}.csv"
    
    combined_df.to_csv(f"s3://{s3_combined_path}", index=False, storage_options=storage_options)
    
    return s3_combined_path # Trả về đường dẫn file gộp

@task(dag=dag)
def preprocess_data(s3_raw_path: str):
    """
    Đọc file đã gộp, làm sạch dữ liệu và lưu file processed.
    """
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Đang tiền xử lý file: {s3_raw_path}")
    df = pd.read_csv(f"s3://{s3_raw_path}", storage_options=storage_options)
    
    # Logic làm sạch (Regex giữ lại chữ cái tiếng Việt và tiếng Anh)
    df['text'] = df['text'].str.lower().str.replace(
        r'[^a-zA-Z\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
        '', regex=True
    )
    
    # Tạo đường dẫn file output (thay raw -> processed, thêm prefix clean)
    s3_clean_path = s3_raw_path.replace("combined_raw", "clean_data")
    
    df.to_csv(f"s3://{s3_clean_path}", index=False, storage_options=storage_options)
    log.info(f"Dữ liệu sạch đã lưu tại: {s3_clean_path}")
    
    return s3_clean_path

@task(dag=dag)
def train_model(s3_clean_path: str):
    """
    Huấn luyện mô hình từ dữ liệu sạch.
    """
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
    }
    
    log.info(f"Đọc dữ liệu huấn luyện từ: {s3_clean_path}")
    df = pd.read_csv(f"s3://{s3_clean_path}", storage_options=storage_options)
    
    if df.empty:
        raise ValueError("Dữ liệu huấn luyện trống!")
    if len(df) < 5:
        raise ValueError("Dữ liệu quá ít để huấn luyện (cần > 5 dòng).")

    # Chia tập train/test
    # Map label sang số
    df['label_id'] = df['label'].map({"positive": 1, "negative": 0, "neutral": 2})
    
    # Loại bỏ các dòng bị NaN ở label (nếu có)
    df = df.dropna(subset=['label_id', 'text'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_id'],
        test_size=0.2, random_state=42
    )

    # Vectorization
    # astype('U') để đảm bảo dữ liệu là string
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train.astype('U'))
    X_test_vec = vectorizer.transform(X_test.astype('U'))

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Lưu model cục bộ tạm thời
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(vectorizer, "/tmp/models/vectorizer.pkl")
    joblib.dump(model, "/tmp/models/model.pkl")

    # MLflow Logging
    mlflow.set_experiment("sentiment_classification")
    run_name = f"minio_run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("data_source", s3_clean_path)
        
        # Log model & vectorizer lên MLflow Artifacts (sẽ đẩy sang MinIO)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/models/vectorizer.pkl", "artifacts")
        
        log.info(f"Training hoàn tất. Accuracy: {acc}")
        
    return {"accuracy": acc, "run_id": run.info.run_id}

@task(dag=dag)
def evaluate_model(train_output: dict):
    acc = train_output.get("accuracy")
    # Ngưỡng đánh giá: 0.6 (Vì dữ liệu thực tế có thể khó hơn dummy data)
    return "register_model" if acc >= 0.6 else "notify_failure"

@task(dag=dag)
def register_model(train_output: dict):
    run_id = train_output.get("run_id")
    model_uri = f"runs:/{run_id}/model"
    
    # Đăng ký model trong MLflow Registry
    registered = mlflow.register_model(model_uri, "SentimentClassifier")
    
    # Lưu bản sao thủ công sang folder models/ trên MinIO (Optional)
    version_path = f"{MINIO_BUCKET}/models/sentiment/v{registered.version}"
    fs.makedirs(version_path, exist_ok=True)
    fs.put("/tmp/models/model.pkl", f"{version_path}/model.pkl")
    fs.put("/tmp/models/vectorizer.pkl", f"{version_path}/vectorizer.pkl")
    
    log.info(f"Model v{registered.version} uploaded to MinIO: {version_path}")

# -------------------------------
# Bash notifications
# -------------------------------
notify_success = BashOperator(
    task_id='notify_success',
    bash_command='echo "Sentiment model deployed successfully!"',
    dag=dag
)
notify_fail = BashOperator(
    task_id='notify_failure',
    bash_command='echo "Model failed accuracy threshold!"',
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag
)

# -------------------------------
# DAG dependencies
# -------------------------------
bucket = ensure_bucket_exists()
raw_data_path = extract_and_combine_data() # Task mới: Quét và gộp file
clean_data_path = preprocess_data(raw_data_path)
train_results = train_model(clean_data_path)
branch = evaluate_model(train_results)
register = register_model(train_results)

bucket >> raw_data_path >> clean_data_path >> train_results >> branch
branch >> register >> notify_success
branch >> notify_fail