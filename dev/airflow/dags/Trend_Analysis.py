from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import s3fs
import boto3
import os
import mlflow
import logging

log = logging.getLogger(__name__)

# --- CONFIG ---
DAG_ID = "trend_analysis_inference_dag"
MINIO_BUCKET = "nexusml"
USE_CASE_PREFIX = "trend_analysis"

# Cấu hình tên Model trong MLflow Registry
SENTIMENT_MODEL_NAME = "sentiment_classifier_Model"
TOPIC_MODEL_NAME = "Nexus_Topic_Classifier"
MODEL_ALIAS = "production"

# Cấu hình Environment (Quan trọng để MLflow load được model từ MinIO)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Set Environment Variables cho Worker
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

default_args = {'owner': 'ml-team', 'start_date': datetime(2024, 1, 1), 'retries': 1}
dag = DAG(DAG_ID, default_args=default_args, schedule='0 12 * * *', catchup=False, tags=['inference', 'trends', 'nexusml'])

def get_storage_options():
    return {"key": MINIO_ACCESS_KEY, "secret": MINIO_SECRET_KEY, "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}}

@task(dag=dag)
def ensure_bucket_exists():
    s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
    try: s3.create_bucket(Bucket=MINIO_BUCKET)
    except: pass
    for folder in ["raw", "processed", "gold"]:
        try: s3.put_object(Bucket=MINIO_BUCKET, Key=f"{USE_CASE_PREFIX}/{folder}/.keep", Body=b'')
        except: pass

@task(dag=dag)
def extract_data():
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    files = fs.glob(f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/*.csv")
    
    if not files:
        # Mock Data: Feedback khách hàng giả lập
        df = pd.DataFrame({
            "comment": [
                "Pin yếu quá dùng nửa ngày hết", "Pin tụt nhanh kinh khủng", "Thời lượng pin kém",
                "Giao hàng nhanh vãi", "Ship siêu tốc", "Đóng gói cẩn thận, giao lẹ",
                "Màn hình đẹp sắc nét", "Màn hình hiển thị tốt", "Màu sắc màn hình tươi",
                "Máy nóng quá", "Dùng xíu là nóng", "Nhiệt độ máy cao"
            ],
            # Giả lập dữ liệu cho 3 ngày khác nhau
            "date": [datetime.now() - timedelta(days=i % 3) for i in range(12)]
        })
        path = "/tmp/demo_trends.csv"
        df.to_csv(path, index=False)
        boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)\
             .upload_file(path, MINIO_BUCKET, f"{USE_CASE_PREFIX}/raw/demo_trends.csv")
        files = [f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/demo_trends.csv"]

    dfs = [pd.read_csv(f"s3://{f}", storage_options=get_storage_options()) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Lưu Parquet cho tối ưu
    save_key = f"{USE_CASE_PREFIX}/processed/combined.parquet"
    full_df.to_parquet(f"s3://{MINIO_BUCKET}/{save_key}", index=False, storage_options=get_storage_options())
    return save_key

@task(dag=dag)
def preprocess_trends(file_key: str):
    # Đọc file (Giả sử input là CSV hoặc Parquet từ bước trước)
    # Lưu ý: Nếu bước extract lưu là csv thì dùng read_csv, parquet thì read_parquet
    path = f"s3://{MINIO_BUCKET}/{file_key}"
    
    if file_key.endswith('.csv'):
        df = pd.read_csv(path, storage_options=get_storage_options())
    else:
        df = pd.read_parquet(path, storage_options=get_storage_options())
    
    # --- 1. XỬ LÝ DATE (ISO 8601 có 'Z') ---
    # Input: "2016-03-08T20:21:53Z"
    # utc=True: Bắt buộc để pandas hiểu 'Z' là múi giờ UTC
    # errors='coerce': Nếu dòng nào lỗi định dạng date thì biến thành NaT (tránh crash code)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    
    # (Tùy chọn) Nếu bạn muốn quy đổi về giờ Việt Nam (UTC+7)
    # df['date'] = df['date'].dt.tz_convert('Asia/Bangkok')

    # Loại bỏ các dòng bị lỗi date (nếu có)
    df = df.dropna(subset=['date'])

    # --- 2. XỬ LÝ TEXT (Bỏ ký tự đặc biệt) ---
    # B1: Chuyển về chữ thường + ép kiểu string
    df['clean_text'] = df['comment'].astype(str).str.lower()
    
    # B2: Dùng Regex để loại bỏ ký tự đặc biệt
    # Pattern r'[^\w\s]': 
    #   ^ : Phủ định (Không phải là...)
    #   \w: Chữ cái (a-z, 0-9, bao gồm cả tiếng Việt có dấu như ă, â, đ...)
    #   \s: Khoảng trắng
    # => Nghĩa là: Thay thế tất cả những gì KHÔNG PHẢI chữ và khoảng trắng bằng rỗng
    df['clean_text'] = df['clean_text'].str.replace(r'[^\w\s]', '', regex=True)
    
    # B3: Xóa khoảng trắng thừa ở đầu/cuối/giữa (ví dụ: "  pin   yếu " -> "pin yếu")
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Lưu kết quả
    save_key = f"{USE_CASE_PREFIX}/processed/clean.parquet"
    df.to_parquet(f"s3://{MINIO_BUCKET}/{save_key}", index=False, storage_options=get_storage_options())
    
    return save_key

@task(dag=dag)
def analyze_trends_inference(file_key: str):
    """
    Task quan trọng: Load model từ MLflow -> Predict -> Aggregate Trend
    """
    log.info(f"Reading data from {file_key}")
    df = pd.read_parquet(f"s3://{MINIO_BUCKET}/{file_key}", storage_options=get_storage_options())
    
    # --- 1. LOAD MODEL TỪ MLFLOW ---
    log.info("Loading models from MLflow Registry...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Load Sentiment Model
        sentiment_uri = f"models:/{SENTIMENT_MODEL_NAME}@{MODEL_ALIAS}"
        sentiment_model = mlflow.pyfunc.load_model(sentiment_uri)
        log.info(f"Loaded Sentiment Model: {sentiment_uri}")
        
        # Load Topic Model
        topic_uri = f"models:/{TOPIC_MODEL_NAME}@{MODEL_ALIAS}"
        topic_model = mlflow.pyfunc.load_model(topic_uri)
        log.info(f"Loaded Topic Model: {topic_uri}")
        
    except Exception as e:
        log.error(f"Failed to load models from MLflow: {e}")
        # Fallback logic (nếu chưa deploy model thật thì dùng dummy để test pipeline)
        log.warning("USING DUMMY PREDICTION FOR TESTING PIPELINE")
        df['topic'] = df['clean_text'].apply(lambda x: 'Battery' if 'pin' in x else ('Shipping' if 'ship' in x else 'Other'))
        df['sentiment'] = df['clean_text'].apply(lambda x: 'Negative' if 'yếu' in x else 'Positive')
    else:
        # --- 2. INFERENCE (DỰ ĐOÁN) ---
        log.info("Running inference...")
        # Giả sử input cho model là list text hoặc series
        df['topic'] = topic_model.predict(df['clean_text'])
        sentiment_output = sentiment_model.predict(df['clean_text'])

        df['sentiment'] = pd.Series(sentiment_output).astype(str).str.capitalize()

    # --- 3. AGGREGATION (TÍNH TOÁN TREND) ---
    log.info("Aggregating trends...")
    
    # Group theo Ngày + Topic + Sentiment
    trend_df = df.groupby([
        pd.Grouper(key='date', freq='D'), 
        'topic', 
        'sentiment'
    ]).size().reset_index(name='volume')
    
    # Pivot bảng: Date | Topic | Negative | Positive | ...
    pivot_trend = trend_df.pivot_table(
        index=['date', 'topic'], 
        columns='sentiment', 
        values='volume', 
        fill_value=0
    ).reset_index()
    
    # Tính các chỉ số quan trọng
    for col in ['Negative', 'Positive', 'Neutral']:
        if col not in pivot_trend.columns:
            pivot_trend[col] = 0
    
    # Tổng volume
    pivot_trend['total_volume'] = pivot_trend['Negative'] + pivot_trend['Positive'] + pivot_trend['Neutral']
    
    # Tính tỷ lệ (Tránh chia cho 0)
    pivot_trend['negative_rate'] = pivot_trend.apply(
        lambda x: x['Negative'] / x['total_volume'] if x['total_volume'] > 0 else 0, axis=1
    )

    # --- 4. SAVE GOLD DATA ---
    save_key = f"{USE_CASE_PREFIX}/gold/daily_trend_analytics.csv"
    pivot_trend.to_csv(f"s3://{MINIO_BUCKET}/{save_key}", index=False, storage_options=get_storage_options())
    
    log.info(f"Trend analysis saved to {save_key}")
    return save_key

# --- FLOW ---
init = ensure_bucket_exists()
raw = extract_data()
clean = preprocess_trends(raw)
analytics = analyze_trends_inference(clean)

success = BashOperator(
    task_id='notify_success', 
    bash_command='echo "Trend Analysis Completed Successfully! Check MinIO Gold folder."', 
    dag=dag
)

init >> raw >> clean >> analytics >> success