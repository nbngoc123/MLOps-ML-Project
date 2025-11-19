from mlflow.tracking import MlflowClient
import joblib
import s3fs
import os
import logging
from fastapi import HTTPException
from pydantic import BaseModel

# ========================
# Config
# ========================
logger = logging.getLogger("sentiment-controller")
logging.basicConfig(level=logging.INFO)

# 1. Config cho MLflow (Chỉ để lấy Run ID)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# 2. Config cho S3FS (Để tải file trực tiếp)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

MODEL_NAME = "SentimentClassifier"
ALIAS = "production"

# Khởi tạo kết nối S3
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': MINIO_ENDPOINT},
    key=MINIO_ACCESS_KEY,
    secret=MINIO_SECRET_KEY
)

vectorizer = None
model = None

class SentimentInput(BaseModel):
    text: str

# ========================
# 1. Hàm Load Model: Kết hợp MLflow Lookup + S3FS Download
# ========================
async def load_sentiment_model(retries=3, delay=2):
    global vectorizer, model
    
    # Khởi tạo MLflow Client (Chỉ dùng để hỏi thông tin, không tải file)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    for attempt in range(1, retries+1):
        try:
            # --- BƯỚC 1: Hỏi MLflow xem Run ID nào đang là Production ---
            logger.info(f"[Attempt {attempt}] Asking MLflow for alias '@{ALIAS}'...")
            try:
                mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
                run_id = mv.run_id
                logger.info(f"🎯 MLflow says: Production Run ID is {run_id}")
            except Exception as e:
                raise FileNotFoundError(f"Không tìm thấy model alias @{ALIAS} trên MLflow. Error: {e}")

            # --- BƯỚC 2: Tự xây dựng đường dẫn MinIO dựa trên Run ID ---
            # Cấu trúc chuẩn: bucket/models/<run_id>/artifacts/...
            base_path = f"{MINIO_BUCKET}/models/{run_id}/artifacts"
            
            # Đường dẫn Model (Do mlflow.sklearn.log_model tạo ra folder 'model')
            model_s3_path = f"{base_path}/model/model.pkl"
            
            # Đường dẫn Vectorizer (Do mlflow.log_artifact tạo ra folder 'artifacts')
            # Dựa trên UI bạn gửi: artifacts/vectorizer.pkl
            # => Full path: nexusml/models/.../artifacts/artifacts/vectorizer.pkl
            vec_s3_path = f"{base_path}/artifacts/vectorizer.pkl"

            # --- BƯỚC 3: Tải và Load bằng s3fs + joblib ---
            
            # A. Load Vectorizer
            logger.info(f"Loading Vectorizer from MinIO: {vec_s3_path}")
            if not fs.exists(vec_s3_path):
                raise FileNotFoundError(f"Vectorizer not found at: {vec_s3_path}")
                
            with fs.open(vec_s3_path, 'rb') as f:
                vectorizer = joblib.load(f)

            # B. Load Model
            logger.info(f"Loading Model from MinIO: {model_s3_path}")
            if not fs.exists(model_s3_path):
                raise FileNotFoundError(f"Model not found at: {model_s3_path}")

            with fs.open(model_s3_path, 'rb') as f:
                model = joblib.load(f)
            
            logger.info("✅ Successfully loaded Model & Vectorizer (Hybrid Method)!")
            return

        except Exception as e:
            import asyncio
            logger.warning(f"Load failed ({e}). Retrying in {delay}s...")
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error("Final failure loading model.")
                raise HTTPException(status_code=500, detail=f"Cannot load model: {e}")

# ========================
# 2. API Endpoint
# ========================
async def predict_sentiment(data: SentimentInput):
    global vectorizer, model
    
    if vectorizer is None or model is None:
        await load_sentiment_model()

    try:
        # Transform
        vec = vectorizer.transform([data.text])
        
        # Predict
        pred = model.predict(vec)[0]
        
        # Map label
        label_map = {0: "tiêu cực", 1: "tích cực", 2: "trung tính"}
        sentiment = label_map.get(int(pred), "unknown")
        
        return {
            "text": data.text,
            "sentiment": sentiment,
            "run_source": "mlflow_lookup_s3_load"
        }
    except Exception as e:
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")