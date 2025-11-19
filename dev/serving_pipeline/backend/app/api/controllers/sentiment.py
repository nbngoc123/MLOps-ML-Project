import mlflow
import os
import logging
import joblib
import s3fs
import asyncio
import pandas as pd
import shutil
from io import BytesIO
from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
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
is_model_ready = False

class SentimentInput(BaseModel):
    text: str

# ========================
# 1. Hàm Load Model (Robust Version)
# ========================
async def load_sentiment_model(retries=3, delay=2):
    global vectorizer, model, is_model_ready
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    for attempt in range(1, retries+1):
        try:
            logger.info(f"🔄 [Attempt {attempt}] Finding production model for '{MODEL_NAME}'...")
            
            # 1. Lấy Run ID từ Alias
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            run_id = mv.run_id
            model_source = mv.source # e.g., s3://nexusml/models/.../artifacts/model
            logger.info(f"🎯 Found Production Run ID: {run_id}")

            # 2. Download Artifacts về thư mục tạm
            # Lý do: Dùng mlflow download an toàn hơn tự mò đường dẫn S3 khi cấu trúc folder thay đổi
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id)
            
            # 3. Tìm file model.pkl và vectorizer.pkl (Đệ quy)
            model_file = None
            vec_file = None
            
            for root, dirs, files in os.walk(local_path):
                if "model.pkl" in files:
                    model_file = os.path.join(root, "model.pkl")
                if "vectorizer.pkl" in files:
                    vec_file = os.path.join(root, "vectorizer.pkl")

            if not model_file or not vec_file:
                raise FileNotFoundError("❌ Không tìm thấy model.pkl hoặc vectorizer.pkl trong artifacts tải về!")

            # 4. Load vào RAM
            model = joblib.load(model_file)
            vectorizer = joblib.load(vec_file)
            is_model_ready = True
            
            logger.info(f"✅ Model Loaded Successfully from {local_path}")
            
            # Dọn dẹp folder tạm để tiết kiệm dung lượng
            try: shutil.rmtree(local_path) 
            except: pass
            return

        except Exception as e:
            logger.warning(f"⚠️ Load failed ({e}). Retrying in {delay}s...")
            is_model_ready = False
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error("❌ Final failure loading model.")

# ========================
# 2. API Endpoints
# ========================

async def startup_event():
    await load_sentiment_model()

async def predict_single(data: SentimentInput):
    """Dự đoán 1 câu text duy nhất"""
    if not is_model_ready:
        await load_sentiment_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        # Transform & Predict
        vec = vectorizer.transform([data.text])
        pred = model.predict(vec)[0]
        
        # Map label
        label_map = {0: "tiêu cực", 1: "tích cực", 2: "trung tính"}
        sentiment = label_map.get(int(pred), "unknown")
        
        # Lấy confidence score nếu có
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(vec).max())

        return {
            "text": data.text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "run_source": "mlflow_production"
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

async def predict_batch(file: UploadFile = File(...)):
    """
    Upload file CSV -> Trả về JSON chứa dữ liệu đã phân loại.
    Dành cho Frontend hiển thị bảng kết quả ngay lập tức.
    """
    if not is_model_ready:
        await load_sentiment_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        # 1. Đọc file CSV từ Upload
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Kiểm tra cột
        text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
        if not text_col:
            raise HTTPException(400, "CSV phải chứa cột 'text', 'comment' hoặc 'content'")

        logger.info(f"Processing batch of {len(df)} records...")

        # 2. Batch Processing (Vectorization)
        # Xử lý hàng loạt cực nhanh, không dùng loop
        raw_text = df[text_col].astype(str).str.lower().fillna("")
        X_vec = vectorizer.transform(raw_text)

        # 3. Predict
        predictions = model.predict(X_vec)
        
        # 4. Map Label & Confidence
        label_map = {0: "tiêu cực", 1: "tích cực", 2: "trung tính"}
        
        # Gán kết quả vào DataFrame
        df['prediction_code'] = predictions
        if pd.api.types.is_numeric_dtype(df['prediction_code']):
             df['sentiment'] = df['prediction_code'].map(label_map)
        else:
             df['sentiment'] = df['prediction_code'] # Trường hợp model trả về string sẵn

        # Tính confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_vec).max(axis=1)
            df['confidence'] = probs.round(4)

        # 5. Chuyển đổi DF thành List of Dicts để trả về JSON
        # orient='records' tạo ra cấu trúc: [{"text": "...", "sentiment": "..."}, ...]
        result_data = df.to_dict(orient="records")

        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": result_data,  # <--- TRẢ VỀ DATA ĐÃ GHÉP BẢNG
            "run_source": "mlflow_production_batch"
        }

    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")