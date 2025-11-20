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
logger = logging.getLogger("email-controller")
logging.basicConfig(level=logging.INFO)

# 1. Config Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

# Tên Model khớp với DAG: mlflow.register_model(..., "Nexus_Email_Classifier")
MODEL_NAME = "Nexus_Email_Classifier"
ALIAS = "production"

# Global model variable (Pipeline chứa cả Vectorizer + Classifier)
model_pipeline = None
is_model_ready = False

class EmailInput(BaseModel):
    text: str

# ========================
# 1. Hàm Load Model (Email Version)
# ========================
async def load_email_model(retries=3, delay=2):
    global model_pipeline, is_model_ready
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    for attempt in range(1, retries+1):
        try:
            logger.info(f"🔄 [Attempt {attempt}] Finding production model for '{MODEL_NAME}'...")
            
            # 1. Lấy Run ID từ Alias 'production'
            try:
                mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            except Exception:
                logger.error(f"❌ Model '{MODEL_NAME}' with alias '{ALIAS}' not found in MLflow.")
                raise

            run_id = mv.run_id
            logger.info(f"🎯 Found Production Run ID: {run_id}")

            # 2. Download Artifacts
            # Trong DAG dùng make_pipeline -> mlflow.sklearn.log_model 
            # Pipeline (Vectorizer + Model) sẽ được pickle vào model.pkl
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id)
            
            # 3. Tìm file model.pkl
            model_file = None
            for root, dirs, files in os.walk(local_path):
                if "model.pkl" in files:
                    model_file = os.path.join(root, "model.pkl")
                    break 

            if not model_file:
                raise FileNotFoundError("❌ Không tìm thấy model.pkl trong artifacts tải về!")

            # 4. Load vào RAM
            model_pipeline = joblib.load(model_file)
            is_model_ready = True
            
            logger.info(f"✅ Email Model Loaded Successfully from {local_path}")
            
            # Dọn dẹp
            try: shutil.rmtree(local_path) 
            except: pass
            return

        except Exception as e:
            logger.warning(f"⚠️ Load failed ({e}). Retrying in {delay}s...")
            is_model_ready = False
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error("❌ Final failure loading email model.")

# ========================
# 2. API Endpoints
# ========================

async def startup_event():
    """Gọi hàm này khi khởi động FastAPI app"""
    await load_email_model()

async def predict_email_single(data: EmailInput):
    """Dự đoán Spam/Ham cho 1 email"""
    if not is_model_ready:
        await load_email_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        # Predict (Pipeline tự handle vectorization)
        # Model trả về 0 hoặc 1 (do DAG map: ham=0, spam=1)
        pred_raw = model_pipeline.predict([data.text])[0]
        
        # Map label
        label = "spam" if pred_raw == 1 else "ham"
        
        # Tính confidence
        confidence = 0.0
        if hasattr(model_pipeline, "predict_proba"):
            try:
                # predict_proba trả về [[prob_0, prob_1]]
                confidence = float(model_pipeline.predict_proba([data.text]).max())
            except:
                confidence = 1.0

        return {
            "text": data.text,
            "is_spam": bool(pred_raw == 1),
            "label": label,
            "confidence": round(confidence, 4),
            "model_name": MODEL_NAME,
            "run_source": "mlflow_production"
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

async def predict_email_batch(file: UploadFile = File(...)):
    """
    Upload CSV -> Trả về JSON phân loại Email.
    """
    if not is_model_ready:
        await load_email_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        # 1. Đọc file
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Kiểm tra cột text
        text_col = next((col for col in ['text', 'content', 'body', 'email'] if col in df.columns), None)
        if not text_col:
            raise HTTPException(400, "CSV phải chứa cột 'text', 'content', 'body' hoặc 'email'")

        logger.info(f"Processing batch of {len(df)} emails...")

        # 2. Batch Processing
        raw_text = df[text_col].astype(str).str.lower().fillna("")
        
        # Predict
        predictions_raw = model_pipeline.predict(raw_text)
        
        # 3. Gán kết quả
        # Map 1 -> spam, 0 -> ham
        df['label_prediction'] = ["spam" if x == 1 else "ham" for x in predictions_raw]
        df['is_spam'] = [bool(x == 1) for x in predictions_raw]
        
        # Tính confidence
        if hasattr(model_pipeline, "predict_proba"):
            try:
                probs = model_pipeline.predict_proba(raw_text).max(axis=1)
                df['confidence'] = probs.round(4)
            except:
                df['confidence'] = 0.0

        # 4. Format JSON
        result_data = df.to_dict(orient="records")

        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": result_data,
            "model": MODEL_NAME
        }

    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")