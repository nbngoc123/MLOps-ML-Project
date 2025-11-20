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
logger = logging.getLogger("recsys-controller")
logging.basicConfig(level=logging.INFO)

# 1. Config Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

# Tên Model khớp với DAG: mlflow.register_model(..., "Nexus_RecSys_SVD")
MODEL_NAME = "Nexus_RecSys_SVD"
ALIAS = "production"
ENCODER_PATH_S3 = "product_recsys/encoders" # Đường dẫn chứa user_le.pkl, item_le.pkl

# Khởi tạo kết nối S3 để tải Encoders
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': MINIO_ENDPOINT},
    key=MINIO_ACCESS_KEY,
    secret=MINIO_SECRET_KEY
)

# Global variables
model_pipeline = None
user_encoder = None
item_encoder = None
is_model_ready = False

class RecSysInput(BaseModel):
    user_id: str
    product_id: str
    description: str = "" # Text mô tả sản phẩm (quan trọng cho TF-IDF feature)

# ========================
# 1. Hàm Load Encoders (Từ MinIO)
# ========================
def load_encoders(local_dir="/tmp/encoders"):
    """Tải LabelEncoders từ MinIO về để map ID"""
    global user_encoder, item_encoder
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Tải user_le.pkl
        logger.info("📥 Downloading User Encoder...")
        fs.get(f"{MINIO_BUCKET}/{ENCODER_PATH_S3}/user_le.pkl", f"{local_dir}/user_le.pkl")
        user_encoder = joblib.load(f"{local_dir}/user_le.pkl")
        
        # Tải item_le.pkl
        logger.info("📥 Downloading Item Encoder...")
        fs.get(f"{MINIO_BUCKET}/{ENCODER_PATH_S3}/item_le.pkl", f"{local_dir}/item_le.pkl")
        item_encoder = joblib.load(f"{local_dir}/item_le.pkl")
        
        logger.info("✅ Encoders loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load encoders: {e}")
        raise e

# ========================
# 2. Hàm Load Model (Từ MLflow)
# ========================
async def load_recsys_model(retries=3, delay=2):
    global model_pipeline, is_model_ready
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    for attempt in range(1, retries+1):
        try:
            logger.info(f"🔄 [Attempt {attempt}] Finding production model for '{MODEL_NAME}'...")
            
            # 1. Load Encoders trước (Bắt buộc phải có để xử lý input)
            load_encoders()

            # 2. Lấy Run ID từ Alias 'production'
            try:
                mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            except Exception:
                logger.error(f"❌ Model '{MODEL_NAME}' with alias '{ALIAS}' not found.")
                raise

            run_id = mv.run_id
            logger.info(f"🎯 Found Production Run ID: {run_id}")

            # 3. Download Model Artifacts
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id)
            
            # 4. Tìm file model.pkl
            model_file = None
            for root, dirs, files in os.walk(local_path):
                if "model.pkl" in files:
                    model_file = os.path.join(root, "model.pkl")
                    break 

            if not model_file:
                raise FileNotFoundError("❌ Không tìm thấy model.pkl!")

            # 5. Load vào RAM
            model_pipeline = joblib.load(model_file)
            is_model_ready = True
            
            logger.info(f"✅ RecSys Model Loaded Successfully from {local_path}")
            
            # Clean up
            try: shutil.rmtree(local_path) 
            except: pass
            return

        except Exception as e:
            logger.warning(f"⚠️ Load failed ({e}). Retrying in {delay}s...")
            is_model_ready = False
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error("❌ Final failure loading RecSys model.")

# ========================
# 3. API Endpoints
# ========================

async def startup_event():
    await load_recsys_model()

async def predict_recsys_score(data: RecSysInput):
    """
    Dự đoán rating (score) cho cặp User-Product.
    """
    if not is_model_ready:
        await load_recsys_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        # 1. Transform ID -> Index
        # Cần handle trường hợp User mới hoặc Item mới (Cold Start)
        try:
            u_idx = user_encoder.transform([str(data.user_id)])[0]
        except ValueError:
            # Fallback cho user mới: Gán index đặc biệt hoặc random, hoặc dùng trung bình
            # Ở đây mình gán -1 để biểu thị (tùy logic model, cây quyết định xử lý số lạ khá tốt)
            u_idx = -1 
        
        try:
            i_idx = item_encoder.transform([str(data.product_id)])[0]
        except ValueError:
            i_idx = -1

        # 2. Tạo DataFrame input (Đúng format Pipeline yêu cầu)
        # Cột: ['user_idx', 'item_idx', 'full_description']
        input_df = pd.DataFrame({
            'user_idx': [u_idx],
            'item_idx': [i_idx],
            'full_description': [data.description.lower()]
        })

        # 3. Predict
        pred_rating = model_pipeline.predict(input_df)[0]
        
        # Clip kết quả trong khoảng hợp lý (ví dụ 1-5 sao)
        final_score = max(1.0, min(5.0, float(pred_rating)))

        return {
            "user_id": data.user_id,
            "product_id": data.product_id,
            "predicted_rating": round(final_score, 2),
            "is_cold_start": (u_idx == -1 or i_idx == -1),
            "model_name": MODEL_NAME
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

async def predict_recsys_batch(file: UploadFile = File(...)):
    """
    Upload CSV chứa danh sách user_id, product_id, description -> Trả về rating dự đoán.
    """
    if not is_model_ready:
        await load_recsys_model()
        if not is_model_ready: raise HTTPException(503, "Model not ready")

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        req_cols = ['user_id', 'product_id', 'description']
        if not all(col in df.columns for col in req_cols):
             raise HTTPException(400, f"CSV phải chứa các cột: {req_cols}")

        # 1. Bulk Transform IDs
        def safe_transform(encoder, value):
            try: return encoder.transform([str(value)])[0]
            except: return -1
            
        df['user_idx'] = df['user_id'].apply(lambda x: safe_transform(user_encoder, x))
        df['item_idx'] = df['product_id'].apply(lambda x: safe_transform(item_encoder, x))
        
        # 2. Prepare Input DF cho Pipeline
        # Pipeline yêu cầu tên cột chính xác: user_idx, item_idx, full_description
        pipeline_input = pd.DataFrame({
            'user_idx': df['user_idx'],
            'item_idx': df['item_idx'],
            'full_description': df['description'].fillna("").astype(str).str.lower()
        })

        # 3. Predict
        preds = model_pipeline.predict(pipeline_input)
        
        # 4. Format Output
        df['predicted_rating'] = [max(1.0, min(5.0, float(p))) for p in preds]
        
        result_data = df[['user_id', 'product_id', 'predicted_rating']].to_dict(orient="records")

        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": result_data,
            "model": MODEL_NAME
        }

    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")