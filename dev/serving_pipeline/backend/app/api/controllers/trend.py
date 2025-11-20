import mlflow
import os
import logging
import joblib
import s3fs
import asyncio
import pandas as pd
import shutil
import numpy as np
from io import BytesIO
from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

# ========================
# Config
# ========================
logger = logging.getLogger("trend-controller")
logging.basicConfig(level=logging.INFO)

# 1. Config Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

# Tên Model khớp với Config trong DAG
SENTIMENT_MODEL_NAME = "sentiment_classifier_Model"
TOPIC_MODEL_NAME = "Nexus_Topic_Classifier"
ALIAS = "production"

# Global variables
sentiment_model = None
topic_model = None
are_models_ready = False

# ========================
# 1. Hàm Load Multiple Models
# ========================
async def load_trend_models(retries=3, delay=2):
    """
    Load đồng thời cả Sentiment Model và Topic Model
    """
    global sentiment_model, topic_model, are_models_ready
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    for attempt in range(1, retries+1):
        try:
            logger.info(f"🔄 [Attempt {attempt}] Loading Trend Analysis Models...")

            # --- A. Load Sentiment Model ---
            mv_sent = client.get_model_version_by_alias(SENTIMENT_MODEL_NAME, ALIAS)
            path_sent = mlflow.artifacts.download_artifacts(run_id=mv_sent.run_id)
            
            # Tìm file model.pkl cho Sentiment
            sent_file = None
            for root, _, files in os.walk(path_sent):
                if "model.pkl" in files:
                    sent_file = os.path.join(root, "model.pkl")
                    break
            if not sent_file: raise FileNotFoundError("Sentiment model.pkl not found")
            sentiment_model = joblib.load(sent_file)
            logger.info("✅ Loaded Sentiment Model")

            # --- B. Load Topic Model ---
            mv_topic = client.get_model_version_by_alias(TOPIC_MODEL_NAME, ALIAS)
            path_topic = mlflow.artifacts.download_artifacts(run_id=mv_topic.run_id)
            
            # Tìm file model.pkl cho Topic
            topic_file = None
            for root, _, files in os.walk(path_topic):
                if "model.pkl" in files:
                    topic_file = os.path.join(root, "model.pkl")
                    break
            if not topic_file: raise FileNotFoundError("Topic model.pkl not found")
            topic_model = joblib.load(topic_file)
            logger.info("✅ Loaded Topic Model")

            # --- Cleanup & Finish ---
            are_models_ready = True
            try:
                shutil.rmtree(path_sent)
                shutil.rmtree(path_topic)
            except: pass
            
            return

        except Exception as e:
            logger.warning(f"⚠️ Load failed ({e}). Retrying in {delay}s...")
            are_models_ready = False
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error("❌ Final failure loading Trend models.")

# ========================
# 2. API Endpoints
# ========================

async def startup_event():
    await load_trend_models()

async def analyze_trend_csv(file: UploadFile = File(...)):
    """
    Input: File CSV chứa cột 'date' và 'comment' (hoặc 'text').
    Output: JSON (dạng bảng CSV) chứa thống kê xu hướng theo ngày.
    """
    if not are_models_ready:
        await load_trend_models()
        if not are_models_ready: raise HTTPException(503, "Models not ready")

    try:
        # 1. Đọc File
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Validate Columns
        text_col = next((col for col in ['comment', 'text', 'content', 'feedback'] if col in df.columns), None)
        date_col = next((col for col in ['date', 'time', 'created_at'] if col in df.columns), None)
        
        if not text_col or not date_col:
            raise HTTPException(400, "CSV phải chứa cột thời gian ('date') và nội dung ('comment'/'text')")

        logger.info(f"Analyzing trends for {len(df)} records...")

        # 2. Preprocessing (Giống logic trong DAG)
        # Xử lý Date
        df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Xử lý Text
        clean_text = df[text_col].astype(str).str.lower().fillna("")
        clean_text = clean_text.str.replace(r'[^\w\s]', '', regex=True) # Bỏ ký tự đặc biệt
        
        # 3. Inference (Chạy 2 model)
        # Predict Topic
        df['topic'] = topic_model.predict(clean_text)
        
        # Predict Sentiment & Map Labels
        sent_preds = sentiment_model.predict(clean_text)
        
        # Map kết quả Sentiment (Giả sử model trả về 0,1,2 hoặc chuỗi)
        # Logic mapping cho nhất quán với báo cáo
        def map_sentiment(val):
            s_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
            if isinstance(val, (int, np.integer)):
                return s_map.get(val, 'Neutral')
            return str(val).capitalize() # Nếu model đã trả về chuỗi 'spam'/'ham' etc.

        df['sentiment'] = [map_sentiment(x) for x in sent_preds]

        # 4. Aggregation (Tạo bảng báo cáo)
        # Group theo Ngày (Daily) + Topic + Sentiment
        trend_df = df.groupby([
            pd.Grouper(key='date', freq='D'), 
            'topic', 
            'sentiment'
        ]).size().reset_index(name='volume')

        # Pivot Table: Biến Sentiment thành các cột (Negative, Positive, Neutral)
        pivot_trend = trend_df.pivot_table(
            index=['date', 'topic'], 
            columns='sentiment', 
            values='volume', 
            fill_value=0
        ).reset_index()

        # Đảm bảo có đủ cột để không lỗi frontend
        for col in ['Negative', 'Positive', 'Neutral']:
            if col not in pivot_trend.columns:
                pivot_trend[col] = 0

        # Tính chỉ số phụ
        pivot_trend['total_volume'] = pivot_trend['Negative'] + pivot_trend['Positive'] + pivot_trend['Neutral']
        pivot_trend['negative_rate'] = pivot_trend.apply(
            lambda x: round(x['Negative'] / x['total_volume'], 2) if x['total_volume'] > 0 else 0.0, 
            axis=1
        )

        # Format Date thành string cho JSON
        pivot_trend['date'] = pivot_trend['date'].dt.strftime('%Y-%m-%d')

        # 5. Return Result
        # Trả về dạng list of dicts, Frontend có thể hiển thị table hoặc cho user tải về CSV
        result_data = pivot_trend.to_dict(orient="records")

        return {
            "filename": file.filename,
            "total_days": len(pivot_trend),
            "data": result_data,
            "columns": list(pivot_trend.columns) # Gửi kèm tên cột để frontend dễ render header
        }

    except Exception as e:
        logger.error(f"Trend Analysis Error: {e}")
        raise HTTPException(500, f"Lỗi phân tích xu hướng: {str(e)}")