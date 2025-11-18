import joblib
import s3fs
import os
import logging
from fastapi import HTTPException
from pydantic import BaseModel
import asyncio

logger = logging.getLogger("sentiment-controller")

# ========================
# MinIO Config
# ========================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio-secret-key")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "nexusml")

MODEL_PATH = f"{MINIO_BUCKET}/models/sentiment/latest/model.pkl"
VECTORIZER_PATH = f"{MINIO_BUCKET}/models/sentiment/latest/vectorizer.pkl"

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
# Load model với retry
# ========================
async def load_sentiment_model(retries=5, delay=5):
    global vectorizer, model
    for attempt in range(1, retries+1):
        try:
            logger.info(f"[Attempt {attempt}] Loading vectorizer from {VECTORIZER_PATH}")
            with fs.open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = joblib.load(f)
            logger.info("Vectorizer loaded successfully.")

            logger.info(f"[Attempt {attempt}] Loading model from {MODEL_PATH}")
            with fs.open(MODEL_PATH, 'rb') as f:
                model = joblib.load(f)
            logger.info("Model loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                raise HTTPException(status_code=500, detail=f"Cannot load model: {e}")

# ========================
# Predict
# ========================
async def predict_sentiment(data: SentimentInput):
    global vectorizer, model
    if vectorizer is None or model is None:
        await load_sentiment_model()

    try:
        vec = vectorizer.transform([data.text])
        pred = model.predict(vec)[0]
        label_map = {0: "tiêu cực", 1: "tích cực", 2: "trung tính"}
        sentiment = label_map.get(pred, "unknown")
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
