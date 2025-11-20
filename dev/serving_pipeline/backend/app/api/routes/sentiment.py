from fastapi import APIRouter, UploadFile, File
from ..controllers.sentiment import predict_single, predict_batch, SentimentInput

router = APIRouter()

@router.post("/predict", summary="Dự đoán cảm xúc văn bản")
async def predict_sentiment(data: SentimentInput):
    """
    Dự đoán cảm xúc (tích cực, tiêu cực, trung tính) của một đoạn văn bản.
    """
    return await predict_single(data)

@router.post("/predict_batch", summary="Dự đoán cảm xúc văn bản")
async def predict_sentiment_b(file: UploadFile = File(...)):
    """
    Dự đoán cảm xúc (tích cực, tiêu cực, trung tính) của một đoạn văn bản.
    """
    return await predict_batch(file)
