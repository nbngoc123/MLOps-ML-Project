from fastapi import APIRouter
from ..controllers.sentiment import predict_sentiment, SentimentInput

router = APIRouter()

@router.post("/predict", summary="Dự đoán cảm xúc văn bản")
async def sentiment_prediction(data: SentimentInput):
    """
    Dự đoán cảm xúc (tích cực, tiêu cực, trung tính) của một đoạn văn bản.
    """
    return await predict_sentiment(data)
