from fastapi import APIRouter, UploadFile, File
from ..controllers.topic import TopicInput, predict_topic_single, predict_topic_batch

router = APIRouter()

@router.post("/predict", summary="Dự đoán  văn bản")
async def analyze_trend_csv(data: TopicInput):
    """
    .
    """
    return await analyze_trend_csv(data)

