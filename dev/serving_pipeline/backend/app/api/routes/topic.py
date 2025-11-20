from fastapi import APIRouter, UploadFile, File
from ..controllers.topic import TopicInput, predict_topic_single, predict_topic_batch

router = APIRouter()

@router.post("/predict", summary="Dự đoán  văn bản")
async def predict_topic_single(data: TopicInput):
    """
    .
    """
    return await predict_topic_single(data)

@router.post("/predict_topic_batch", summary="Dự đoán văn bản")
async def predict_topic_batch(file: UploadFile = File(...)):
    """
    
    """
    return await predict_topic_batch(file)
