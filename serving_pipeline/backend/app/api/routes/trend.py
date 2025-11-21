from fastapi import APIRouter, UploadFile, File
from ..controllers.trend import analyze_trend_csv

router = APIRouter()

@router.post("/predict", summary="Dự đoán  văn bản")
async def analyze_trend(file: UploadFile = File(...)):
    """
    """
    return await analyze_trend_csv(file)

