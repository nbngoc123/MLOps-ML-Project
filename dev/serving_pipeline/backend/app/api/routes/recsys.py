from fastapi import APIRouter, UploadFile, File
from ..controllers.recsys import RecSysInput, predict_recsys_score, predict_recsys_batch

router = APIRouter()

@router.post("/predict", summary="Dự đoán  văn bản")
async def predict_recsys(data: RecSysInput):
    """
    .
    """
    return await predict_recsys_score(data)

@router.post("/predict_batch", summary="Dự đoán văn bản")
async def predict_recsys_b(file: UploadFile = File(...)):
    """
    
    """
    return await predict_recsys_batch(file)
