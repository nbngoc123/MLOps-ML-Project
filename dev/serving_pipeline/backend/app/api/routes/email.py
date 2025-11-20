from fastapi import APIRouter, UploadFile, File
from ..controllers.email import predict_email_single, predict_email_batch, EmailInput

router = APIRouter()

@router.post("/predict", summary="Dự đoán  văn bản")
async def predict_email_single(data: EmailInput):
    """
    .
    """
    return await predict_email_single(data)

@router.post("/predict_email_batch", summary="Dự đoán văn bản")
async def predict_email_batch(file: UploadFile = File(...)):
    """
    
    """
    return await predict_email_batch(file)
