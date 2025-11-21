from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from app.services.model_manager import model_manager
from app.models.inference import SentimentModel

class SentimentInput(PydanticBaseModel):
    text: str

# ========================
# Hàm Load Model đã được thay thế bằng ModelManager
# ========================

# ========================
# API Endpoints
# ========================

async def startup_event():
    # ModelManager sẽ tự động đăng ký model classes khi ứng dụng khởi động
    pass

async def predict_single(data: SentimentInput):
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="sentiment",
            model_name="sentiment_classifier_Model",
            model_version="production"
        )
        
        # Gọi phương thức predict của model
        result = await model_instance.predict({"text": data.text})
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))



async def predict_batch(file: UploadFile = File(...)):
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="sentiment",
            model_name="sentiment_classifier_Model",
            model_version="production"
        )
        
        # Gọi phương thức predict_batch của model
        result = await model_instance.predict_batch(file)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")
