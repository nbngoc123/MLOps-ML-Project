from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from app.services.model_manager import model_manager
from app.models.inference import EmailModel

class EmailInput(PydanticBaseModel):
    text: str

# ========================
# Hàm Load Model đã được thay thế bằng ModelManager
# ========================

# ========================
# API Endpoints
# ========================

async def startup_event():
    """Gọi hàm này khi khởi động FastAPI app"""
    # ModelManager sẽ tự động đăng ký model classes khi ứng dụng khởi động
    pass

async def predict_email_single(data: EmailInput):
    """Dự đoán Spam/Ham cho 1 email"""
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="email",
            model_name="Nexus_Email_Classifier",
            model_version="production"
        )
        
        # Gọi phương thức predict của model
        result = await model_instance.predict({"text": data.text})
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))

async def predict_email_batch(file: UploadFile = File(...)):
    """
    Upload CSV -> Trả về JSON phân loại Email.
    """
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="email",
            model_name="Nexus_Email_Classifier",
            model_version="production"
        )
        
        # Gọi phương thức predict_batch của model
        result = await model_instance.predict_batch(file)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")
