from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from app.services.model_manager import model_manager
from app.models.inference import RecSysModel

class RecSysInput(PydanticBaseModel):
    user_id: str
    product_id: str
    title: str = ""
    text: str = ""
    name: str = ""
    manufacturer: str = ""
    categories: str = ""

# ========================
# Hàm Load Model đã được thay thế bằng ModelManager
# ========================

# ========================
# API Endpoints
# ========================

async def startup_event():
    # ModelManager sẽ tự động đăng ký model classes khi ứng dụng khởi động
    pass

async def predict_recsys_score(data: RecSysInput):
    """
    Dự đoán rating (score) cho cặp User-Product (Single).
    """
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="recsys",
            model_name="Nexus_RecSys_SVD",
            model_version="production"
        )
        
        # Gọi phương thức predict của model
        result = await model_instance.predict({
            "user_id": data.user_id,
            "product_id": data.product_id,
            "title": data.title,
            "text": data.text,
            "name": data.name,
            "manufacturer": data.manufacturer,
            "categories": data.categories
        })
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))

async def predict_recsys_batch(file: UploadFile = File(...)):
    """
    Dự đoán rating (Batch CSV).
    """
    try:
        # Sử dụng ModelManager để lấy model
        model_instance = await model_manager.get_model(
            model_type="recsys",
            model_name="Nexus_RecSys_SVD",
            model_version="production"
        )
        
        # Gọi phương thức predict_batch của model
        result = await model_instance.predict_batch(file)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý file: {str(e)}")
