import logging
import os
import joblib
import asyncio
import pandas as pd
import shutil
from io import BytesIO
from typing import Any, Dict, List, Optional, Type
from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from mlflow.tracking import MlflowClient
from app.models._base import BaseModel
from app.core.config import settings

# Logging setup
logger = logging.getLogger("inference")

# Lớp SentimentModel kế thừa từ BaseModel
class SentimentModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.vectorizer = None
        self.label_map = {
            'negative': 'tiêu cực',
            'positive': 'tích cực',
            'neutral': 'trung tính'
        }

    def load(self, model_path: str):
        # Logic tải model sẽ được ModelManager xử lý. Ở đây không cần implement.
        pass

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Sentiment model not loaded")
        
        try:
            text_clean = str(data['text']).strip()
            prediction = self.model.predict([text_clean])[0]
            sentiment = self.label_map.get(prediction, str(prediction))

            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                confidence = float(self.model.predict_proba([text_clean]).max())

            return {
                "text": data['text'],
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "run_source": "mlflow_production"
            }
        except Exception as e:
            logger.error(f"Error in SentimentModel.predict: {e}")
            raise HTTPException(500, str(e))
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Sentiment model not loaded")
        
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))

            text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
            if not text_col:
                raise HTTPException(400, "CSV must contain 'text', 'comment' or 'content' column")
            
            raw_text = df[text_col].astype(str).fillna("")
            predictions = self.model.predict(raw_text)
            df['sentiment'] = [self.label_map.get(p, str(p)) for p in predictions]

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(raw_text).max(axis=1)
                df['confidence'] = probs.round(4)

            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df.to_dict(orient="records"),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            raise HTTPException(500, f"File processing error: {str(e)}")

# Tương tự, tạo các lớp cho Topic, Email, RecSys, Trend. Mỗi lớp sẽ có logic predict riêng.
# Ví dụ cho TopicModel (logic đơn giản hóa)
class TopicModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    def load(self, model_path: str):
        pass

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Topic model not loaded")
        # Implement logic predict topic here
        # Placeholder logic
        return {
            "text": data['text'],
            "topics": ["sample_topic_1", "sample_topic_2"],
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Topic model not loaded")
        # Implement logic predict topic batch here
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), "text")
        # Placeholder logic
        df['topics'] = ["sample_topic_1", "sample_topic_2"]
        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": df.to_dict(orient="records"),
            "run_source": "mlflow_production_batch"
        }

class EmailModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    def load(self, model_path: str):
        pass

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Email model not loaded")
        # Placeholder logic
        return {
            "text": data['text'],
            "category": "sample_category",
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Email model not loaded")
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), "text")
        df['category'] = "sample_category"
        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": df.to_dict(orient="records"),
            "run_source": "mlflow_production_batch"
        }

class RecSysModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    def load(self, model_path: str):
        pass

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("RecSys model not loaded")
        # Placeholder logic
        return {
            "user_id": data.get('user_id'),
            "recommended_items": ["item_1", "item_2"],
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("RecSys model not loaded")
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        df['recommended_items'] = "item_1, item_2"
        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": df.to_dict(orient="records"),
            "run_source": "mlflow_production_batch"
        }

class TrendModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    def load(self, model_path: str):
        pass

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Trend model not loaded")
        # Placeholder logic
        return {
            "trend_analysis": "sample_trend",
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Trend model not loaded")
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        df['trend_analysis'] = "sample_trend"
        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": df.to_dict(orient="records"),
            "run_source": "mlflow_production_batch"
        }
