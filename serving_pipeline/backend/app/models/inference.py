import logging
import os
import joblib
import asyncio
import pandas as pd
import shutil
import mlflow
import s3fs
import math
import numpy as np
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Type
from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from mlflow.tracking import MlflowClient
from app.models._base import BaseModel
from app.core.config import settings
from app.core.metrics import (
    record_sentiment_metrics,
    record_email_metrics,
    record_recsys_metrics,
    record_topic_metrics,
    record_trend_metrics,
    record_batch_processing_metrics,
    record_model_load_metrics
)

# Logging setup
logger = logging.getLogger("inference")

# Cấu hình môi trường cho MLflow
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
MINIO_ENDPOINT = settings.MINIO_ENDPOINT
MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
MINIO_BUCKET = "nexusml"
# Thiết lập biến môi trường cho MLflow/Boto3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# --- HELPER FUNCTION ĐỂ XỬ LÝ JSON SAFE ---
def safe_float(value):
    """Chuyển đổi numpy float/NaN thành python float hoặc None"""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None

def df_to_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Chuyển DataFrame sang list dict, thay thế NaN/Inf bằng None"""
    df_clean = df.where(pd.notnull(df), None)
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    return df_clean.to_dict(orient="records")
# ------------------------------------------

class TopicModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        """Load topic model từ MLflow"""
        start_time = time.time()  # Bắt đầu đo thời gian
        try:
            logger.info(f"Loading topic model: {self.model_name} v{self.model_version}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{self.model_name}@production"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Tính thời gian thực
            load_duration = time.time() - start_time
            record_model_load_metrics("topic", load_duration, success=True)
            
            logger.info(f"Topic model loaded successfully in {load_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading topic model: {e}")
            record_model_load_metrics("topic", time.time() - start_time, success=False)
            raise RuntimeError(f"Failed to load topic model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            text_clean = str(data['text']).strip().lower()
            prediction = self.model.predict([text_clean])[0]
            record_topic_metrics(str(prediction))
            
            return {
                "text": data['text'],
                "topic": prediction,
                "run_source": "mlflow_production"
            }
        except Exception as e:
            logger.error(f"Error in TopicModel.predict: {e}")
            raise HTTPException(500, str(e))
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        start_time = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
            
            if not text_col:
                raise HTTPException(400, "CSV must contain 'text', 'comment' or 'content' column")
            
            raw_text = df[text_col].astype(str).str.lower().fillna("")
            predictions = self.model.predict(raw_text)
            df['topic'] = predictions
            
            processing_time = time.time() - start_time
            record_batch_processing_metrics("topic", len(df), processing_time, True)
            
            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df.to_dict(orient="records"),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            processing_time = time.time() - start_time
            record_batch_processing_metrics("topic", 0, processing_time, False)
            raise HTTPException(500, f"File processing error: {str(e)}")


class SentimentModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.label_map = {
            'negative': 'tiêu cực',
            'positive': 'tích cực',
            'neutral': 'trung tính'
        }

    async def load(self):
        start_time = time.time() # Bắt đầu đo thời gian
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Tính thời gian thực
            load_duration = time.time() - start_time
            record_model_load_metrics("sentiment", load_duration, success=True)
            
            logger.info(f"Sentiment model loaded in {load_duration:.2f}s")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            record_model_load_metrics("sentiment", time.time() - start_time, success=False)
            raise RuntimeError(f"Failed to load model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            text_clean = str(data['text']).strip()
            prediction = self.model.predict([text_clean])[0]
            sentiment = self.label_map.get(prediction, str(prediction))

            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                try:
                    proba_arr = self.model.predict_proba([text_clean])
                    confidence = float(np.max(proba_arr[0]))
                except Exception as e:
                    logger.warning(f"Confidence calc error: {e}")
            
            record_sentiment_metrics(sentiment, confidence)
            
            return {
                "text": data['text'],
                "sentiment": sentiment,
                "confidence": safe_float(confidence),
                "run_source": "mlflow_production"
            }
        except Exception as e:
            logger.error(f"Predict error: {e}")
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        start_time = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((c for c in ['text', 'comment', 'content'] if c in df.columns), None)
            if not text_col: raise HTTPException(400, "Missing text column")
            
            raw_text = df[text_col].astype(str).fillna("")
            predictions = self.model.predict(raw_text)
            df['sentiment'] = [self.label_map.get(p, str(p)) for p in predictions]

            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(raw_text)
                    df['confidence'] = np.max(probs, axis=1).round(4)
                except:
                    df['confidence'] = 0.5
            else:
                df['confidence'] = 0.5

            processing_time = time.time() - start_time
            record_batch_processing_metrics("sentiment", len(df), processing_time, True)

            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(df),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            processing_time = time.time() - start_time
            record_batch_processing_metrics("sentiment", 0, processing_time, False)
            raise HTTPException(500, f"Batch error: {e}")


class EmailModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        start_time = time.time() # Bắt đầu đo
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Tính thời gian thực
            load_duration = time.time() - start_time
            record_model_load_metrics("email", load_duration, success=True)
            
            logger.info(f"Email model loaded in {load_duration:.2f}s")
        except Exception as e:
            logger.error(f"Error loading email model: {e}")
            record_model_load_metrics("email", time.time() - start_time, success=False)
            raise RuntimeError(f"Failed to load email model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            text_clean = str(data['text']).strip().lower()
            prediction = self.model.predict([text_clean])[0]
            is_spam = bool(prediction == 1)
            label = "spam" if is_spam else "ham"

            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                try:
                    proba_arr = self.model.predict_proba([text_clean])
                    confidence = float(np.max(proba_arr[0]))
                except:
                    pass

            record_email_metrics(is_spam, confidence)
            
            return {
                "text": data['text'],
                "is_spam": is_spam,
                "label": label,
                "confidence": safe_float(confidence),
                "run_source": "mlflow_production"
            }
        except Exception as e:
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        start_time = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((c for c in ['text', 'content', 'body', 'email'] if c in df.columns), None)
            if not text_col: raise HTTPException(400, "Missing text column")

            raw_text = df[text_col].astype(str).str.lower().fillna("")
            predictions = self.model.predict(raw_text)
            
            df['label'] = ["spam" if p == 1 else "ham" for p in predictions]
            df['is_spam'] = [bool(p == 1) for p in predictions]

            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(raw_text)
                    df['confidence'] = np.max(probs, axis=1).round(4)
                except:
                    df['confidence'] = 0.5
            else:
                df['confidence'] = 0.5

            processing_time = time.time() - start_time
            record_batch_processing_metrics("email", len(df), processing_time, True)

            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(df),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            processing_time = time.time() - start_time
            record_batch_processing_metrics("email", 0, processing_time, False)
            raise HTTPException(500, f"Batch error: {e}")


class RecSysModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.user_encoder = None
        self.item_encoder = None

    async def load(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        
        start_time = time.time() # Bắt đầu đo
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Fake load encoders (giữ nguyên logic của bạn)
            local_dir = "/tmp/encoders"
            os.makedirs(local_dir, exist_ok=True)
            try:
                # Giả lập load OK để không lỗi
                from sklearn.preprocessing import LabelEncoder
                self.user_encoder = LabelEncoder()
                self.item_encoder = LabelEncoder()
            except Exception:
                pass
            
            # Tính thời gian thực
            load_duration = time.time() - start_time
            record_model_load_metrics("recsys", load_duration, success=True)
            
        except Exception as e:
            logger.error(f"Error loading recsys model: {e}")
            record_model_load_metrics("recsys", time.time() - start_time, success=False)
            raise RuntimeError(f"Failed to load recsys model: {e}")

    def _get_encoded_idx(self, encoder, raw_id):
        try:
            return encoder.transform([str(raw_id)])[0]
        except:
            return 0

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.user_encoder is None: await self.load()
        
        try:
            # Feature engineering
            full_desc = f"{data.get('title','')} {data.get('text','')} {data.get('categories','')}"
            user_idx = self._get_encoded_idx(self.user_encoder, data.get('user_id'))
            item_idx = self._get_encoded_idx(self.item_encoder, data.get('product_id'))
            
            input_df = pd.DataFrame({
                'user_idx': [user_idx],
                'item_idx': [item_idx],
                'full_description': [full_desc.lower()]
            })
            
            pred = self.model.predict(input_df)[0]
            
            pred_val = safe_float(pred)
            final_score = 3.0 if pred_val is None else max(1.0, min(5.0, pred_val))

            is_cold_start = False 
            record_recsys_metrics(final_score, is_cold_start)
            
            return {
                "user_id": data.get('user_id'),
                "product_id": data.get('product_id'),
                "predicted_rating": round(final_score, 2),
                "is_cold_start": is_cold_start,
                "run_source": "mlflow_production"
            }
        except Exception as e:
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None or self.user_encoder is None: await self.load()
        
        start_time = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            
            col_mapping = {
                'reviews.username': 'user_id', 'username': 'user_id',
                'asins': 'product_id', 'product_id': 'product_id', 'item_id': 'product_id',
                'reviews.title': 'title', 'title': 'title',
                'reviews.text': 'text', 'text': 'text'
            }
            df = df.rename(columns=col_mapping)
            
            required = ['user_id', 'product_id']
            missing = [c for c in required if c not in df.columns]
            if missing: raise HTTPException(400, f"Missing columns: {missing}")

            for col in ['title', 'text', 'categories']:
                if col not in df.columns: df[col] = ""

            df['full_description'] = df.fillna("").apply(
                lambda r: f"{r.get('title','')} {r.get('text','')} {r.get('categories','')}", axis=1
            ).str.lower()

            df['user_idx'] = df['user_id'].apply(lambda x: self._get_encoded_idx(self.user_encoder, x))
            df['item_idx'] = df['product_id'].apply(lambda x: self._get_encoded_idx(self.item_encoder, x))

            pipeline_input = df[['user_idx', 'item_idx', 'full_description']]
            preds = self.model.predict(pipeline_input)

            clean_preds = []
            for p in preds:
                val = safe_float(p)
                clean_preds.append(3.0 if val is None else max(1.0, min(5.0, val)))
            
            df['predicted_rating'] = clean_preds
            result_df = df[['user_id', 'product_id', 'predicted_rating']]
            
            processing_time = time.time() - start_time
            record_batch_processing_metrics("recsys", len(df), processing_time, True)
            
            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(result_df),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            processing_time = time.time() - start_time
            record_batch_processing_metrics("recsys", 0, processing_time, False)
            raise HTTPException(500, f"Batch recsys error: {str(e)}")


class TrendModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        start_time = time.time() # Bắt đầu đo
        try:
            logger.info(f"Loading trend model: {self.model_name} v{self.model_version}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{self.model_name}@production"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Tính thời gian thực
            load_duration = time.time() - start_time
            record_model_load_metrics("trend", load_duration, success=True)
            
            logger.info(f"Trend model loaded in {load_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading trend model: {e}")
            record_model_load_metrics("trend", time.time() - start_time, success=False)
            raise RuntimeError(f"Failed to load trend model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        return {
            "trend_analysis": "Time series analysis required",
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        start_time = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            
            record_trend_metrics(time.time() - start_time)
            df['trend_analysis'] = "Time series analysis required"
            
            processing_time = time.time() - start_time
            record_batch_processing_metrics("trend", len(df), processing_time, True)
            
            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df.to_dict(orient="records"),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            processing_time = time.time() - start_time
            record_batch_processing_metrics("trend", 0, processing_time, False)
            raise HTTPException(500, f"Trend batch error: {str(e)}")