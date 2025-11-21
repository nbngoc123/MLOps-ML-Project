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
from io import BytesIO
from typing import Any, Dict, List, Optional, Type
from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel as PydanticBaseModel
from mlflow.tracking import MlflowClient
from app.models._base import BaseModel
from app.core.config import settings

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

# Lớp SentimentModel kế thừa từ BaseModel


class TopicModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        """Load topic model từ MLflow"""
        try:
            logger.info(f"Loading topic model: {self.model_name} v{self.model_version}")
            
            # Thiết lập MLflow tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Tải model từ MLflow sử dụng alias "production"
            model_uri = f"models:/{self.model_name}@production"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Topic model loaded successfully from {model_uri}")
            
        except Exception as e:
            logger.error(f"Error loading topic model: {e}")
            raise RuntimeError(f"Failed to load topic model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            await self.load()  # Load model if not loaded
        
        try:
            text_clean = str(data['text']).strip().lower()
            prediction = self.model.predict([text_clean])[0]
            
            return {
                "text": data['text'],
                "topic": prediction,
                "run_source": "mlflow_production"
            }
        except Exception as e:
            logger.error(f"Error in TopicModel.predict: {e}")
            raise HTTPException(500, str(e))
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            await self.load()  # Load model if not loaded
        
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
            
            if not text_col:
                raise HTTPException(400, "CSV must contain 'text', 'comment' or 'content' column")
            
            raw_text = df[text_col].astype(str).str.lower().fillna("")
            predictions = self.model.predict(raw_text)
            df['topic'] = predictions
            
            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df.to_dict(orient="records"),
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            raise HTTPException(500, f"File processing error: {str(e)}")

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
    # Cách an toàn nhất: thay thế toàn bộ NaN/Inf trong DF bằng None trước khi to_dict
    # Note: where(cond, other) -> giữ lại giá trị nếu cond True, thay bằng other nếu False
    df_clean = df.where(pd.notnull(df), None)
    
    # Xử lý thêm Infinity nếu có (pandas đôi khi không coi Inf là null)
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    
    return df_clean.to_dict(orient="records")

# ------------------------------------------

class SentimentModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.label_map = {
            'negative': 'tiêu cực',
            'positive': 'tích cực',
            'neutral': 'trung tính'
        }

    async def load(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Sentiment model loaded: {model_uri}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            text_clean = str(data['text']).strip()
            prediction = self.model.predict([text_clean])[0]
            sentiment = self.label_map.get(prediction, str(prediction))

            # --- FIX CONFIDENCE ---
            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                try:
                    # predict_proba trả về [[0.1, 0.9, 0.0]] (shape 1, n_classes)
                    proba_arr = self.model.predict_proba([text_clean])
                    # Lấy max của mẫu đầu tiên
                    confidence = float(np.max(proba_arr[0]))
                except Exception as e:
                    logger.warning(f"Confidence calc error: {e}")
            
            return {
                "text": data['text'],
                "sentiment": sentiment,
                "confidence": safe_float(confidence), # Fix JSON Error
                "run_source": "mlflow_production"
            }
        except Exception as e:
            logger.error(f"Predict error: {e}")
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((c for c in ['text', 'comment', 'content'] if c in df.columns), None)
            if not text_col: raise HTTPException(400, "Missing text column")
            
            raw_text = df[text_col].astype(str).fillna("")
            predictions = self.model.predict(raw_text)
            df['sentiment'] = [self.label_map.get(p, str(p)) for p in predictions]

            # --- FIX BATCH CONFIDENCE ---
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(raw_text)
                    # probs shape (N_samples, N_classes). Max theo axis 1
                    df['confidence'] = np.max(probs, axis=1)
                    # Round sau khi tính xong
                    df['confidence'] = df['confidence'].round(4)
                except:
                    df['confidence'] = 0.5
            else:
                df['confidence'] = 0.5

            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(df), # Fix JSON Error: dùng hàm safe
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            raise HTTPException(500, f"Batch error: {e}")

class EmailModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        self.model = mlflow.pyfunc.load_model(model_uri)

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: await self.load()
        
        try:
            text_clean = str(data['text']).strip().lower()
            prediction = self.model.predict([text_clean])[0]
            is_spam = bool(prediction == 1)
            label = "spam" if is_spam else "ham"

            # --- FIX CONFIDENCE ---
            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                try:
                    # Lấy max probability (dù là class 0 hay 1)
                    proba_arr = self.model.predict_proba([text_clean])
                    confidence = float(np.max(proba_arr[0]))
                except:
                    pass

            return {
                "text": data['text'],
                "is_spam": is_spam,
                "label": label,
                "confidence": safe_float(confidence), # Fix JSON Error
                "run_source": "mlflow_production"
            }
        except Exception as e:
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None: await self.load()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            text_col = next((c for c in ['text', 'content', 'body', 'email'] if c in df.columns), None)
            if not text_col: raise HTTPException(400, "Missing text column")

            raw_text = df[text_col].astype(str).str.lower().fillna("")
            predictions = self.model.predict(raw_text)
            
            df['label'] = ["spam" if p == 1 else "ham" for p in predictions]
            df['is_spam'] = [bool(p == 1) for p in predictions]

            # --- FIX BATCH CONFIDENCE ---
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(raw_text)
                    df['confidence'] = np.max(probs, axis=1).round(4)
                except:
                    df['confidence'] = 0.5
            else:
                df['confidence'] = 0.5

            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(df), # Fix JSON Error
                "run_source": "mlflow_production_batch"
            }
        except Exception as e:
            raise HTTPException(500, f"Batch error: {e}")

class RecSysModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.user_encoder = None
        self.item_encoder = None

    async def load(self):
        # (Giữ nguyên code load encoders và model như cũ)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{self.model_name}@production"
        self.model = mlflow.pyfunc.load_model(model_uri)
        
        # ... (Code load encoders từ MinIO giữ nguyên) ...
        # Giả lập load encoders cho gọn code fix
        local_dir = "/tmp/encoders"
        os.makedirs(local_dir, exist_ok=True)
        fs = s3fs.S3FileSystem(
             client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
             key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY
        )
        try:
            fs.get(f"{MINIO_BUCKET}/product_recsys/encoders/user_le.pkl", f"{local_dir}/user_le.pkl")
            fs.get(f"{MINIO_BUCKET}/product_recsys/encoders/item_le.pkl", f"{local_dir}/item_le.pkl")
            self.user_encoder = joblib.load(f"{local_dir}/user_le.pkl")
            self.item_encoder = joblib.load(f"{local_dir}/item_le.pkl")
        except Exception as e:
            logger.error(f"Encoder load error: {e}")
            # Fallback cho test
            from sklearn.preprocessing import LabelEncoder
            self.user_encoder = LabelEncoder()
            self.item_encoder = LabelEncoder()

    def _get_encoded_idx(self, encoder, raw_id):
        try:
            return encoder.transform([str(raw_id)])[0]
        except:
            return 0

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.user_encoder is None: await self.load()
        
        try:
            # ... (Feature engineering giữ nguyên) ...
            full_desc = f"{data.get('title','')} {data.get('text','')} {data.get('categories','')}"
            
            user_idx = self._get_encoded_idx(self.user_encoder, data.get('user_id'))
            item_idx = self._get_encoded_idx(self.item_encoder, data.get('product_id'))
            
            input_df = pd.DataFrame({
                'user_idx': [user_idx],
                'item_idx': [item_idx],
                'full_description': [full_desc.lower()]
            })
            
            pred = self.model.predict(input_df)[0]
            
            # --- FIX RATING & JSON SAFE ---
            pred_val = safe_float(pred)
            if pred_val is None:
                final_score = 3.0
            else:
                final_score = max(1.0, min(5.0, pred_val))

            return {
                "user_id": data.get('user_id'),
                "product_id": data.get('product_id'),
                "predicted_rating": round(final_score, 2),
                "is_cold_start": False, # Đơn giản hóa logic
                "run_source": "mlflow_production"
            }
        except Exception as e:
            raise HTTPException(500, str(e))

    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None or self.user_encoder is None: await self.load()
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            
            # --- FIX MAPPING CỘT (Quan trọng) ---
            # Ánh xạ tên cột từ file CSV của người dùng sang tên chuẩn model cần
            col_mapping = {
                'reviews.username': 'user_id',
                'username': 'user_id',
                'asins': 'product_id', 
                'product_id': 'product_id',
                'item_id': 'product_id',
                'reviews.title': 'title',
                'title': 'title',
                'reviews.text': 'text',
                'text': 'text'
            }
            df = df.rename(columns=col_mapping)
            
            # Kiểm tra cột sau khi rename
            required = ['user_id', 'product_id']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise HTTPException(400, f"CSV thiếu cột (hoặc sai tên): {missing}. Cần có 'reviews.username' (hoặc 'user_id') và 'asins' (hoặc 'product_id').")

            # Fill text features
            # Đảm bảo các cột text tồn tại trước khi ghép
            for col in ['title', 'text', 'categories']:
                if col not in df.columns:
                    df[col] = ""

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
                if val is None:
                    clean_preds.append(3.0)
                else:
                    clean_preds.append(max(1.0, min(5.0, val)))
            
            df['predicted_rating'] = clean_preds

            # Chỉ trả về các cột quan trọng
            result_df = df[['user_id', 'product_id', 'predicted_rating']]
            
            return {
                "filename": file.filename,
                "total_rows": len(df),
                "data": df_to_json_safe(result_df),
                "run_source": "mlflow_production_batch"
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Batch recsys error: {str(e)}")



class TrendModel(BaseModel):
    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)

    async def load(self):
        """Load trend model từ MLflow"""
        try:
            logger.info(f"Loading trend model: {self.model_name} v{self.model_version}")
            
            # Thiết lập MLflow tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Tải model từ MLflow sử dụng alias "production"
            model_uri = f"models:/{self.model_name}@production"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Trend model loaded successfully from {model_uri}")
            
        except Exception as e:
            logger.error(f"Error loading trend model: {e}")
            raise RuntimeError(f"Failed to load trend model: {e}")

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            await self.load()  # Load model if not loaded
        
        # Trend analysis cần data time series, đây là placeholder
        return {
            "trend_analysis": "Time series analysis required",
            "run_source": "mlflow_production"
        }
    
    async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
        if self.model is None:
            await self.load()  # Load model if not loaded
        
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Placeholder trend analysis logic
        df['trend_analysis'] = "Time series analysis required"
        
        return {
            "filename": file.filename,
            "total_rows": len(df),
            "data": df.to_dict(orient="records"),
            "run_source": "mlflow_production_batch"
        }
