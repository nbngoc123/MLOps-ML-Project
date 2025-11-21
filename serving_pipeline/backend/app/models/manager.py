# import logging
# import os
# import joblib
# import asyncio
# import pandas as pd
# import shutil
# import mlflow
# from io import BytesIO
# from typing import Any, Dict, List, Optional, Type
# from fastapi import HTTPException, UploadFile, File
# from pydantic import BaseModel as PydanticBaseModel
# from mlflow.tracking import MlflowClient
# from app.models._base import BaseModel
# from app.core.config import settings

# # Logging setup
# logger = logging.getLogger("inference")

# # Cấu hình môi trường cho MLflow
# MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
# MINIO_ENDPOINT = settings.MINIO_ENDPOINT
# MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
# MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY

# # Thiết lập biến môi trường cho MLflow/Boto3
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
# os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
# os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
# os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# # Lớp SentimentModel kế thừa từ BaseModel
# class SentimentModel(BaseModel):
#     def __init__(self, model_name: str, model_version: str):
#         super().__init__(model_name, model_version)
#         self.vectorizer = None
#         self.label_map = {
#             'negative': 'tiêu cực',
#             'positive': 'tích cực',
#             'neutral': 'trung tính'
#         }

#     async def load(self):
#         """Load sentiment model và vectorizer từ MLflow"""
#         try:
#             logger.info(f"Loading sentiment model: {self.model_name} v{self.model_version}")
            
#             # Thiết lập MLflow tracking URI
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
#             # Tải model từ MLflow sử dụng alias "production"
#             model_uri = f"models:/{self.model_name}@production"
#             self.model = mlflow.pyfunc.load_model(model_uri)
            
#             logger.info(f"Sentiment model loaded successfully from {model_uri}")
            
#         except Exception as e:
#             logger.error(f"Error loading sentiment model: {e}")
#             raise RuntimeError(f"Failed to load sentiment model: {e}")

#     async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             text_clean = str(data['text']).strip()
#             prediction = self.model.predict([text_clean])[0]
#             sentiment = self.label_map.get(prediction, str(prediction))

#             confidence = 0.0
#             try:
#                 if hasattr(self.model, "predict_proba"):
#                     proba = self.model.predict_proba([text_clean])
#                     if proba is not None and len(proba) > 0:
#                         confidence = float(proba.max())
#                 else:
#                     # Nếu không có predict_proba, set confidence = 0.5 cho default
#                     confidence = 0.5
#             except Exception as e:
#                 logger.warning(f"Could not get confidence: {e}, using default 0.5")
#                 confidence = 0.5

#             return {
#                 "text": data['text'],
#                 "sentiment": sentiment,
#                 "confidence": round(confidence, 4),
#                 "run_source": "mlflow_production"
#             }
#         except Exception as e:
#             logger.error(f"Error in SentimentModel.predict: {e}")
#             raise HTTPException(500, str(e))
    
#     async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             content = await file.read()
#             df = pd.read_csv(BytesIO(content))

#             text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
#             if not text_col:
#                 raise HTTPException(400, "CSV must contain 'text', 'comment' or 'content' column")
            
#             raw_text = df[text_col].astype(str).fillna("")
#             predictions = self.model.predict(raw_text)
#             df['sentiment'] = [self.label_map.get(p, str(p)) for p in predictions]

#             try:
#                 if hasattr(self.model, "predict_proba"):
#                     probs = self.model.predict_proba(raw_text)
#                     if probs is not None and len(probs) > 0:
#                         df['confidence'] = probs.max(axis=1).round(4)
#                     else:
#                         df['confidence'] = 0.5
#                 else:
#                     df['confidence'] = 0.5
#             except Exception as e:
#                 logger.warning(f"Could not get batch confidence: {e}, using default 0.5")
#                 df['confidence'] = 0.5

#             return {
#                 "filename": file.filename,
#                 "total_rows": len(df),
#                 "data": df.to_dict(orient="records"),
#                 "run_source": "mlflow_production_batch"
#             }
#         except Exception as e:
#             raise HTTPException(500, f"File processing error: {str(e)}")

# class TopicModel(BaseModel):
#     def __init__(self, model_name: str, model_version: str):
#         super().__init__(model_name, model_version)

#     async def load(self):
#         """Load topic model từ MLflow"""
#         try:
#             logger.info(f"Loading topic model: {self.model_name} v{self.model_version}")
            
#             # Thiết lập MLflow tracking URI
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
#             # Tải model từ MLflow sử dụng alias "production"
#             model_uri = f"models:/{self.model_name}@production"
#             self.model = mlflow.pyfunc.load_model(model_uri)
            
#             logger.info(f"Topic model loaded successfully from {model_uri}")
            
#         except Exception as e:
#             logger.error(f"Error loading topic model: {e}")
#             raise RuntimeError(f"Failed to load topic model: {e}")

#     async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             text_clean = str(data['text']).strip().lower()
#             prediction = self.model.predict([text_clean])[0]
            
#             return {
#                 "text": data['text'],
#                 "topic": prediction,
#                 "run_source": "mlflow_production"
#             }
#         except Exception as e:
#             logger.error(f"Error in TopicModel.predict: {e}")
#             raise HTTPException(500, str(e))
    
#     async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             content = await file.read()
#             df = pd.read_csv(BytesIO(content))
#             text_col = next((col for col in ['text', 'comment', 'content'] if col in df.columns), None)
            
#             if not text_col:
#                 raise HTTPException(400, "CSV must contain 'text', 'comment' or 'content' column")
            
#             raw_text = df[text_col].astype(str).str.lower().fillna("")
#             predictions = self.model.predict(raw_text)
#             df['topic'] = predictions
            
#             return {
#                 "filename": file.filename,
#                 "total_rows": len(df),
#                 "data": df.to_dict(orient="records"),
#                 "run_source": "mlflow_production_batch"
#             }
#         except Exception as e:
#             raise HTTPException(500, f"File processing error: {str(e)}")

# class EmailModel(BaseModel):
#     def __init__(self, model_name: str, model_version: str):
#         super().__init__(model_name, model_version)

#     async def load(self):
#         """Load email model từ MLflow"""
#         try:
#             logger.info(f"Loading email model: {self.model_name} v{self.model_version}")
            
#             # Thiết lập MLflow tracking URI
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
#             # Tải model từ MLflow sử dụng alias "production"
#             model_uri = f"models:/{self.model_name}@production"
#             self.model = mlflow.pyfunc.load_model(model_uri)
            
#             logger.info(f"Email model loaded successfully from {model_uri}")
            
#         except Exception as e:
#             logger.error(f"Error loading email model: {e}")
#             raise RuntimeError(f"Failed to load email model: {e}")

#     async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             text_clean = str(data['text']).strip().lower()
#             prediction = self.model.predict([text_clean])[0]
            
#             # Map prediction (0=ham, 1=spam)
#             is_spam = bool(prediction == 1)
#             label = "spam" if is_spam else "ham"
            
#             confidence = 0.0
#             try:
#                 if hasattr(self.model, "predict_proba"):
#                     proba = self.model.predict_proba([text_clean])
#                     if proba is not None and len(proba) > 0:
#                         confidence = float(proba.max())
#                 else:
#                     confidence = 0.5
#             except Exception as e:
#                 logger.warning(f"Could not get confidence: {e}, using default 0.5")
#                 confidence = 0.5
            
#             return {
#                 "text": data['text'],
#                 "is_spam": is_spam,
#                 "label": label,
#                 "confidence": round(confidence, 4),
#                 "run_source": "mlflow_production"
#             }
#         except Exception as e:
#             logger.error(f"Error in EmailModel.predict: {e}")
#             raise HTTPException(500, str(e))
    
#     async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             content = await file.read()
#             df = pd.read_csv(BytesIO(content))
#             text_col = next((col for col in ['text', 'content', 'body', 'email'] if col in df.columns), None)
            
#             if not text_col:
#                 raise HTTPException(400, "CSV must contain 'text', 'content', 'body' or 'email' column")
            
#             raw_text = df[text_col].astype(str).str.lower().fillna("")
#             predictions = self.model.predict(raw_text)
            
#             # Map predictions
#             df['label'] = ["spam" if p == 1 else "ham" for p in predictions]
#             df['is_spam'] = [bool(p == 1) for p in predictions]
            
#             try:
#                 if hasattr(self.model, "predict_proba"):
#                     probs = self.model.predict_proba(raw_text)
#                     if probs is not None and len(probs) > 0:
#                         df['confidence'] = probs.max(axis=1).round(4)
#                     else:
#                         df['confidence'] = 0.5
#                 else:
#                     df['confidence'] = 0.5
#             except Exception as e:
#                 logger.warning(f"Could not get batch confidence: {e}, using default 0.5")
#                 df['confidence'] = 0.5
            
#             return {
#                 "filename": file.filename,
#                 "total_rows": len(df),
#                 "data": df.to_dict(orient="records"),
#                 "run_source": "mlflow_production_batch"
#             }
#         except Exception as e:
#             raise HTTPException(500, f"File processing error: {str(e)}")

# class RecSysModel(BaseModel):
#     def __init__(self, model_name: str, model_version: str):
#         super().__init__(model_name, model_version)

#     async def load(self):
#         """Load recsys model từ MLflow"""
#         try:
#             logger.info(f"Loading recsys model: {self.model_name} v{self.model_version}")
            
#             # Thiết lập MLflow tracking URI
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
#             # Tải model từ MLflow sử dụng alias "production"
#             model_uri = f"models:/{self.model_name}@production"
#             self.model = mlflow.pyfunc.load_model(model_uri)
            
#             logger.info(f"RecSys model loaded successfully from {model_uri}")
            
#         except Exception as e:
#             logger.error(f"Error loading recsys model: {e}")
#             raise RuntimeError(f"Failed to load recsys model: {e}")

#     async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         try:
#             # Feature engineering đồng bộ với training
#             full_desc_str = f"{data.get('title', '')} {data.get('text', '')} {data.get('name', '')} {data.get('manufacturer', '')} {data.get('categories', '')}"
            
#             # Tạo DataFrame input (cần implement logic encoding user/item IDs)
#             input_df = pd.DataFrame({
#                 'user_idx': [0],  # Placeholder - cần implement proper encoding
#                 'item_idx': [0],  # Placeholder - cần implement proper encoding
#                 'full_description': [full_desc_str.lower()] 
#             })
            
#             pred_rating = self.model.predict(input_df)[0]
#             final_score = max(1.0, min(5.0, float(pred_rating)))
            
#             return {
#                 "user_id": data.get('user_id'),
#                 "product_id": data.get('product_id'),
#                 "predicted_rating": round(final_score, 2),
#                 "is_cold_start": True,  # Placeholder
#                 "run_source": "mlflow_production"
#             }
#         except Exception as e:
#             logger.error(f"Error in RecSysModel.predict: {e}")
#             raise HTTPException(500, str(e))
    
#     def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Chuẩn hóa tên cột linh hoạt"""
#         # Map các biến thể về tên chuẩn
#         col_mapping = {
#             'reviews.username': 'user_id', 
#             'username': 'user_id',
#             'asins': 'product_id', 
#             'item_id': 'product_id',
#             'reviews.title': 'title',
#             'reviews.text': 'text'
#         }
#         df = df.rename(columns=col_mapping)
#         return df

#     def _get_encoded_idx(self, encoder, raw_id, default_idx=0):
#         """Helper để encode ID, xử lý trường hợp ID mới (Unseen)"""
#         try:
#             # Transform trả về array, lấy phần tử đầu tiên
#             return encoder.transform([str(raw_id)])[0]
#         except ValueError:
#             # Nếu ID chưa từng thấy lúc train -> Trả về default (ví dụ 0 hoặc user trung bình)
#             return default_idx

#     async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
#         if self.model is None or self.user_encoder is None:
#             await self.load()
        
#         try:
#             content = await file.read()
#             df = pd.read_csv(BytesIO(content))
            
#             # 1. Chuẩn hóa tên cột (Fix lỗi CSV validation)
#             df = self._map_columns(df)
            
#             # 2. Kiểm tra lại cột sau khi rename
#             required = ['user_id', 'product_id']
#             missing = [c for c in required if c not in df.columns]
#             if missing:
#                 raise HTTPException(400, f"CSV missing columns (checked variants): {missing}")

#             # 3. Feature Engineering (Text) - Đồng bộ logic với Training
#             text_cols = ['title', 'text', 'name', 'manufacturer', 'categories']
#             # Đảm bảo các cột text tồn tại để không lỗi
#             for col in text_cols:
#                 if col not in df.columns:
#                     df[col] = ""
            
#             df['full_description'] = (
#                 df['title'].astype(str) + " " + 
#                 df['text'].astype(str) + " " + 
#                 df['name'].astype(str) + " " + 
#                 df['manufacturer'].astype(str) + " " + 
#                 df['categories'].astype(str)
#             ).str.lower()

#             # 4. Encoding User/Item ID (Fix lỗi placeholder = 0)
#             # Sử dụng encoder đã load thay vì gán 0
#             df['user_idx'] = df['user_id'].apply(lambda x: self._get_encoded_idx(self.user_encoder, x))
#             df['item_idx'] = df['product_id'].apply(lambda x: self._get_encoded_idx(self.item_encoder, x))

#             # 5. Predict
#             pipeline_input = df[['user_idx', 'item_idx', 'full_description']]
            
#             # Model pipeline đã bao gồm cả preprocessing text, nên đưa thẳng vào predict
#             preds = self.model.predict(pipeline_input)
            
#             # Clip kết quả trong khoảng 1-5
#             df['predicted_rating'] = [max(1.0, min(5.0, float(p))) for p in preds]

#             return {
#                 "filename": file.filename,
#                 "total_rows": len(df),
#                 "data": df[['user_id', 'product_id', 'predicted_rating']].to_dict(orient="records"),
#                 "run_source": "mlflow_production_hybrid"
#             }

#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"Batch prediction error: {e}")
#             raise HTTPException(500, f"Processing error: {str(e)}")

# class TrendModel(BaseModel):
#     def __init__(self, model_name: str, model_version: str):
#         super().__init__(model_name, model_version)

#     async def load(self):
#         """Load trend model từ MLflow"""
#         try:
#             logger.info(f"Loading trend model: {self.model_name} v{self.model_version}")
            
#             # Thiết lập MLflow tracking URI
#             mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
#             # Tải model từ MLflow sử dụng alias "production"
#             model_uri = f"models:/{self.model_name}@production"
#             self.model = mlflow.pyfunc.load_model(model_uri)
            
#             logger.info(f"Trend model loaded successfully from {model_uri}")
            
#         except Exception as e:
#             logger.error(f"Error loading trend model: {e}")
#             raise RuntimeError(f"Failed to load trend model: {e}")

#     async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         # Trend analysis cần data time series, đây là placeholder
#         return {
#             "trend_analysis": "Time series analysis required",
#             "run_source": "mlflow_production"
#         }
    
#     async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
#         if self.model is None:
#             await self.load()  # Load model if not loaded
        
#         content = await file.read()
#         df = pd.read_csv(BytesIO(content))
        
#         # Placeholder trend analysis logic
#         df['trend_analysis'] = "Time series analysis required"
        
#         return {
#             "filename": file.filename,
#             "total_rows": len(df),
#             "data": df.to_dict(orient="records"),
#             "run_source": "mlflow_production_batch"
#         }
