"""
Example cách sử dụng metrics recording functions trong model inference.

File này minh họa cách integrate metrics recording vào model inference process.
"""
import asyncio
from app.core.metrics import (
    record_sentiment_metrics,
    record_email_metrics,
    record_recsys_metrics,
    record_topic_metrics,
    record_trend_metrics,
    record_batch_processing_metrics,
    record_model_load_metrics
)

# Example 1: Sentiment Model Integration
class SentimentModelWithMetrics:
    """Sentiment model với metrics integration"""
    
    def __init__(self):
        self.load_time = None
    
    async def load(self):
        """Simulate model loading với metrics"""
        import time
        start_time = time.time()
        
        # Simulate model loading
        await self._load_model()
        
        load_time = time.time() - start_time
        record_model_load_metrics("sentiment", load_time, success=True)
        self.load_time = load_time
    
    async def _load_model(self):
        """Actual model loading logic (simulation)"""
        await asyncio.sleep(0.1)  # Simulate loading time
    
    async def predict(self, text: str):
        """Predict sentiment với metrics"""
        import time
        import random
        
        start_time = time.time()
        
        # Simulate inference
        sentiment = random.choice(["positive", "negative", "neutral"])
        confidence = random.uniform(0.5, 1.0)
        
        # Record metrics
        record_sentiment_metrics(sentiment, confidence)
        
        processing_time = time.time() - start_time
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "processing_time": processing_time
        }

# Example 2: Email Model Integration
class EmailModelWithMetrics:
    """Email spam detection model với metrics"""
    
    async def predict(self, email_text: str):
        """Predict email classification với metrics"""
        import random
        
        # Simulate inference
        is_spam = random.choice([True, False])
        confidence = random.uniform(0.6, 0.95)
        
        # Record metrics
        record_email_metrics(is_spam, confidence)
        
        return {
            "is_spam": is_spam,
            "confidence": confidence
        }

# Example 3: RecSys Model Integration  
class RecSysModelWithMetrics:
    """Recommendation system model với metrics"""
    
    async def predict(self, user_id: str, product_id: str):
        """Predict rating với metrics"""
        import random
        
        # Simulate inference
        predicted_rating = random.uniform(1.0, 5.0)
        is_cold_start = random.choice([True, False])
        
        # Record metrics
        record_recsys_metrics(predicted_rating, is_cold_start)
        
        return {
            "user_id": user_id,
            "product_id": product_id,
            "predicted_rating": round(predicted_rating, 2),
            "is_cold_start": is_cold_start
        }

# Example 4: Batch Processing Integration
class BatchProcessor:
    """Batch processing với metrics"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
    
    async def process_batch(self, data_list):
        """Process batch data với comprehensive metrics"""
        import time
        
        start_time = time.time()
        success = True
        
        try:
            # Simulate batch processing
            results = []
            for item in data_list:
                # Simulate processing each item
                await asyncio.sleep(0.01)
                results.append({"processed": True})
            
            processing_time = time.time() - start_time
            
            # Record batch metrics
            record_batch_processing_metrics(
                model_type=self.model_type,
                batch_size=len(data_list),
                processing_time=processing_time,
                success=success
            )
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            success = False
            
            # Record failed batch metrics
            record_batch_processing_metrics(
                model_type=self.model_type,
                batch_size=len(data_list),
                processing_time=processing_time,
                success=success
            )
            
            raise e

# Example 5: Controller Integration Pattern
from fastapi import FastAPI, UploadFile
from app.models.inference import SentimentModel
import time

class MetricsIntegratedController:
    """Pattern để integrate metrics vào existing controllers"""
    
    def __init__(self):
        self.model = SentimentModel("sentiment_model", "v1")
    
    async def predict_single(self, text: str):
        """Single prediction với metrics"""
        start_time = time.time()
        
        try:
            # Existing model inference logic
            result = await self.model.predict({"text": text})
            
            # Record metrics
            if "sentiment" in result and "confidence" in result:
                record_sentiment_metrics(result["sentiment"], result["confidence"])
            
            return result
            
        except Exception as e:
            # Record error metrics if needed
            raise e
    
    async def predict_batch(self, file: UploadFile):
        """Batch prediction với comprehensive metrics"""
        start_time = time.time()
        
        try:
            # Existing batch inference logic
            result = await self.model.predict_batch(file)
            
            processing_time = time.time() - start_time
            
            # Record batch metrics
            if "total_rows" in result:
                record_batch_processing_metrics(
                    model_type="sentiment",
                    batch_size=result["total_rows"],
                    processing_time=processing_time,
                    success=True
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record failed batch metrics
            if hasattr(file, 'filename'):
                record_batch_processing_metrics(
                    model_type="sentiment",
                    batch_size=0,  # Unknown size if failed
                    processing_time=processing_time,
                    success=False
                )
            
            raise e

# Example 6: Usage in FastAPI Route
"""
from fastapi import APIRouter, UploadFile, File
from app.examples.metrics_example import MetricsIntegratedController

router = APIRouter()
controller = MetricsIntegratedController()

@router.post("/predict_single")
async def predict_single_route(data: dict):
    return await controller.predict_single(data["text"])

@router.post("/predict_batch") 
async def predict_batch_route(file: UploadFile = File(...)):
    return await controller.predict_batch(file)
"""

# Example 7: Monitoring Dashboard Queries
"""
Useful Prometheus queries for monitoring:

# Sentiment model confidence score distribution
model_sentiment_confidence_score

# Email spam detection rate
rate(model_email_spam_detected_total[5m]) / 
(rate(model_email_spam_detected_total[5m]) + rate(model_email_ham_detected_total[5m]))

# RecSys cold start ratio
rate(model_recsys_cold_start_total[5m]) / rate(model_recsys_prediction_count_total[5m])

# Model prediction latency
rate(model_recsys_latency_seconds_sum[5m]) / rate(model_recsys_latency_seconds_count[5m])

# Batch processing throughput
sum(rate(batch_processed_records_total[5m])) by (model_type)
"""
