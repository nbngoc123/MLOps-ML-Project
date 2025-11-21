"""
Metrics recording helper functions cho các model types.
Các hàm này được gọi sau khi model inference hoàn thành để record metrics.
"""

from app.core.monitoring import (
    # Sentiment metrics
    SENTIMENT_PREDICTION_COUNT, SENTIMENT_CONFIDENCE_SCORE,
    
    # Email metrics  
    EMAIL_SPAM_DETECTED, EMAIL_HAM_DETECTED,
    
    # RecSys metrics
    RECSYS_COLD_START_COUNT, RECSYS_PREDICTED_RATING,
    
    # Topic metrics
    TOPIC_PREDICTION_COUNT,
    
    # Trend metrics
    TREND_PROCESSING_TIME,

    MODEL_LOAD_DURATION, MODEL_LOAD_STATUS,
    BATCH_PROCESSING_SIZE, BATCH_PROCESSING_LATENCY
)
import time
import logging

logger = logging.getLogger(__name__)

def record_sentiment_metrics(sentiment: str, confidence: float):
    """Gọi hàm này sau khi model sentiment trả kết quả"""
    try:
        # Validate inputs
        if not sentiment or not isinstance(sentiment, str):
            logger.warning(f"Invalid sentiment value: {sentiment}")
            return
        
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence value: {confidence}")
            confidence = 0.5  # Default confidence if invalid
        
        # Record metrics
        SENTIMENT_PREDICTION_COUNT.labels(sentiment_label=sentiment).inc()
        SENTIMENT_CONFIDENCE_SCORE.observe(confidence)
        
        logger.debug(f"Recorded sentiment metrics: {sentiment}, confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"Error recording sentiment metrics: {e}")

def record_email_metrics(is_spam: bool, confidence: float):
    """Gọi hàm này sau khi model email trả kết quả"""
    try:
        # Validate inputs
        if not isinstance(is_spam, bool):
            logger.warning(f"Invalid is_spam value: {is_spam}")
            is_spam = False  # Default to not spam if invalid
        
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence value: {confidence}")
            confidence = 0.5  # Default confidence if invalid
        
        # Record metrics
        if is_spam:
            EMAIL_SPAM_DETECTED.inc()
        else:
            EMAIL_HAM_DETECTED.inc()
        
        logger.debug(f"Recorded email metrics: is_spam={is_spam}, confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"Error recording email metrics: {e}")

def record_recsys_metrics(predicted_rating: float, is_cold_start: bool = False):
    """Gọi hàm này sau khi model recsys trả kết quả"""
    try:
        # Validate inputs
        if not isinstance(predicted_rating, (int, float)):
            logger.warning(f"Invalid predicted_rating value: {predicted_rating}")
            predicted_rating = 3.0  # Default rating if invalid
        else:
            # Clamp rating to valid range [1.0, 5.0]
            predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        if not isinstance(is_cold_start, bool):
            logger.warning(f"Invalid is_cold_start value: {is_cold_start}")
            is_cold_start = False
        
        # Record metrics
        RECSYS_PREDICTED_RATING.observe(predicted_rating)
        
        if is_cold_start:
            RECSYS_COLD_START_COUNT.inc()
        
        logger.debug(f"Recorded recsys metrics: rating={predicted_rating}, cold_start={is_cold_start}")
        
    except Exception as e:
        logger.error(f"Error recording recsys metrics: {e}")

def record_topic_metrics(topic: str):
    """Gọi hàm này sau khi model topic trả kết quả"""
    try:
        # Validate inputs
        if not topic or not isinstance(topic, str):
            logger.warning(f"Invalid topic value: {topic}")
            topic = "unknown"  # Default topic if invalid
        
        # Record metrics
        TOPIC_PREDICTION_COUNT.labels(topic_label=topic).inc()
        
        logger.debug(f"Recorded topic metrics: {topic}")
        
    except Exception as e:
        logger.error(f"Error recording topic metrics: {e}")

def record_trend_metrics(processing_time: float):
    """Gọi hàm này sau khi model trend trả kết quả"""
    try:
        # Validate inputs
        if not isinstance(processing_time, (int, float)) or processing_time < 0:
            logger.warning(f"Invalid processing_time value: {processing_time}")
            processing_time = 0.0  # Default processing time if invalid
        
        # Record metrics
        TREND_PROCESSING_TIME.observe(processing_time)
        
        logger.debug(f"Recorded trend metrics: processing_time={processing_time}s")
        
    except Exception as e:
        logger.error(f"Error recording trend metrics: {e}")

def record_batch_processing_metrics(model_type: str, batch_size: int, processing_time: float, success: bool = True):
    """Ghi nhận metrics xử lý batch"""
    try:
        # Validate inputs
        if batch_size <= 0: batch_size = 1
        if processing_time < 0: processing_time = 0.0
        
        if success:
            BATCH_PROCESSING_SIZE.labels(model_type=model_type).observe(batch_size)
            BATCH_PROCESSING_LATENCY.labels(model_type=model_type).observe(processing_time)
        
        throughput = batch_size / processing_time if processing_time > 0 else 0
        logger.info(f"Batch metrics - Model: {model_type}, Size: {batch_size}, Time: {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error recording batch metrics: {e}")


def record_model_load_metrics(model_type: str, load_time: float, success: bool = True):
    """Ghi nhận metrics load model"""
    try:
        if load_time < 0: load_time = 0.0
        
        status = "success" if success else "failed"
        
        # GỬI LÊN PROMETHEUS ---
        MODEL_LOAD_STATUS.labels(model_type=model_type, status=status).inc()
        
        if success:
            MODEL_LOAD_DURATION.labels(model_type=model_type).set(load_time)
        
        logger.info(f"Model load metrics - Model: {model_type}, Time: {load_time:.2f}s, Status: {status}")
        
    except Exception as e:
        logger.error(f"Error recording load metrics: {e}")


