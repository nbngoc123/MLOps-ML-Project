import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, CONTENT_TYPE_LATEST, generate_latest
from typing import Callable
from fastapi import Request, Response

# --- 1. SYSTEM & INFRASTRUCTURE METRICS ---
SYSTEM_CPU_USAGE = Gauge("system_cpu_usage_percent", "System CPU usage percentage")
SYSTEM_RAM_USAGE = Gauge("system_ram_usage_percent", "System RAM usage percentage")
PROCESS_MEMORY = Gauge("process_memory_usage_bytes", "Memory usage of the application process")

# --- 2. API PERFORMANCE METRICS ---
# Theo dõi tổng số request và latency chung
API_REQUEST_COUNT = Counter(
    "api_request_count_total", 
    "Total API requests", 
    ["method", "endpoint", "status_code"]
)
API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "API request latency in seconds", 
    ["method", "endpoint"]
)

# --- 3. MODEL-SPECIFIC BUSINESS METRICS ---

# A. Sentiment Analysis
SENTIMENT_PREDICTION_COUNT = Counter(
    "model_sentiment_prediction_count", 
    "Sentiment predictions breakdown", 
    ["sentiment_label"] # positive, negative, neutral
)
SENTIMENT_CONFIDENCE_SCORE = Histogram(
    "model_sentiment_confidence_score", 
    "Distribution of confidence scores for sentiment model", 
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# B. Recommender System (RecSys)
RECSYS_PREDICTION_LATENCY = Histogram(
    "model_recsys_latency_seconds",
    "Latency specifically for RecSys pipelines"
)
RECSYS_COLD_START_COUNT = Counter(
    "model_recsys_cold_start_total",
    "Number of cold-start recommendations served"
)
# Giả sử chúng ta track được rating predict ra nằm trong khoảng nào
RECSYS_PREDICTED_RATING = Histogram(
    "model_recsys_predicted_rating_dist",
    "Distribution of predicted ratings",
    buckets=[1.0, 2.0, 3.0, 4.0, 5.0]
)

# C. Email (Spam Detection)
EMAIL_SPAM_DETECTED = Counter(
    "model_email_spam_detected_total",
    "Total number of spam emails detected"
)
EMAIL_HAM_DETECTED = Counter(
    "model_email_ham_detected_total",
    "Total number of ham emails detected"
)

# D. Topic Classification
TOPIC_PREDICTION_COUNT = Counter(
    "model_topic_prediction_count",
    "Count of predicted topics",
    ["topic_label"]
)

# E. Trend Analysis
TREND_PROCESSING_TIME = Histogram(
    "model_trend_processing_seconds",
    "Time taken to process time-series data"
)

# --- 4. INFRASTRUCTURE DEPENDENCY HEALTH ---
REDIS_CONNECTED = Gauge("infra_redis_connected", "1 if Redis is connected, 0 otherwise")
MINIO_CONNECTED = Gauge("infra_minio_connected", "1 if MinIO is connected, 0 otherwise")


class PrometheusMiddleware:
    """Middleware để tự động đo lường latency và count cho mọi request"""
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Xử lý request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            process_time = time.time() - start_time
            
            # Lấy endpoint path (gom nhóm các path có ID, ví dụ /items/1 -> /items/{id})
            # Nếu route không tìm thấy, dùng "unknown"
            route = request.scope.get("route")
            path = route.path if route else request.url.path

            # Không log metrics endpoint để tránh nhiễu
            if path != "/metrics" and path != "/health":
                API_REQUEST_COUNT.labels(
                    method=request.method, 
                    endpoint=path, 
                    status_code=status_code
                ).inc()
                
                API_REQUEST_LATENCY.labels(
                    method=request.method, 
                    endpoint=path
                ).observe(process_time)
                
        return response

def update_system_metrics():
    """Hàm này nên được gọi định kỳ hoặc mỗi khi scrape /metrics"""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
    process = psutil.Process()
    PROCESS_MEMORY.set(process.memory_info().rss)