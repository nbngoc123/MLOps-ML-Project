# Hướng Dẫn Hệ Thống Monitoring NexusML

## 📋 Tổng Quan

Tài liệu này mô tả chi tiết hệ thống monitoring end-to-end cho NexusML ML pipeline, từ model inference đến infrastructure monitoring với Prometheus, Grafana, và AlertManager.

## 🏗️ Kiến Trúc Hệ Thống Monitoring

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │───▶│   Backend API   │───▶│   Prometheus    │
│  (Inference)    │    │  (FastAPI)      │    │   Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │  Health Check   │    │   Grafana       │
         └─────────────▶│   Endpoint      │───▶│   Dashboards    │
                        └─────────────────┘    └─────────────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │  Infrastructure │    │  AlertManager   │
                        │  (Redis/MinIO)  │    │   (Alerts)      │
                        └─────────────────┘    └─────────────────┘
```

## 🔧 Components Đã Xây Dựng

### 1. Core Monitoring Infrastructure

#### `app/core/monitoring.py`
**Prometheus metrics definitions và system monitoring:**
```python
# System metrics
cpu_usage_percent = Gauge("system_cpu_usage_percent", "CPU usage percentage")
memory_usage_bytes = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
api_request_count = Counter("api_request_count_total", "Total API requests")
api_request_latency = Histogram("api_request_latency_seconds", "API request latency")

# Model-specific metrics
model_sentiment_prediction_count = Counter("model_sentiment_prediction_count", 
    "Sentiment predictions count", ["sentiment_label"])
model_sentiment_confidence_score = Histogram("model_sentiment_confidence_score",
    "Sentiment confidence scores")
```

#### `app/core/metrics.py`
**Business logic metrics recording functions:**
```python
# Core metrics functions
record_sentiment_metrics(sentiment: str, confidence: float)
record_email_metrics(is_spam: bool, confidence: float)
record_recsys_metrics(predicted_rating: float, is_cold_start: bool)
record_topic_metrics(topic: str)
record_trend_metrics(processing_time: float)
record_batch_processing_metrics(model_type: str, batch_size: int, processing_time: float, success: bool)
record_model_load_metrics(model_type: str, load_time: float, success: bool)
```

### 2. Model Integration

#### `app/models/inference.py`
**5 model types với comprehensive metrics integration:**

**SentimentModel**
```python
async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing logic ...
    record_sentiment_metrics(sentiment, confidence)

async def predict_batch(self, file: UploadFile) -> Dict[str, Any]:
    start_time = time.time()
    # ... processing logic ...
    processing_time = time.time() - start_time
    record_batch_processing_metrics("sentiment", len(df), processing_time, True)

async def load(self):
    # ... model loading ...
    record_model_load_metrics("sentiment", load_time, success=True)
```

**EmailModel**
- Email spam detection metrics
- Batch processing metrics với timing
- Model load performance metrics

**RecSysModel** 
- Rating prediction metrics
- Cold start detection metrics
- User/Item encoder performance tracking

**TopicModel**
- Topic classification metrics
- Topic distribution tracking

**TrendModel**
- Time series processing metrics
- Placeholder cho advanced trend analysis

### 3. Health Monitoring

#### `app/api/controllers/health.py`
**Comprehensive health check endpoint:**
```python
@classmethod
async def check_health(cls):
    # Real Redis connection check
    redis_client.ping()
    
    # Real MinIO connection check  
    minio_client.client.bucket_exists(settings.MINIO_BUCKET)
    
    # Prometheus metrics integration
    REDIS_CONNECTED.set(redis_status)
    MINIO_CONNECTED.set(minio_status)
    
    return health_status
```

### 4. Monitoring Examples & Documentation

#### `app/examples/metrics_example.py`
**Integration patterns và best practices:**
```python
class MetricsIntegratedController:
    async def predict_single(self, text: str):
        result = await self.model.predict({"text": text})
        
        if "sentiment" in result and "confidence" in result:
            record_sentiment_metrics(result["sentiment"], result["confidence"])
        
        return result

    async def predict_batch(self, file: UploadFile):
        start_time = time.time()
        result = await self.model.predict_batch(file)
        
        processing_time = time.time() - start_time
        record_batch_processing_metrics(
            model_type="sentiment",
            batch_size=result["total_rows"],
            processing_time=processing_time,
            success=True
        )
        return result
```

### 5. Prometheus Configuration

#### `monitor/prometheus/prometheus.yml`
**Complete Prometheus setup:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nexusml-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

#### Alert Rules (`monitor/prometheus/alert_rules.yml`)
```yaml
groups:
- name: nexusml-alerts
  rules:
  - alert: HighModelLatency
    expr: model_prediction_latency_seconds{quantile="0.95"} > 5
    for: 5m
    annotations:
      summary: "High model prediction latency"
      
  - alert: ModelLoadFailure
    expr: model_load_success_total == 0
    for: 1m
    annotations:
      summary: "Model loading failures detected"
```

### 6. Grafana Dashboards

#### Key Dashboard Metrics

**1. Model Performance Dashboard**
- Prediction latency (p50, p95, p99)
- Prediction throughput (requests/second)
- Error rates by model type
- Confidence score distributions

**2. Infrastructure Health Dashboard**
- Redis connection status
- MinIO storage health
- System resource usage (CPU, RAM)
- API request patterns

**3. Business Metrics Dashboard**
- Sentiment analysis trends
- Email spam detection rates
- RecSys rating distributions
- Topic classification patterns

### 7. AlertManager Configuration

#### `monitor/alertmanager/alertmanager.yml`
```yaml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'
```

## 📊 Metrics Catalog

### Model-Specific Metrics

#### Sentiment Analysis
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_sentiment_prediction_count` | Counter | Total sentiment predictions | `sentiment_label` |
| `model_sentiment_confidence_score` | Histogram | Confidence distribution | - |

#### Email Classification
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_email_spam_detected_total` | Counter | Total spam emails detected | - |
| `model_email_ham_detected_total` | Counter | Total ham emails detected | - |

#### Recommendation System
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_recsys_predicted_rating` | Histogram | Rating predictions | - |
| `model_recsys_cold_start_total` | Counter | Cold start recommendations | - |

#### Topic Classification
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_topic_prediction_count` | Counter | Topic predictions | `topic_label` |

#### Trend Analysis
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_trend_processing_seconds` | Histogram | Processing time | - |

### Infrastructure Metrics

#### API Performance
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `api_request_count_total` | Counter | Total API requests | `endpoint`, `method`, `status` |
| `api_request_latency_seconds` | Histogram | Request latency | `endpoint` |

#### System Health
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `infra_redis_connected` | Gauge | Redis connection status | - |
| `infra_minio_connected` | Gauge | MinIO connection status | - |
| `system_cpu_usage_percent` | Gauge | CPU usage percentage | - |
| `system_memory_usage_bytes` | Gauge | Memory usage in bytes | - |

### Batch Processing Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `batch_processing_count_total` | Counter | Total batch processes | `model_type`, `status` |
| `batch_processing_duration_seconds` | Histogram | Batch processing time | `model_type` |
| `batch_records_processed_total` | Counter | Total records processed | `model_type` |

### Model Loading Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `model_load_duration_seconds` | Histogram | Model loading time | `model_type` |
| `model_load_success_total` | Counter | Successful loads | `model_type` |
| `model_load_failure_total` | Counter | Failed loads | `model_type` |

## 🚀 Implementation Guide

### 1. Setting Up Monitoring Stack

```bash
# Start monitoring services
cd monitor
docker-compose up -d

# Access Grafana (default credentials: admin/admin)
open http://localhost:3000

# Access Prometheus
open http://localhost:9090
```

### 2. Backend API Integration

Add to `app/main.py`:
```python
from app.core.monitoring import PrometheusMiddleware, update_system_metrics

app.add_middleware("http")(PrometheusMiddleware())

@app.get("/metrics")
async def metrics():
    update_system_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### 3. Model Integration Pattern

```python
# In your model class
from app.core.metrics import record_model_metrics

class YourModel(BaseModel):
    async def load(self):
        start_time = time.time()
        try:
            # Load your model
            self.model = await self._load_model()
            load_time = time.time() - start_time
            record_model_load_metrics("your_model", load_time, success=True)
        except Exception as e:
            record_model_load_metrics("your_model", 0.0, success=False)
            raise

    async def predict(self, data):
        result = await self._predict(data)
        record_your_model_metrics(result)
        return result
```

### 4. Adding Custom Metrics

```python
from app.core.metrics import record_custom_metric
from prometheus_client import Counter, Histogram

# Define custom metrics
custom_counter = Counter('custom_operations_total', 'Custom operations', ['operation_type'])
custom_histogram = Histogram('custom_duration_seconds', 'Custom duration', ['operation_type'])

# Record metrics
custom_counter.labels(operation_type='processing').inc()
custom_histogram.labels(operation_type='processing').observe(duration)
```

## 📈 Dashboard Queries

### Key Prometheus Queries

**Model Latency (95th percentile):**
```promql
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))
```

**Error Rate by Model:**
```promql
rate(model_prediction_errors_total[5m]) by (model_type)
```

**Throughput by Model:**
```promql
rate(model_predictions_total[5m]) by (model_type)
```

**System Resource Usage:**
```promql
# CPU Usage
system_cpu_usage_percent

# Memory Usage  
system_memory_usage_bytes / (1024 * 1024 * 1024)  # Convert to GB
```

**API Health:**
```promql
# Request rate
rate(api_request_count_total[5m])

# Error rate
rate(api_request_count_total{status=~"5.."}[5m]) / rate(api_request_count_total[5m])
```

### Grafana Dashboard JSON

Import from `monitor/grafana/dashboards/` for pre-built dashboards:

1. **Model Performance Dashboard** (`vqa-evidently-dashboard.json`)
2. **Infrastructure Dashboard** (`yolo-evidently-dashboard.json`)

## 🔔 Alerting Strategy

### Critical Alerts

1. **Model Service Down**
   ```yaml
   alert: ModelServiceDown
   expr: up{job="nexusml-backend"} == 0
   for: 1m
   ```

2. **High Error Rate**
   ```yaml
   alert: HighErrorRate
   expr: rate(model_prediction_errors_total[5m]) > 0.1
   for: 5m
   ```

### Warning Alerts

1. **High Latency**
   ```yaml
   alert: HighLatency
   expr: histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m])) > 2
   for: 10m
   ```

2. **Low Model Confidence**
   ```yaml
   alert: LowModelConfidence
   expr: avg_over_time(model_sentiment_confidence_score[1h]) < 0.6
   for: 30m
   ```

### Infrastructure Alerts

1. **Redis Connection Loss**
   ```yaml
   alert: RedisDown
   expr: infra_redis_connected == 0
   for: 1m
   ```

2. **High Memory Usage**
   ```yaml
   alert: HighMemoryUsage
   expr: system_memory_usage_bytes / 1024 / 1024 / 1024 > 8
   for: 10m
   ```

## 🛠️ Best Practices

### 1. Metric Naming
- Use snake_case for metric names
- Include units in metric name (`_seconds`, `_bytes`, `_total`)
- Use clear, descriptive names
- Follow Prometheus naming conventions

### 2. Labels Strategy
- Keep label cardinality low (< 10 unique values per label)
- Avoid labels with high cardinality (e.g., user IDs, timestamps)
- Use job and instance labels automatically

### 3. Performance
- Use appropriate metric types (Counter for rates, Histogram for distributions)
- Avoid excessive metric collection in hot paths
- Use aggregation in queries rather than in application code

### 4. Reliability
- Always wrap metrics calls in try/catch
- Use default values when metrics recording fails
- Monitor metrics endpoint health

### 5. Security
- Restrict access to `/metrics` endpoint
- Sanitize sensitive data from metrics
- Use TLS for metrics transmission

## 📝 Maintenance

### Regular Tasks

1. **Metric Review**
   - Review unused metrics quarterly
   - Update alert thresholds based on production data
   - Clean up obsolete labels and metrics

2. **Dashboard Updates**
   - Add new metrics as features are added
   - Update queries based on data patterns
   - Review dashboard effectiveness

3. **Alert Tuning**
   - Analyze alert fatigue and adjust thresholds
   - Update notification channels
   - Review alert escalation procedures

### Troubleshooting

**Common Issues:**

1. **Metrics Not Appearing**
   - Check `/metrics` endpoint accessibility
   - Verify Prometheus scrape configuration
   - Check metric name and label consistency

2. **High Cardinality Issues**
   - Review label values with many unique entries
   - Consider removing or aggregating high-cardinality labels

3. **Performance Impact**
   - Profile metrics collection overhead
   - Optimize frequent metric updates
   - Use sampling for high-volume metrics

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Libraries](https://prometheus.io/docs/instrumenting/clientlibs/)
- [Best Practices](https://prometheus.io/docs/practices/naming/)

---

**Generated on:** $(date)
**Version:** 1.0.0
**Last Updated:** 2024-11-22
