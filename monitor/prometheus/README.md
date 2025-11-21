# Prometheus Configuration for NexusML

## 📋 Tổng Quan

Thư mục này chứa cấu hình Prometheus cho hệ thống monitoring của NexusML ML pipeline, bao gồm alert rules và scraping configuration cho tất cả các services.

## 📁 Cấu Trúc Files

```
monitor/prometheus/
├── prometheus.yml      # Cấu hình chính của Prometheus
├── alert_rules.yml     # Alert rules cho các metrics
└── README.md           # File hướng dẫn này
```

## 🏗️ Cấu Hình Prometheus

### Global Settings
```yaml
global:
  scrape_interval: 15s      # Tần suất thu thập metrics
  scrape_timeout: 10s       # Timeout cho mỗi scrape
  evaluation_interval: 15s  # Tần suất evaluate alert rules
```

### Alert Rules Integration
- **Auto-load**: `alert_rules.yml` được load tự động
- **AlertManager**: Tự động gửi alerts đến AlertManager
- **Evaluation**: Alerts được evaluate mỗi 15 giây

## 📊 Scrape Jobs Configuration

### 1. **NexusML Backend** (`nexusml-backend`)
**Thu thập metrics từ ML API:**
- **Endpoint**: `http://backend:8000/metrics`
- **Frequency**: 5 seconds (high-frequency cho real-time monitoring)
- **Timeout**: 4 seconds
- **Metrics Collected**: Tất cả model inference metrics

### 2. **Model Predictions** (`model_predictions`)
**Chuyên thu thập model prediction metrics:**
- **Endpoint**: `http://backend:8000/api/v1/metrics`
- **Frequency**: 15 seconds
- **Metrics Filtered**: 
  - Model metrics (`model_.*`)
  - API metrics (`api_.*`)
  - Infrastructure metrics (`infra_.*`)
  - System metrics (`system_.*`)
  - Batch processing metrics (`batch_.*`)

**Relabeling Rules:**
- Thêm environment labels: `environment=production`
- Service identification: `service=nexusml`
- Team assignment: `team=mlops`
- Instance tracking: `instance=backend-1`

### 3. **Performance Metrics** (`performance_metrics`)
**Focus vào latency và performance metrics:**
- **Endpoint**: `http://backend:8000/metrics`
- **Frequency**: 5 seconds (ultra high-frequency)
- **Timeout**: 3 seconds
- **Metrics Focused**:
  - All latency metrics (`*_latency_seconds`)
  - Duration metrics (`*_duration_seconds`)
  - Processing metrics (`*_processing_seconds`)
  - API request latency

**Custom Labels:**
- `latency_type=prediction` cho inference latency
- `latency_type=processing` cho batch processing latency

### 4. **Business Metrics** (`business_metrics`)
**Thu thập business-critical metrics:**
- **Endpoint**: `http://backend:8000/api/v1/business/metrics`
- **Frequency**: 30 seconds (business logic frequency)
- **Timeout**: 10 seconds
- **Model-Specific Labels**:
  - `model_type=sentiment`
  - `model_type=email`
  - `model_type=recsys`
  - `model_type=topic`
  - `model_type=trend`

### 5. **Infrastructure Services**

#### Node Exporter
- **Target**: `node_exporter:9100`
- **Frequency**: 15 seconds
- **Metrics**: CPU, Memory, Disk, Network

#### Redis Exporter
- **Target**: `redis_exporter:9121`
- **Frequency**: 15 seconds
- **Metrics**: Redis performance và health

#### MinIO Metrics
- **Target**: `minio:9000`
- **Path**: `/minio/v2/metrics/cluster`
- **Frequency**: 30 seconds
- **Metrics**: Storage performance

#### MLflow Tracking
- **Target**: `mlflow:5000`
- **Path**: `/metrics`
- **Frequency**: 30 seconds
- **Metrics**: MLflow server health

#### Airflow
- **Target**: `airflow-statsd-exporter:8125`
- **Frequency**: 15 seconds
- **Metrics**: DAG performance và task execution

## 🚨 Alert Rules

### Critical Alerts Group (`nexusml-critical-alerts`)

#### Model Service Down
```yaml
alert: ModelServiceDown
expr: up{job="nexusml-backend"} == 0
for: 1m
labels:
  severity: critical
```
- **Purpose**: Detect khi ML backend service down
- **Threshold**: 1 phút downtime
- **Action**: Immediate escalation

#### High Model Error Rate
```yaml
alert: HighModelErrorRate
expr: rate(model_prediction_errors_total[5m]) > 0.1
for: 5m
```
- **Purpose**: Phát hiện high error rate trong model predictions
- **Threshold**: > 0.1 errors/sec
- **Action**: Investigate model performance

#### Infrastructure Failures
- **RedisConnectionDown**: Redis connection loss
- **MinIOConnectionDown**: MinIO storage connection loss
- **ModelLoadFailures**: Model loading failures

### Warning Alerts Group (`nexusml-warning-alerts`)

#### Performance Degradation
- **HighModelLatency**: P95 latency > 2 seconds
- **LowModelConfidence**: Average confidence < 0.6
- **SentimentModelSlow**: Throughput < 1/sec

#### Resource Usage
- **HighMemoryUsage**: Memory > 8GB
- **HighCPUUsage**: CPU > 80%
- **HighBatchProcessingFailureRate**: Batch failure rate > 5%

#### Model-Specific Alerts
- **EmailModelSpamDetectionIssues**: Abnormal spam detection patterns
- **RecSysHighColdStartRate**: Cold start rate > 50%

### Infrastructure Alerts Group (`nexusml-infrastructure-alerts`)

#### API Health
- **HighAPIRequestRate**: Request rate > 100/sec (potential DDoS)
- **HighAPIErrorRate**: API error rate > 5%
- **HighModelLoadTime**: Model load time > 30s

### Business Alerts Group (`nexusml-business-alerts`)

#### Business Logic Anomalies
- **SentimentTrendAnomaly**: Unusual sentiment patterns
- **TopicClassificationImbalance**: Topic distribution issues
- **RecSysRatingDistributionAnomaly**: Rating prediction anomalies

## 🎯 Metrics Filtering Strategy

### Keep Patterns
- `model_.*` - All model-specific metrics
- `api_.*` - API performance metrics
- `infra_.*` - Infrastructure metrics
- `system_.*` - System resource metrics
- `batch_.*` - Batch processing metrics

### Drop Patterns
- `debug_.*` - Debug metrics
- `internal_.*` - Internal implementation metrics
- `_internal_.*` - Private/internal metrics

### Custom Labels
- **Environment**: `environment=production`
- **Service**: `service=nexusml`
- **Team**: `team=mlops`
- **Instance**: `instance=backend-1`
- **Model Type**: `model_type=sentiment|email|recsys|topic|trend`
- **Latency Type**: `latency_type=prediction|processing`

## 🚀 Deployment và Usage

### 1. Docker Compose Setup
```bash
# Start Prometheus với custom config
cd monitor
docker-compose up -d prometheus

# Check Prometheus status
docker-compose logs prometheus
```

### 2. Manual Configuration
```bash
# Validate configuration
docker run --rm -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --check-config

# Test rule files
docker run --rm -v $(pwd)/alert_rules.yml:/etc/prometheus/alert_rules.yml \
  prom/prometheus:latest \
  --config.file=/dev/null \
  --check-config \
  --rule.file=/etc/prometheus/alert_rules.yml
```

### 3. Reload Configuration
```bash
# Send SIGHUP to reload (docker)
docker kill -s HUP prometheus

# API reload (if enabled)
curl -X POST http://localhost:9090/-/reload
```

### 4. Access Prometheus UI
- **URL**: http://localhost:9090
- **Status**: Check `http://localhost:9090/targets`
- **Rules**: Check `http://localhost:9090/rules`
- **Config**: Check `http://localhost:9090/config`

## 📈 Query Examples

### Model Performance Queries
```promql
# Overall model prediction rate
rate(model_predictions_total[5m])

# Model latency by type
histogram_quantile(0.95, 
  rate(model_prediction_latency_seconds_bucket[5m])
)

# Model error rate by type
rate(model_prediction_errors_total[5m]) by (model_type)
```

### Infrastructure Health
```promql
# System resource usage
system_cpu_usage_percent
system_memory_usage_bytes / 1024 / 1024 / 1024

# Service availability
up{job="nexusml-backend"}
```

### Business Metrics
```promql
# Sentiment analysis distribution
rate(model_sentiment_prediction_count[5m]) by (sentiment_label)

# Email spam detection rate
rate(model_email_spam_detected_total[5m]) / 
(rate(model_email_spam_detected_total[5m]) + rate(model_email_ham_detected_total[5m]))

# RecSys average rating
rate(model_recsys_predicted_rating_sum[5m]) / 
rate(model_recsys_predicted_rating_count[5m])
```

## 🔧 Customization

### Adding New Scrape Jobs
```yaml
- job_name: "new_service"
  static_configs:
    - targets: ["new-service:8080"]
  scrape_interval: 15s
  metrics_path: "/metrics"
  scrape_timeout: 5s
```

### Modifying Alert Thresholds
Chỉnh sửa trong `alert_rules.yml`:
```yaml
- alert: HighModelLatency
  expr: histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m])) > 3  # Changed từ 2 thành 3
  for: 10m
```

### Adding Custom Labels
Thêm vào metric_relabel_configs:
```yaml
metric_relabel_configs:
  - target_label: custom_label
    replacement: "custom_value"
```

## 📊 Performance Optimization

### 1. Scrape Frequency Tuning
- **High-frequency jobs**: 5s cho critical metrics
- **Standard jobs**: 15s cho normal monitoring
- **Low-frequency jobs**: 30s cho business metrics

### 2. Retention Policy
Thêm vào prometheus.yml:
```yaml
global:
  retention: 30d

storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

### 3. Recording Rules
Tạo file `recording_rules.yml`:
```yaml
groups:
  - name: nexusml_recordings
    interval: 30s
    rules:
      - record: nexusml:model_rate_5m
        expr: rate(model_predictions_total[5m])
      
      - record: nexusml:api_latency_p95_5m
        expr: histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Targets Down
```bash
# Check target status
curl http://localhost:9090/api/v1/targets

# Verify service health
curl http://localhost:8000/metrics
```

#### 2. High Cardinality
```bash
# Check metric cardinality
curl http://localhost:9090/api/v1/label/__name__/values

# Identify high-cardinality metrics
curl http://localhost:9090/api/v1/series | jq '.data[].metric | select(keys | length > 10)'
```

#### 3. Alert Not Firing
```bash
# Check rule evaluation
curl http://localhost:9090/api/v1/rules

# Test alert expression
curl "http://localhost:9090/api/v1/query?query=up{job='nexusml-backend'}"
```

#### 4. Configuration Issues
```bash
# Validate config syntax
docker run --rm -v $(pwd):/prometheus prom/prometheus:latest --config.file=/prometheus/prometheus.yml --check-config

# Check rule syntax
docker run --rm -v $(pwd)/alert_rules.yml:/rules prom/prometheus:latest --check-config --rule.file=/rules
```

### Debug Commands
```bash
# Check Prometheus logs
docker-compose logs prometheus

# Examine specific target
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job == "nexusml-backend")'

# Test metric scraping
curl http://localhost:8000/metrics | head -20

# Check alert manager connectivity
curl http://localhost:9090/api/v1/status/config
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Alerting Rules Guide](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Relabeling Guide](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Best Practices](https://prometheus.io/docs/practices/recording_rules/)

## 🔒 Security Considerations

### 1. Access Control
- Restrict `/metrics` endpoint access
- Use TLS for metric transmission
- Implement authentication cho Prometheus UI

### 2. Data Sanitization
- Avoid sensitive data trong metrics
- Use labels thay vì metric values
- Implement metric name validation

### 3. Network Security
- Use network segmentation
- Implement rate limiting
- Monitor for metric scraping abuse

---

**Created**: 2024-11-22  
**Version**: 1.0.0  
**Last Updated**: 2024-11-22  
**Maintained by**: NexusML Team
