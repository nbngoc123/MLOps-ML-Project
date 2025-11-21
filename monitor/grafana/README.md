# Grafana Dashboards for NexusML Monitoring

## 📋 Tổng Quan

Thư mục này chứa các Grafana dashboards để monitoring hệ thống ML pipeline của NexusML.

## 📁 Cấu Trúc Files

```
monitor/grafana/
├── dashboard.json                    # Dashboard tổng quan hệ thống ML
├── model-performance-dashboard.json  # Dashboard chuyên sâu model performance
├── dashboards.yml                    # Cấu hình dashboard providers
├── datasources.yml                   # Cấu hình data sources (Prometheus)
└── README.md                         # File hướng dẫn này
```

## 🎯 Dashboards Available

### 1. **NexusML ML Pipeline Monitoring** (`dashboard.json`)
**Dashboard tổng quan cho toàn bộ hệ thống ML pipeline:**

- **Model Performance Overview**: Tổng quan về throughput predictions/giây
- **Model Latency (P95)**: Độ trễ 95th percentile của các model
- **Sentiment Analysis Predictions**: Tốc độ predictions theo từng sentiment label
- **Email Classification Rate**: Tốc độ phát hiện spam vs ham emails
- **Recommendation System Metrics**: Average rating và cold start events
- **Topic Classification Distribution**: Phân phối các topics
- **API Request Rate**: Tốc độ request theo endpoint
- **Infrastructure Health**: Status kết nối Redis và MinIO
- **System Resources**: CPU và Memory usage
- **Model Confidence Distribution**: Phân phối confidence scores
- **Batch Processing Performance**: Performance của batch processing
- **Error Rate by Model**: Tỷ lệ lỗi theo từng model

### 2. **NexusML Model Performance Deep Dive** (`model-performance-dashboard.json`)
**Dashboard chuyên sâu cho từng model:**

#### Sentiment Model Section:
- **Sentiment Predictions/sec**: Throughput của sentiment model
- **Sentiment Latency (P95)**: Độ trễ P95 của sentiment model
- **Sentiment Confidence Avg**: Average confidence score
- **Sentiment Prediction Rate by Label**: Tốc độ predictions theo từng label
- **Sentiment Confidence Distribution**: Phân phối confidence scores

#### Email Model Section:
- **Email Classifications/sec**: Throughput của email classification
- **Spam Detection Rate**: Tỷ lệ phát hiện spam
- **Email Confidence Avg**: Average confidence score
- **Email Classification Rate**: Rate spam vs ham detection

#### RecSys Model Section:
- **RecSys Predictions/sec**: Throughput của recommendation system
- **RecSys Cold Start Rate**: Tỷ lệ cold start recommendations
- **RecSys Avg Rating**: Average predicted rating
- **RecSys Rating Distribution**: Phân phối các rating predictions
- **Cold Start Events**: Số lượng cold start events theo thời gian

#### Model Loading Section:
- **Model Load Time by Type**: Thời gian load model theo từng loại
- **Model Load Success Rate**: Tỷ lệ load model thành công

## 🚀 Setup và Sử Dụng

### 1. Docker Compose Setup
```bash
# Start toàn bộ monitoring stack
cd monitor
docker-compose up -d

# Kiểm tra các containers đang chạy
docker-compose ps
```

### 2. Truy cập Grafana
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin
- **Thay đổi password sau lần đầu đăng nhập**

### 3. Dashboards Auto-Load
Các dashboards sẽ tự động được load khi Grafana khởi động:
1. `NexusML ML Pipeline Monitoring` - xuất hiện trong folder gốc
2. `NexusML Model Performance Deep Dive` - xuất hiện trong folder "Model Performance"

### 4. Manual Dashboard Import
Nếu cần import thủ công:
1. Vào Grafana UI → Dashboards → Import
2. Upload file `dashboard.json` hoặc `model-performance-dashboard.json`
3. Chọn Prometheus data source
4. Click Import

## 📊 Metrics Được Sử Dụng

### System Metrics
- `system_cpu_usage_percent`
- `system_memory_usage_bytes`
- `infra_redis_connected`
- `infra_minio_connected`

### API Metrics
- `api_request_count_total`
- `api_request_latency_seconds`

### Model Metrics
- `model_sentiment_prediction_count`
- `model_sentiment_confidence_score`
- `model_email_spam_detected_total`
- `model_email_ham_detected_total`
- `model_recsys_predicted_rating`
- `model_recsys_cold_start_total`
- `model_topic_prediction_count`
- `model_load_duration_seconds`
- `model_load_success_total`
- `batch_processing_count_total`

## 🎨 Dashboard Features

### Visualizations Included
- **Stat Panels**: Real-time metrics với color-coded thresholds
- **Time Series**: Trend lines cho performance metrics
- **Pie Charts**: Distribution visualizations
- **Histograms**: Confidence score distributions
- **Row Groups**: Tổ chức theo model types

### Color Coding
- **Green**: Normal/Good performance
- **Yellow**: Warning level
- **Red**: Critical/Problem level

### Refresh Settings
- **Auto-refresh**: 30 seconds
- **Time range**: 1h (overview) hoặc 2h (performance dashboard)

## 🔧 Customization

### Thêm Metrics
Để thêm metrics mới vào dashboard:
1. Edit dashboard trong Grafana UI
2. Thêm panel mới với Prometheus query
3. Configure visualization và thresholds
4. Save changes

### Thay đổi Thresholds
Chỉnh sửa thresholds trong panel configuration:
```json
"thresholds": {
  "steps": [
    {"color": "green", "value": 0},
    {"color": "yellow", "value": 10},
    {"color": "red", "value": 50}
  ]
}
```

### Tạo Dashboard Mới
1. Copy existing dashboard JSON
2. Modify panels và metrics
3. Update title và tags
4. Import vào Grafana

## 🚨 Alerting Integration

Các dashboards này được thiết kế để làm việc với AlertManager:
- Critical metrics có red thresholds
- Warning metrics có yellow thresholds
- Sử dụng same metrics cho alerting rules

## 📝 Best Practices

### Panel Organization
- Group related metrics vào cùng row
- Sử dụng logical flow từ top-left đến bottom-right
- Keep stat panels cho key metrics ở top

### Performance
- Tránh quá nhiều panels (max 20 per dashboard)
- Sử dụng appropriate time ranges
- Optimize Prometheus queries cho performance

### Maintenance
- Regular review dashboard effectiveness
- Update thresholds based on production data
- Archive unused dashboards

## 🐛 Troubleshooting

### Dashboard Not Loading
```bash
# Check Grafana logs
docker-compose logs grafana

# Check Prometheus connectivity
curl http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up
```

### No Data Showing
1. Verify Prometheus is scraping metrics
2. Check metrics endpoint: http://localhost:8000/metrics
3. Verify time range trong dashboard
4. Check data source configuration

### Poor Performance
1. Reduce dashboard refresh rate
2. Simplify complex queries
3. Use recording rules trong Prometheus
4. Consider data retention policies

## 📚 Additional Resources

- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/dashboard-best-practices/)

---

**Created**: 2024-11-22  
**Version**: 1.0.0  
**Maintained by**: NexusML Team
