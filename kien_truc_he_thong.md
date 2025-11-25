# Kiến trúc hệ thống NexusML

Dưới đây là các sơ đồ kiến trúc hệ thống của dự án NexusML, ngoại trừ phần giám sát (monitoring).

## 1. Sơ đồ tổng quan hệ thống (High-Level System Architecture)

```
┌─────────┐    ┌───────────────┐
│ Người dùng │──▶│  Frontend App │
└─────────┘    └───────────────┘
                     │
                     ▼
             ┌─────────────────┐
             │   Backend API   │
             └─────────────────┘
                     │
                     ▼
             ┌─────────────────┐
             │ Tải/Sử dụng mô hình, Lưu trữ/Truy xuất dữ liệu │
             └─────────────────┘
                     │
                     ▼
┌─────────────────┐    ┌─────────────────────┐
│ MLflow Tracking │    │  MinIO Object Storage │
│      Server     │    │                     │
└─────────────────┘    └─────────────────────┘
         │               ▲
         └───────────────┘
          Lưu trữ metadata/Mô hình

┌─────────┐    ┌─────────────────┐
│ Dữ liệu thô │──▶│  Apache Airflow │
└─────────┘    └─────────────────┘
                     │
                     ▼
             ┌─────────────────┐
             │ Log Experiment/Mô hình, Lưu trữ dữ liệu/Artifacts │
             └─────────────────┘
                     │
                     ▼
┌─────────────────┐    ┌─────────────────────┐
│ MLflow Tracking │    │  MinIO Object Storage │
│      Server     │    │                     │
└─────────────────┘    └─────────────────────┘
```

## 2. Sơ đồ luồng dữ liệu và mô hình (Data & Model Flow Diagram)

```
┌───────────────────┐    ┌─────────────┐    ┌─────────────────────┐
│    Dữ liệu thô    │──▶│  Airflow DAGs │──▶│  Tiền xử lý dữ liệu │
└───────────────────┘    └─────────────┘    └─────────────────────┘
                                                │
                                                ▼
                                            ┌─────────────────────┐
                                            │ Dữ liệu đã xử lý - │
                                            │        MinIO        │
                                            └─────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────┐    ┌───────────────────────────┐    ┌─────────────────┐
│ Airflow DAGs - Huấn luyện mô hình │──▶│ MLflow - Theo dõi Experiment │──▶│ MLflow Model    │
└───────────────────────┘    └───────────────────────────┘    │    Registry     │
                                                               └─────────────────┘
                                                                       │
                                                                       ▼
                                                               ┌──────────────────┐
                                                               │ Mô hình/Artifacts│
                                                               │      - MinIO     │
                                                               └──────────────────┘

┌─────────────────┐    ┌───────────────────────────┐
│   Frontend App  │──▶│ Backend API - Inference Request │
└─────────────────┘    └───────────────────────────┘
                           │        ▲        │
                           │ Tải mô hình │  │ Load Artifacts │
                           ▼        │        ▼
             ┌─────────────────┐    ┌──────────────────┐
             │ MLflow Model    │    │ Mô hình đã đăng ký │
             │    Registry     │    │      - MinIO     │
             └─────────────────┘    └──────────────────┘
                           │        │
                           ▼        ▼
                      ┌──────────────────┐
                      │  Model Inference │
                      └──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Frontend App  │
                       └─────────────────┘
```

## 3. Sơ đồ triển khai (Deployment Diagram)

```
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│ Frontend App (Streamlit/Flask) │──▶│     Backend API (FastAPI)     │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ HTTP                                 │ HTTP/S3
         │                                      │
         ▼                                      ▼
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│    MLflow Tracking Server      │──▶│     MinIO Object Storage      │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ S3                                   ▲ S3
         │                                      │
         ▼                                      │
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│     Airflow Webserver          │──▶│     Airflow Scheduler         │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ HTTP                                 │ Redis
         │                                      │
         ▼                                      ▼
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│     Airflow Worker (Celery)    │──▶│            Redis              │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ Redis                                ▲ PostgreSQL
         │                                      │
         ▼                                      ▼
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│ PostgreSQL (Airflow Metadata)  │──▶│   PostgreSQL (Airflow Metadata) │
└────────────────────────────────┘    └─────────────────────────────────┘

┌────────────────────────────────┐    ┌─────────────────────────────────┐
│            Prometheus          │──▶│             Grafana           │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ Scrape Metrics                       ▲ Visualize Metrics
         │                                      │
         ▼                                      │
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│     Alertmanager               │──▶│               Loki            │
└────────────────────────────────┘    └─────────────────────────────────┘
         │ Alerts                               ▲ Log Aggregation
         │                                      │
         ▼                                      │
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│     Backend API (FastAPI)      │──▶│       Various Services        │
└────────────────────────────────┘    └─────────────────────────────────┘

Frontend App: http://localhost:7860
Backend API: http://localhost:8000
MLflow Server: http://localhost:5000
MinIO Storage: http://localhost:9000/9001
Airflow Webserver: http://localhost:8080
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
Alertmanager: http://localhost:9093
Loki: http://localhost:3100
