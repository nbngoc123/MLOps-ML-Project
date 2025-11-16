#!/bin/bash

# --- 1. Cấu hình & Lược bỏ ---
# Các thành phần đã lược bỏ: Kafka, Zookeeper, Jenkins, Loki.
# Thay thế: PostgreSQL sẽ được triển khai cục bộ từ manifests.

# Thiết lập tên Images (Đã được Build trước đó)
AIRFLOW_IMAGE="airflow-image:latest"
MLFLOW_IMAGE="mlflow-image:latest"
POSTGRES_IMAGE="postgres-custom:latest" # Hoặc postgres:latest nếu không dùng Dockerfile của bạn
BACKEND_IMAGE="backend-image:latest"

# Thư mục chứa manifests
K8S_MANIFESTS="./kubernetes"

# --- 2. Khởi động Minikube & Load Images ---

# echo "--- Khởi động Minikube và Load Images ---"
# minikube start --driver=docker --memory=6g --cpus=2
# if [ $? -ne 0 ]; then
#     echo "Lỗi: Không thể khởi động Minikube. Vui lòng kiểm tra Docker/WSL2."
#     exit 1
# fi

# Load các images đã build vào Minikube
# minikube image load ${AIRFLOW_IMAGE}
# minikube image load ${MLFLOW_IMAGE}
# minikube image load ${POSTGRES_IMAGE}
# minikube image load ${BACKEND_IMAGE}


# --- 3. Triển khai PostgreSQL (Database) ---

echo "--- 3. Triển khai PostgreSQL ---"
# Thứ tự: Secret -> Service -> StatefulSet
kubectl apply -f ${K8S_MANIFESTS}/postgres/postgres-secret.yaml
kubectl apply -f ${K8S_MANIFESTS}/postgres/postgres-service.yaml
kubectl apply -f ${K8S_MANIFESTS}/postgres/postgres-statefulset.yaml

# Đợi PostgreSQL sẵn sàng (Quan trọng cho Airflow/MLflow)
echo "Đang đợi PostgreSQL khởi động..."
kubectl wait --for=condition=ready pod -l app=postgres --timeout=120s


# --- 4. Triển khai Redis (Cần cho Airflow) ---

echo "--- 4. Triển khai Redis ---"
kubectl apply -f ${K8S_MANIFESTS}/redis/redis-service.yaml
kubectl apply -f ${K8S_MANIFESTS}/redis/redis-statefulset.yaml
kubectl wait --for=condition=ready pod -l app=redis --timeout=60s


# --- 5. Triển khai MLflow ---

echo "--- 5. Triển khai MLflow ---"
# Cần PVC, Deployment và Service
kubectl apply -f ${K8S_MANIFESTS}/mlflow/mlflow-pvc.yaml
kubectl apply -f ${K8S_MANIFESTS}/mlflow/mlflow-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/mlflow/mlflow-service.yaml


# --- 6. Triển khai Airflow ---

echo "--- 6. Triển khai Airflow ---"
# Cần Secret, PVC, Webserver, Scheduler
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-secret.yaml
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-pvc.yaml

# Khởi tạo DB (Chạy lần đầu)
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-init-job.yaml
# Đợi Init Job hoàn thành
kubectl wait --for=condition=complete job/airflow-init-job --timeout=300s
kubectl delete job airflow-init-job # Xóa Job sau khi hoàn thành

# Triển khai các thành phần chính
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-scheduler-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-webserver-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/airflow/airflow-webserver-service.yaml


# --- 7. Triển khai Monitoring (Prometheus & Grafana) ---

echo "--- 7. Triển khai Monitoring ---"
# Prometheus
kubectl apply -f ${K8S_MANIFESTS}/monitoring/prometheus-configmap.yaml
kubectl apply -f ${K8S_MANIFESTS}/monitoring/prometheus-statefulset.yaml
kubectl apply -f ${K8S_MANIFESTS}/monitoring/prometheus-service.yaml

# Grafana
kubectl apply -f ${K8S_MANIFESTS}/monitoring/grafana-pvc.yaml
kubectl apply -f ${K8S_MANIFESTS}/monitoring/grafana-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/monitoring/grafana-service.yaml


# --- 8. Triển khai Serving Pipeline ---

echo "--- 8. Triển khai Serving Pipeline ---"
# Backend
kubectl apply -f ${K8S_MANIFESTS}/serving_pipeline/backend-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/serving_pipeline/backend-service.yaml

# Frontend
kubectl apply -f ${K8S_MANIFESTS}/serving_pipeline/frontend-deployment.yaml
kubectl apply -f ${K8S_MANIFESTS}/serving_pipeline/frontend-service.yaml


echo "--- Triển khai Hoàn tất. Cluster Ready! ---"
echo "Chạy 'kubectl get pods' để kiểm tra trạng thái."