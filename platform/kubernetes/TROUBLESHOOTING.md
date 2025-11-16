# Troubleshooting K8s Deployment - NexusML

## Vấn Đề Chính: Airflow CrashLoopBackOff

### Triệu chứng
```
airflow-scheduler-xxx     0/1     CrashLoopBackOff
airflow-webserver-xxx     0/1     CrashLoopBackOff
airflow-worker-xxx        0/1     CrashLoopBackOff
airflow-triggerer-xxx     0/1     CrashLoopBackOff
airflow-init-xxx          0/1     Pending
```

### Lỗi trong logs
```
psycopg2.OperationalError: connection to server at "postgres-service.nexusml.svc.cluster.local" (10.96.0.50), port 5432 failed: 
FATAL: database "airflow" does not exist
```

### Nguyên nhân
1. **PostgreSQL chỉ tạo 1 database**: Cấu hình cũ chỉ tạo database `image_db` cho MLflow
2. **Airflow cần database riêng**: Airflow yêu cầu database `airflow` để lưu metadata
3. **Init script không chạy**: PVC cũ còn data, init script không chạy lại

### Giải pháp đã triển khai

#### 1. Tạo ConfigMap với Init Script
**File**: `platform/kubernetes/postgres/postgres-init-configmap.yaml`

Script tự động tạo 2 databases khi PostgreSQL khởi động lần đầu:
- `image_db` - cho MLflow
- `airflow` - cho Airflow

#### 2. Cập nhật PostgreSQL StatefulSet
**File**: `platform/kubernetes/postgres/postgres-statefulset.yaml`

Thay đổi:
- Mount init script vào `/docker-entrypoint-initdb.d/`
- Đổi `POSTGRES_DB` từ `image_db` sang `postgres` (default DB)
- Sửa mountPath: `/var/lib/postgresql/data` (đúng theo PostgreSQL standard)

---

## Các Vấn Đề Khác Đã Phát Hiện

### 1. PVC Mounting
**Vấn đề**: Airflow pods không thể start vì "unbound immediate PersistentVolumeClaims"

**Kiểm tra**:
```bash
kubectl get pvc -n nexusml
```

**Giải pháp**: Đảm bảo các PVC được tạo trước khi deploy pods:
```bash
kubectl apply -f platform/kubernetes/airflow/airflow-pvc.yaml
```

### 2. Image Pull
**Lưu ý**: Các custom images phải được build và load vào Minikube:
```bash
# Build images
docker build -t nbngoc1303/airflow-image:latest platform/airflow/
docker build -t nbngoc1303/postgres-custom:latest platform/psql_img/
docker build -t nbngoc1303/mlflow-image:latest platform/mlflow/

# Load vào Minikube
minikube image load nbngoc1303/airflow-image:latest
minikube image load nbngoc1303/postgres-custom:latest
minikube image load nbngoc1303/mlflow-image:latest
```

### 3. Airflow Init Job
**Vấn đề**: Init job cần `postgresql-client` để chạy `psql` commands

**Kiểm tra Dockerfile**: `platform/airflow/Dockerfile`
```dockerfile
FROM apache/airflow:2.10.3

USER root
RUN apt-get update && \
    apt-get install -y git postgresql-client && \  # ← Cần package này
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**Status**: ✅ Đã có `postgresql-client` trong Dockerfile

### 4. Redis Dependency
**Vấn đề**: Airflow CeleryExecutor cần Redis làm message broker

**Kiểm tra**:
```bash
kubectl get pods -n nexusml -l app=redis
```

**Nếu Redis chưa chạy**:
```bash
kubectl apply -f platform/kubernetes/redis/redis-service.yaml
kubectl apply -f platform/kubernetes/redis/redis-statefulset.yaml
```

### 5. MinIO cho MLflow Artifacts
**Lưu ý**: MLflow cần MinIO để lưu artifacts (models, files)

**Kiểm tra**:
```bash
kubectl get pods -n nexusml -l app=minio
```

**Deploy nếu chưa có**:
```bash
kubectl apply -f platform/kubernetes/minio/minio-secret.yaml
kubectl apply -f platform/kubernetes/minio/minio-pvc.yaml
kubectl apply -f platform/kubernetes/minio/minio-service.yaml
kubectl apply -f platform/kubernetes/minio/minio-deployment.yaml
```

---

## Thứ Tự Deploy Đúng

### 1. Prerequisites
```bash
# Tạo namespace
kubectl create namespace nexusml

# Verify Minikube
minikube status
```

### 2. Storage (PVC)
```bash
# Airflow PVCs
kubectl apply -f platform/kubernetes/airflow/airflow-pvc.yaml

# MLflow PVC
kubectl apply -f platform/kubernetes/mlflow/mlflow-pvc.yaml

# MinIO PVC
kubectl apply -f platform/kubernetes/minio/minio-pvc.yaml
```

### 3. Databases & Storage Services
```bash
# PostgreSQL (với init script mới)
kubectl apply -f platform/kubernetes/postgres/postgres-init-configmap.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-secret.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-service.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-statefulset.yaml

# Redis
kubectl apply -f platform/kubernetes/redis/redis-service.yaml
kubectl apply -f platform/kubernetes/redis/redis-statefulset.yaml

# MinIO
kubectl apply -f platform/kubernetes/minio/minio-secret.yaml
kubectl apply -f platform/kubernetes/minio/minio-service.yaml
kubectl apply -f platform/kubernetes/minio/minio-deployment.yaml
```

### 4. Đợi Databases Sẵn Sàng
```bash
# PostgreSQL
kubectl wait --for=condition=ready pod -l app=postgres -n nexusml --timeout=120s

# Redis
kubectl wait --for=condition=ready pod -l app=redis -n nexusml --timeout=60s

# MinIO
kubectl wait --for=condition=ready pod -l app=minio -n nexusml --timeout=60s
```

### 5. MLflow
```bash
kubectl apply -f platform/kubernetes/mlflow/mlflow-deployment.yaml
kubectl apply -f platform/kubernetes/mlflow/mlflow-service.yaml
```

### 6. Airflow
```bash
# Secret
kubectl apply -f platform/kubernetes/airflow/airflow-secret.yaml

# Init Job (tạo schema)
kubectl apply -f platform/kubernetes/airflow/airflow-init-job.yaml
kubectl wait --for=condition=complete job/airflow-init -n nexusml --timeout=300s

# Components
kubectl apply -f platform/kubernetes/airflow/airflow-scheduler-deployment.yaml
kubectl apply -f platform/kubernetes/airflow/airflow-webserver-deployment.yaml
kubectl apply -f platform/kubernetes/airflow/airflow-webserver-service.yaml
kubectl apply -f platform/kubernetes/airflow/airflow-worker-deployment.yaml
kubectl apply -f platform/kubernetes/airflow/airflow-triggerer-deployment.yaml
```

### 7. Monitoring (Optional)
```bash
kubectl apply -f platform/kubernetes/monitoring/prometheus-configmap.yaml
kubectl apply -f platform/kubernetes/monitoring/prometheus-statefulset.yaml
kubectl apply -f platform/kubernetes/monitoring/prometheus-service.yaml
kubectl apply -f platform/kubernetes/monitoring/grafana-pvc.yaml
kubectl apply -f platform/kubernetes/monitoring/grafana-deployment.yaml
kubectl apply -f platform/kubernetes/monitoring/grafana-service.yaml
```

---

## Debug Commands

### Kiểm tra Pods
```bash
# Tất cả pods trong namespace
kubectl get pods -n nexusml

# Watch realtime
kubectl get pods -n nexusml -w

# Pods theo label
kubectl get pods -n nexusml -l app=airflow-scheduler
```

### Xem Logs
```bash
# Logs hiện tại
kubectl logs <pod-name> -n nexusml

# Logs của container trước đó (khi restart)
kubectl logs <pod-name> -n nexusml --previous

# Follow logs
kubectl logs -f <pod-name> -n nexusml

# Logs của deployment
kubectl logs -f deployment/airflow-scheduler -n nexusml
```

### Describe Resources
```bash
# Pod details & events
kubectl describe pod <pod-name> -n nexusml

# PVC status
kubectl describe pvc <pvc-name> -n nexusml

# Service endpoints
kubectl describe service <service-name> -n nexusml
```

### Exec vào Pod
```bash
# PostgreSQL
kubectl exec -it postgres-0 -n nexusml -- psql -U postgres

# List databases
kubectl exec -it postgres-0 -n nexusml -- psql -U postgres -c "\l"

# Check specific database
kubectl exec -it postgres-0 -n nexusml -- psql -U postgres -d airflow -c "\dt"
```

### Test Connectivity
```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -n nexusml -- nslookup postgres-service.nexusml.svc.cluster.local

# Test PostgreSQL connection
kubectl run -it --rm debug --image=postgres:13 --restart=Never -n nexusml -- \
  psql postgresql://postgres:aivn2025@postgres-service.nexusml.svc.cluster.local:5432/airflow
```

### Clean Up
```bash
# Xóa deployment
kubectl delete deployment <deployment-name> -n nexusml

# Xóa statefulset và data
kubectl delete statefulset <sts-name> -n nexusml
kubectl delete pvc <pvc-name> -n nexusml

# Xóa toàn bộ namespace
kubectl delete namespace nexusml
```

---

## Port Forwarding để Test

### Airflow Webserver
```bash
kubectl port-forward -n nexusml service/airflow-webserver-service 8080:8080
# Truy cập: http://localhost:8080
# User: admin / Pass: admin
```

### MLflow
```bash
kubectl port-forward -n nexusml service/mlflow-service 5000:5000
# Truy cập: http://localhost:5000
```

### MinIO Console
```bash
kubectl port-forward -n nexusml service/minio-service 9001:9001
# Truy cập: http://localhost:9001
```

### Grafana
```bash
kubectl port-forward -n nexusml service/grafana-service 3000:3000
# Truy cập: http://localhost:3000
```

---

## Quick Fix Script

Đã tạo script tự động sửa lỗi database:
```bash
bash platform/kubernetes/fix-airflow-db.sh
```

Script sẽ:
1. Xóa PostgreSQL cũ
2. Apply init script
3. Deploy lại PostgreSQL với 2 databases
4. Restart Airflow components
5. Verify kết quả

**Lưu ý**: Script sẽ **XÓA DATA** trong PostgreSQL. Chỉ dùng trong dev/test!

---

## Checklist Trước Khi Deploy

- [ ] Minikube đã start: `minikube status`
- [ ] Namespace đã tạo: `kubectl get namespace nexusml`
- [ ] Images đã build và load vào Minikube
- [ ] PVCs đã được tạo trước
- [ ] Secrets đã được apply
- [ ] PostgreSQL init script đã apply
- [ ] PostgreSQL running và có cả 2 databases (airflow, image_db)
- [ ] Redis running trước khi deploy Airflow
- [ ] MinIO running trước khi deploy MLflow

---

## Tham Khảo

- **Hướng dẫn chi tiết**: `platform/kubernetes/FIX_AIRFLOW_DATABASE.md`
- **Script tự động**: `platform/kubernetes/fix-airflow-db.sh`
- **Deploy script**: `platform/deploy_minikube.sh`
