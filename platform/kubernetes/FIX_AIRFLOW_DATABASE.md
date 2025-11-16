# Hướng Dẫn Sửa Lỗi Database Airflow

## Vấn Đề
Airflow pods bị CrashLoopBackOff với lỗi:
```
FATAL: database "airflow" does not exist
```

## Nguyên Nhân
PostgreSQL chỉ được cấu hình tạo database `image_db` (cho MLflow), không tạo database `airflow` mà Airflow cần.

## Giải Pháp Đã Thực Hiện

### 1. Tạo Init Script cho PostgreSQL
File mới: `platform/kubernetes/postgres/postgres-init-configmap.yaml`
- ConfigMap chứa script tự động tạo cả 2 databases: `image_db` và `airflow`

### 2. Cập Nhật PostgreSQL StatefulSet
File đã sửa: `platform/kubernetes/postgres/postgres-statefulset.yaml`
- Mount init script vào `/docker-entrypoint-initdb.d/`
- Thay đổi POSTGRES_DB từ `image_db` sang `postgres` (default DB)
- Sửa mountPath từ `/var/lib/postgresql` sang `/var/lib/postgresql/data`

## Các Bước Deploy Lại

### Bước 1: Xóa PostgreSQL cũ
```bash
# Xóa StatefulSet và Pod
kubectl delete statefulset postgres -n nexusml

# Xóa PVC (QUAN TRỌNG: Xóa data cũ để chạy lại init script)
kubectl delete pvc postgres-data-postgres-0 -n nexusml
```

### Bước 2: Apply ConfigMap mới
```bash
kubectl apply -f platform/kubernetes/postgres/postgres-init-configmap.yaml
```

### Bước 3: Deploy lại PostgreSQL
```bash
# Apply Secret (nếu chưa có)
kubectl apply -f platform/kubernetes/postgres/postgres-secret.yaml

# Apply Service
kubectl apply -f platform/kubernetes/postgres/postgres-service.yaml

# Apply StatefulSet mới
kubectl apply -f platform/kubernetes/postgres/postgres-statefulset.yaml
```

### Bước 4: Đợi PostgreSQL sẵn sàng
```bash
kubectl wait --for=condition=ready pod -l app=postgres -n nexusml --timeout=120s
```

### Bước 5: Verify databases đã được tạo
```bash
# Exec vào PostgreSQL pod
kubectl exec -it postgres-0 -n nexusml -- psql -U postgres -c "\l"

# Bạn sẽ thấy cả 2 databases:
# - image_db
# - airflow
```

### Bước 6: Xóa Airflow pods để restart
```bash
# Xóa init job nếu còn pending
kubectl delete job airflow-init -n nexusml 2>/dev/null || true

# Restart các Airflow deployments
kubectl rollout restart deployment airflow-scheduler -n nexusml
kubectl rollout restart deployment airflow-webserver -n nexusml
kubectl rollout restart deployment airflow-worker -n nexusml
kubectl rollout restart deployment airflow-triggerer -n nexusml
```

### Bước 7: Kiểm tra status
```bash
# Xem pods
kubectl get pods -n nexusml -w

# Xem logs của scheduler
kubectl logs -f deployment/airflow-scheduler -n nexusml

# Xem logs của webserver
kubectl logs -f deployment/airflow-webserver -n nexusml
```

## Script Deploy Hoàn Chỉnh

```bash
#!/bin/bash

echo "=== Fix Airflow Database Issue ==="

# Namespace
NS="nexusml"

echo "Step 1: Cleaning up old PostgreSQL..."
kubectl delete statefulset postgres -n $NS --ignore-not-found=true
kubectl delete pvc postgres-data-postgres-0 -n $NS --ignore-not-found=true

echo "Step 2: Applying ConfigMap..."
kubectl apply -f platform/kubernetes/postgres/postgres-init-configmap.yaml

echo "Step 3: Deploying PostgreSQL..."
kubectl apply -f platform/kubernetes/postgres/postgres-secret.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-service.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-statefulset.yaml

echo "Step 4: Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NS --timeout=180s

echo "Step 5: Verifying databases..."
sleep 5
kubectl exec -it postgres-0 -n $NS -- psql -U postgres -c "\l"

echo "Step 6: Restarting Airflow components..."
kubectl delete job airflow-init -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-scheduler -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-webserver -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-worker -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-triggerer -n $NS --ignore-not-found=true

echo "Step 7: Monitoring pods..."
kubectl get pods -n $NS
```

## Lưu Ý Quan Trọng

1. **Xóa PVC**: Phải xóa PVC để PostgreSQL chạy lại init script. Nếu không xóa, script sẽ không chạy.

2. **Data Loss**: Xóa PVC sẽ mất hết data trong PostgreSQL. Chỉ làm điều này trong môi trường dev/test.

3. **Production**: Trong production, nên:
   - Backup data trước
   - Hoặc exec vào pod và tạo database manually:
     ```bash
     kubectl exec -it postgres-0 -n nexusml -- psql -U postgres -c "CREATE DATABASE airflow;"
     ```

4. **Airflow Init Job**: Sau khi PostgreSQL có database `airflow`, init job sẽ chạy thành công và tạo schema cho Airflow.

## Kiểm Tra Lỗi

### Nếu pods vẫn bị CrashLoopBackOff:
```bash
# Xem logs chi tiết
kubectl logs <pod-name> -n nexusml --previous

# Xem events
kubectl describe pod <pod-name> -n nexusml
```

### Nếu database chưa được tạo:
```bash
# Kiểm tra init script có chạy không
kubectl logs postgres-0 -n nexusml | grep -i "database"

# Kiểm tra ConfigMap
kubectl describe configmap postgres-init-script -n nexusml
```

### Nếu kết nối database bị lỗi:
```bash
# Test kết nối từ một pod khác
kubectl run -it --rm debug --image=postgres:13 --restart=Never -n nexusml -- \
  psql postgresql://postgres:aivn2025@postgres-service.nexusml.svc.cluster.local:5432/airflow
```

## Kết Quả Mong Đợi

Sau khi hoàn thành, bạn sẽ thấy:
- PostgreSQL pod: Running
- Airflow scheduler: Running
- Airflow webserver: Running
- Airflow worker: Running
- Airflow triggerer: Running
- MLflow: Running (vì vẫn có database `image_db`)

Không còn lỗi "database does not exist"!
