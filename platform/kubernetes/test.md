Hiểu rồi! Bạn muốn **quản lý riêng lẻ từng ứng dụng** trong namespace `nexusml` (ví dụ chỉ Airflow, chỉ MLflow hoặc chỉ Postgres), mà vẫn áp dụng hoặc xóa nhanh các resource liên quan.

Dưới đây là cách làm:

---

## 1️⃣ Apply hoặc delete riêng một ứng dụng

### a) Airflow

```bash
# Apply tất cả resource Airflow
kubectl apply -f platform/kubernetes/airflow/ -n nexusml

# Xóa tất cả resource Airflow
kubectl delete -f platform/kubernetes/airflow/ -n nexusml
```

### b) MLflow

```bash
# Apply MLflow
kubectl apply -f platform/kubernetes/mlflow/ -n nexusml

# Delete MLflow
kubectl delete -f platform/kubernetes/mlflow/ -n nexusml
```

### c) PostgreSQL

```bash
# Apply PostgreSQL
kubectl apply -f platform/kubernetes/postgres/ -n nexusml

# Delete PostgreSQL
kubectl delete -f platform/kubernetes/postgres/ -n nexusml
```

---

## 2️⃣ Kiểm tra trạng thái pods của ứng dụng đó

```bash
# Chỉ xem pod Airflow
kubectl get pods -n nexusml -l app=airflow

# Chỉ xem pod MLflow
kubectl get pods -n nexusml -l app=mlflow

# Chỉ xem pod PostgreSQL
kubectl get pods -n nexusml -l app=postgres
```

> Yêu cầu mỗi Deployment/StatefulSet phải có **label `app=<tên>`** để filter.

---

## 3️⃣ Lợi ích

* Chỉ deploy hoặc delete **ứng dụng cần thiết** mà không ảnh hưởng đến các service khác.
* Dễ quản lý, test từng phần riêng lẻ.
* Có thể kết hợp với `kustomize` nếu muốn deploy nhiều resource liên quan một cách logic.

---

Nếu bạn muốn, mình có thể viết **ví dụ kustomization + folder structure** cho Airflow, MLflow, PostgreSQL, để chỉ cần **1 lệnh apply/delete riêng từng app**, mà vẫn giữ cấu trúc dự án gọn gàng.

Bạn có muốn mình làm không?
