#!/bin/bash

echo "=== Fix Airflow Database Issue ==="
echo "Lỗi: FATAL: database 'airflow' does not exist"
echo ""

# Namespace
NS="nexusml"

# Hỏi người dùng trước khi xóa data
echo "CẢNH BÁO: Script này sẽ XÓA PostgreSQL PVC và tất cả data trong đó!"
echo "Chỉ nên chạy trong môi trường dev/test."
read -p "Bạn có chắc chắn muốn tiếp tục? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Hủy bỏ."
    exit 0
fi

echo ""
echo "=== Bước 1: Xóa PostgreSQL cũ ==="
kubectl delete statefulset postgres -n $NS --ignore-not-found=true
echo "Đang đợi StatefulSet bị xóa..."
sleep 5

kubectl delete pvc postgres-data-postgres-0 -n $NS --ignore-not-found=true
echo "Đã xóa PVC"
sleep 3

echo ""
echo "=== Bước 2: Apply ConfigMap init script ==="
kubectl apply -f platform/kubernetes/postgres/postgres-init-configmap.yaml

echo ""
echo "=== Bước 3: Deploy lại PostgreSQL ==="
kubectl apply -f platform/kubernetes/postgres/postgres-secret.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-service.yaml
kubectl apply -f platform/kubernetes/postgres/postgres-statefulset.yaml

echo ""
echo "=== Bước 4: Đợi PostgreSQL sẵn sàng ==="
echo "Đang đợi PostgreSQL khởi động (timeout 180s)..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NS --timeout=180s

if [ $? -ne 0 ]; then
    echo "CẢNH BÁO: PostgreSQL không khởi động thành công!"
    echo "Kiểm tra logs:"
    kubectl logs postgres-0 -n $NS --tail=50
    exit 1
fi

echo ""
echo "=== Bước 5: Verify databases đã được tạo ==="
sleep 5
echo "Danh sách databases:"
kubectl exec -it postgres-0 -n $NS -- psql -U postgres -c "\l" | grep -E "airflow|image_db"

echo ""
echo "=== Bước 6: Xóa và restart Airflow components ==="
kubectl delete job airflow-init -n $NS --ignore-not-found=true
echo "Đã xóa airflow-init job"

echo "Restarting deployments..."
kubectl rollout restart deployment airflow-scheduler -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-webserver -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-worker -n $NS --ignore-not-found=true
kubectl rollout restart deployment airflow-triggerer -n $NS --ignore-not-found=true

echo ""
echo "=== Bước 7: Monitoring pods ==="
echo "Trạng thái hiện tại:"
kubectl get pods -n $NS

echo ""
echo "=== HOÀN THÀNH ==="
echo ""
echo "Để theo dõi pods realtime:"
echo "  kubectl get pods -n $NS -w"
echo ""
echo "Để xem logs của scheduler:"
echo "  kubectl logs -f deployment/airflow-scheduler -n $NS"
echo ""
echo "Để xem logs của webserver:"
echo "  kubectl logs -f deployment/airflow-webserver -n $NS"
echo ""
echo "Nếu vẫn còn lỗi, xem file FIX_AIRFLOW_DATABASE.md để debug."
