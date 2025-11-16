#!/bin/bash

echo "========================================="
echo "  Port Forwarding All NexusML Services"
echo "========================================="
echo ""

# Namespace
NS="nexusml"

# Kiểm tra các services có tồn tại không
echo "Checking services..."
kubectl get service airflow-webserver-service -n $NS >/dev/null 2>&1 || { echo "❌ Airflow service not found"; }
kubectl get service mlflow-service -n $NS >/dev/null 2>&1 || { echo "❌ MLflow service not found"; }
kubectl get service minio-service -n $NS >/dev/null 2>&1 || { echo "❌ MinIO service not found"; }

echo ""
echo "Starting port forwarding..."
echo ""

# Port forward Airflow
kubectl port-forward -n $NS service/airflow-webserver-service 8080:8080 --address=0.0.0.0 > /dev/null 2>&1 &
PID1=$!
echo "✅ Airflow:  http://localhost:8080 (admin/admin)"

# Port forward MLflow
kubectl port-forward -n $NS service/mlflow-service 5000:5000 --address=0.0.0.0 > /dev/null 2>&1 &
PID2=$!
echo "✅ MLflow:   http://localhost:5000"

# Port forward MinIO
kubectl port-forward -n $NS service/minio-service 9000:9000 9001:9001 --address=0.0.0.0 > /dev/null 2>&1 &
PID3=$!
echo "✅ MinIO:    http://localhost:9000 (API), http://localhost:9001 (Console)"

# Port forward Grafana (nếu có)
if kubectl get service grafana-service -n $NS >/dev/null 2>&1; then
    kubectl port-forward -n $NS service/grafana-service 3000:3000 --address=0.0.0.0 > /dev/null 2>&1 &
    PID4=$!
    echo "✅ Grafana:  http://localhost:3000"
fi

echo ""
echo "========================================="
echo "  All services are now accessible!"
echo "========================================="
echo ""
echo "📝 Từ máy local của bạn, chạy:"
echo "   ssh -i ~/.ssh/vm-k8s.pem -L 8080:localhost:8080 -L 5000:localhost:5000 -L 9000:localhost:9000 -L 9001:localhost:9001 -L 3000:localhost:3000 azureuser@40.82.143.98"
echo ""
echo "   Sau đó mở browser:"
echo "   - Airflow:  http://localhost:8080"
echo "   - MLflow:   http://localhost:5000"
echo "   - MinIO:    http://localhost:9000"
echo "   - Grafana:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all port forwarding"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all port forwarding..."
    kill $PID1 $PID2 $PID3 2>/dev/null
    [ ! -z "$PID4" ] && kill $PID4 2>/dev/null
    echo "Done!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Keep script running
wait
