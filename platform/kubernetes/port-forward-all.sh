#!/bin/bash

echo "========================================="
echo "  Port Forwarding All NexusML Services"
echo "========================================="
echo ""

# Namespace
NS="nexusml"

# Function: check if service exists
service_exists() {
    kubectl get svc "$1" -n "$NS" >/dev/null 2>&1
}

echo "Checking services..."
for svc in airflow-webserver-service mlflow-service minio-service prometheus-service grafana-service; do
    if service_exists $svc; then
        echo "✅ $svc exists"
    else
        if [[ "$svc" == "prometheus-service" || "$svc" == "grafana-service" ]]; then
            echo "⚠️  $svc not found (optional)"
        else
            echo "❌ $svc not found"
        fi
    fi
done

echo ""
echo "Starting port forwarding..."
echo ""

declare -A PORTS
PORTS=(
    ["airflow-webserver-service"]="8080:8080"
    ["mlflow-service"]="5000:5000"
    ["minio-service"]="9000:9000 9001:9001"
    ["prometheus-service"]="9090:9090"
    ["grafana-service"]="3000:3000"
)

declare -A PIDS

for svc in "${!PORTS[@]}"; do
    if service_exists $svc; then
        kubectl port-forward -n $NS service/$svc ${PORTS[$svc]} --address=0.0.0.0 >/dev/null 2>&1 &
        PIDS[$svc]=$!
        echo "✅ $svc forwarded: ${PORTS[$svc]}"
    fi
done

echo ""
echo "========================================="
echo "  All available services are now accessible!"
echo "========================================="
echo ""
echo "📝 From your local machine, run SSH tunnel:"
echo "   ssh -L 8080:localhost:8080 -L 5000:localhost:5000 -L 9000:localhost:9000 -L 9001:localhost:9001 -L 9090:localhost:9090 -L 3000:localhost:3000 azureuser@40.82.143.98"
echo ""
echo "Then open browser:"
echo "   - Airflow:    http://localhost:8080 (admin/admin)"
echo "   - MLflow:     http://localhost:5000"
echo "   - MinIO API:  http://localhost:9000"
echo "   - MinIO UI:   http://localhost:9001"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "Press Ctrl+C to stop all port forwarding"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping all port forwarding..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "Done!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Keep script running
wait
