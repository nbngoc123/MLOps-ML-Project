#!/bin/bash

echo "========================================="
echo "  Port Forwarding All NexusML Services"
echo "========================================="
echo ""

NS="nexusml"

service_exists() {
    kubectl get svc "$1" -n "$NS" >/dev/null 2>&1
}

echo "Checking services..."
for svc in airflow-webserver-service mlflow-service minio-service prometheus-service grafana-service backend frontend; do
    if service_exists $svc; then
        echo "Found: $svc"
    else
        if [[ "$svc" == "prometheus-service" || "$svc" == "grafana-service" ]]; then
            echo "Warning: $svc not found (optional)"
        else
            echo "Error: $svc not found"
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
    ["backend"]="8000:8000"           
    ["frontend"]="7860:7860"          
)

declare -A PIDS

for svc in "${!PORTS[@]}"; do
    if service_exists $svc; then
        ports=${PORTS[$svc]}
        for port_map in $ports; do
            local_port=$(echo $port_map | cut -d':' -f1)
            kubectl port-forward -n $NS service/$svc $port_map --address=0.0.0.0 >/dev/null 2>&1 &
            pid=$!
            PIDS["$svc:$local_port"]=$pid
            echo "Found: $svc → localhost:$local_port"
        done
    fi
done

echo ""
echo "========================================="
echo "  All services are now accessible!"
echo "========================================="
echo ""
echo "From your local machine, run SSH tunnel:"
echo "   ssh -L 8080:localhost:8080 -L 5000:localhost:5000 -L 9000:localhost:9000 -L 9001:localhost:9001 -L 9090:localhost:9090 -L 3000:localhost:3000 -L 8000:localhost:8000 -L 7860:localhost:7860 azureuser@40.82.143.98"
echo ""
echo "Then open browser:"
echo "   - Airflow:     http://localhost:8080 (admin/admin)"
echo "   - MLflow:      http://localhost:5000"
echo "   - MinIO API:   http://localhost:9000"
echo "   - MinIO UI:    http://localhost:9001"
echo "   - Prometheus:  http://localhost:9090"
echo "   - Grafana:     http://localhost:3000 (admin/admin)"
echo "   - Backend API: http://localhost:8000/docs"
echo "   - Frontend UI: http://localhost:7860"    
echo ""
echo "Press Ctrl+C to stop all port forwarding"
echo ""

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

wait