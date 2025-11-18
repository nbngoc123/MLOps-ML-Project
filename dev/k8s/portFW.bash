#!/bin/bash

echo "========================================="
echo "  Port Forwarding 4 Services (default namespace)"
echo "========================================="
echo ""

# Services và port: <service>:<local-port>:<service-port>
declare -A PORTS
PORTS=(
    ["backend-service"]="8000:8000"
    ["frontend-service"]="3000:80"          # <-- sửa frontend port đúng
    ["mlflow-service"]="5000:5000"
    ["minio-service"]="9000:9000 9001:9001"
)

declare -A PIDS

echo "Starting port forwarding..."
for svc in "${!PORTS[@]}"; do
    ports=${PORTS[$svc]}
    for port_map in $ports; do
        kubectl port-forward service/$svc $port_map --address=0.0.0.0 >/dev/null 2>&1 &
        pid=$!
        PIDS["$svc:$port_map"]=$pid
        local_port=$(echo $port_map | cut -d':' -f1)
        echo "$svc → localhost:$local_port"
    done
done

echo ""
echo "========================================="
echo "  All services are now accessible!"
echo "========================================="
echo ""
echo "From your local machine, run SSH tunnel:"
echo "   ssh -i ~/.ssh/vm-k8s.pem \\"
echo "       -L 8000:localhost:8000 \\"
echo "       -L 3000:localhost:3000 \\"
echo "       -L 5000:localhost:5000 \\"
echo "       -L 9000:localhost:9000 \\"
echo "       -L 9001:localhost:9001 \\"
echo "       azureuser@40.82.143.98"
echo ""
echo "Then open browser / API client:"
echo "   - Backend API:   http://localhost:8000/docs"
echo "   - Frontend UI:   http://localhost:3000"
echo "   - MLflow UI:     http://localhost:5000"
echo "   - MinIO API:     http://localhost:9000"
echo "   - MinIO Console: http://localhost:9001"
echo ""
echo "Press Ctrl+C to stop all port forwarding."

# Cleanup khi Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all port forwarding..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "Done!"
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
