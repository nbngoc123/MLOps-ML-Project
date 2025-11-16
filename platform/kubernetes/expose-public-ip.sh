#!/bin/bash

echo "========================================="
echo "  Expose Services qua Public IP"
echo "  Azure VM: 40.82.143.98"
echo "========================================="
echo ""

NS="nexusml"

echo "Bước 1: Chuyển Services sang NodePort..."
echo ""

# Airflow Webserver
echo "✅ Configuring Airflow Webserver..."
kubectl patch service airflow-webserver-service -n $NS -p '{"spec":{"type":"NodePort","ports":[{"port":8080,"targetPort":8080,"nodePort":30080}]}}'

# MLflow
echo "✅ Configuring MLflow..."
kubectl patch service mlflow-service -n $NS -p '{"spec":{"type":"NodePort","ports":[{"port":5000,"targetPort":5000,"nodePort":30500}]}}'

# MinIO
echo "✅ Configuring MinIO..."
kubectl patch service minio-service -n $NS -p '{"spec":{"type":"NodePort","ports":[{"port":9000,"targetPort":9000,"nodePort":30900,"name":"api"},{"port":9001,"targetPort":9001,"nodePort":30901,"name":"console"}]}}'

# Grafana (nếu có)
if kubectl get service grafana-service -n $NS >/dev/null 2>&1; then
    echo "✅ Configuring Grafana..."
    kubectl patch service grafana-service -n $NS -p '{"spec":{"type":"NodePort","ports":[{"port":3000,"targetPort":3000,"nodePort":30300}]}}'
fi

echo ""
echo "Bước 2: Lấy thông tin services..."
echo ""
kubectl get services -n $NS

echo ""
echo "Bước 3: Lấy Minikube IP..."
MINIKUBE_IP=$(minikube ip)
echo "Minikube IP: $MINIKUBE_IP"

echo ""
echo "========================================="
echo "  Services đã expose qua NodePort!"
echo "========================================="
echo ""
echo "📝 QUAN TRỌNG: Cần làm 2 việc tiếp theo:"
echo ""
echo "1️⃣  MỞ PORTS TRÊN AZURE NSG:"
echo "   Vào Azure Portal → VM → Networking → Add inbound security rule"
echo ""
echo "   Rule 1 - Airflow:"
echo "   - Destination port ranges: 8080"
echo "   - Protocol: TCP"
echo "   - Source: Any (hoặc IP của bạn)"
echo "   - Action: Allow"
echo "   - Priority: 310"
echo "   - Name: Allow-Airflow"
echo ""
echo "   Rule 2 - MLflow:"
echo "   - Port: 5000, Priority: 320"
echo ""
echo "   Rule 3 - MinIO:"
echo "   - Port: 9000,9001, Priority: 330"
echo ""
echo "   Rule 4 - Grafana:"
echo "   - Port: 3000, Priority: 340"
echo ""
echo "2️⃣  PORT FORWARDING TRÊN VM:"
echo "   (Chạy các lệnh này để forward từ VM port sang Minikube NodePort)"
echo ""
echo "   sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 8080 -j DNAT --to-destination $MINIKUBE_IP:30080"
echo "   sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 5000 -j DNAT --to-destination $MINIKUBE_IP:30500"
echo "   sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 9000 -j DNAT --to-destination $MINIKUBE_IP:30900"
echo "   sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 9001 -j DNAT --to-destination $MINIKUBE_IP:30901"
echo "   sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 3000 -j DNAT --to-destination $MINIKUBE_IP:30300"
echo ""
echo "   # Enable IP forwarding"
echo "   sudo sysctl -w net.ipv4.ip_forward=1"
echo "   sudo iptables -t nat -A POSTROUTING -j MASQUERADE"
echo ""
echo "3️⃣  SAU KHI HOÀN THÀNH, TRUY CẬP:"
echo ""
echo "   Airflow:      http://40.82.143.98:8080 (admin/admin)"
echo "   MLflow:       http://40.82.143.98:5000"
echo "   MinIO API:    http://40.82.143.98:9000"
echo "   MinIO Console: http://40.82.143.98:9001"
echo "   Grafana:      http://40.82.143.98:3000"
echo ""
echo "========================================="
echo ""
echo "💡 Lưu ý:"
echo "   - Cần quyền sudo để chạy iptables"
echo "   - iptables rules sẽ mất sau khi reboot VM"
echo "   - Để persistent, cài iptables-persistent:"
echo "     sudo apt-get install iptables-persistent"
echo "     sudo netfilter-persistent save"
echo ""
