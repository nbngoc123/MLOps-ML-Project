#!/bin/bash

echo "========================================="
echo "  Deploy Monitoring Stack"
echo "  Prometheus + Grafana"
echo "========================================="
echo ""

NS="nexusml"

echo "Bước 1: Deploy Prometheus..."
echo ""

# PVC
echo "✅ Creating Prometheus PVC..."
kubectl apply -f platform/kubernetes/monitoring/prometheus-pvc.yaml

# ConfigMap
echo "✅ Applying Prometheus Config..."
kubectl apply -f platform/kubernetes/monitoring/prometheus-configmap.yaml

# Service
echo "✅ Creating Prometheus Service..."
kubectl apply -f platform/kubernetes/monitoring/prometheus-service.yaml

# StatefulSet
echo "✅ Deploying Prometheus StatefulSet..."
kubectl apply -f platform/kubernetes/monitoring/prometheus-statefulset.yaml

echo ""
echo "Đang đợi Prometheus sẵn sàng..."
kubectl wait --for=condition=ready pod -l app=prometheus -n $NS --timeout=120s

if [ $? -eq 0 ]; then
    echo "✅ Prometheus đã sẵn sàng!"
else
    echo "⚠️  Prometheus chưa sẵn sàng, kiểm tra logs:"
    kubectl logs -l app=prometheus -n $NS --tail=20
fi

echo ""
echo "========================================="
echo "Bước 2: Deploy Grafana..."
echo ""

# PVC
echo "✅ Creating Grafana PVC..."
kubectl apply -f platform/kubernetes/monitoring/grafana-pvc.yaml

# Deployment
echo "✅ Deploying Grafana..."
kubectl apply -f platform/kubernetes/monitoring/grafana-deployment.yaml

# Service
echo "✅ Creating Grafana Service..."
kubectl apply -f platform/kubernetes/monitoring/grafana-service.yaml

echo ""
echo "Đang đợi Grafana sẵn sàng..."
kubectl wait --for=condition=ready pod -l app=grafana -n $NS --timeout=120s

if [ $? -eq 0 ]; then
    echo "✅ Grafana đã sẵn sàng!"
else
    echo "⚠️  Grafana chưa sẵn sàng, kiểm tra logs:"
    kubectl logs -l app=grafana -n $NS --tail=20
fi

echo ""
echo "========================================="
echo "  Deployment Hoàn Thành!"
echo "========================================="
echo ""

# Get pod status
echo "📊 Trạng thái Pods:"
kubectl get pods -n $NS -l 'app in (prometheus,grafana)'

echo ""
echo "🌐 Services:"
kubectl get services -n $NS -l 'app in (prometheus,grafana)'

echo ""
echo "========================================="
echo "  Truy Cập Monitoring"
echo "========================================="
echo ""
echo "📝 Chạy port-forward để truy cập:"
echo ""
echo "   # Prometheus"
echo "   kubectl port-forward -n $NS service/prometheus-service 9090:9090 --address=0.0.0.0"
echo "   → http://localhost:9090"
echo ""
echo "   # Grafana"
echo "   kubectl port-forward -n $NS service/grafana-service 3000:3000 --address=0.0.0.0"
echo "   → http://localhost:3000"
echo "   Login: admin/admin (đổi password lần đầu)"
echo ""
echo "📝 Hoặc expose qua public IP:"
echo "   bash platform/kubernetes/expose-public-ip.sh"
echo ""
echo "========================================="
echo "  Setup Grafana (Lần Đầu)"
echo "========================================="
echo ""
echo "1. Truy cập Grafana: http://[IP]:3000"
echo "2. Login: admin / admin"
echo "3. Add Datasource:"
echo "   - Type: Prometheus"
echo "   - URL: http://prometheus-service.nexusml.svc.cluster.local:9090"
echo "   - Save & Test"
echo "4. Import Dashboards (optional)"
echo ""
echo "✅ Monitoring stack đã sẵn sàng!"
echo ""
