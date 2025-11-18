# Expose Services trên Minikube Azure VM

## Tình Huống
- ✅ Tất cả pods đang Running trong Minikube
- 📍 VM Azure: IP Public `40.82.143.98`, IP Private `172.16.0.4`
- 🔒 NSG hiện tại chỉ mở port 22 (SSH)

## Vấn Đề
Không thể truy cập Airflow, MLflow, MinIO, Grafana từ browser vì:
1. Services chỉ accessible trong cluster
2. Azure NSG chưa mở ports
3. Minikube không có LoadBalancer thực sự

---

## Giải Pháp 1: Expose Services qua NodePort + Mở Ports trên Azure NSG

### Bước 1: Chuyển Services sang NodePort

#### Airflow Webserver
```bash
kubectl patch service airflow-webserver-service -n nexusml -p '{"spec":{"type":"NodePort"}}'
```

Hoặc sửa file `platform/kubernetes/airflow/airflow-webserver-service.yaml`:
```yaml
spec:
  type: NodePort
  ports:
  - port: 8080
    targetPort: 8080
    nodePort: 30080  # Port cố định
  selector:
    app: airflow-webserver
```

#### MLflow
```bash
kubectl patch service mlflow-service -n nexusml -p '{"spec":{"type":"NodePort"}}'
```

Hoặc sửa file:
```yaml
spec:
  type: NodePort
  ports:
  - port: 5000
    targetPort: 5000
    nodePort: 30500
```

#### MinIO Console
```bash
kubectl patch service minio-service -n nexusml -p '{"spec":{"type":"NodePort"}}'
```

#### Grafana
```bash
kubectl patch service grafana-service -n nexusml -p '{"spec":{"type":"NodePort"}}'
```

### Bước 2: Lấy NodePort được assign
```bash
kubectl get services -n nexusml
```

Output mẫu:
```
NAME                          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
airflow-webserver-service     NodePort    10.96.x.x       <none>        8080:30080/TCP   20m
mlflow-service                NodePort    10.96.x.x       <none>        5000:30500/TCP   20m
minio-service                 NodePort    10.96.x.x       <none>        9000:30900/TCP   20m
grafana-service               NodePort    10.96.x.x       <none>        3000:30300/TCP   20m
```

### Bước 3: Lấy Minikube Node IP
```bash
minikube ip
```
Ví dụ: `192.168.49.2`

### Bước 4: Port Forward từ VM ra Internet

Vì Minikube IP là internal, cần forward từ VM:

```bash
# Forward Airflow
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j REDIRECT --to-port 30080

# Forward MLflow
sudo iptables -t nat -A PREROUTING -p tcp --dport 5000 -j REDIRECT --to-port 30500

# Forward MinIO
sudo iptables -t nat -A PREROUTING -p tcp --dport 9000 -j REDIRECT --to-port 30900

# Forward Grafana
sudo iptables -t nat -A PREROUTING -p tcp --dport 3000 -j REDIRECT --to-port 30300

# Save rules
sudo netsh advfirewall firewall add rule name="Airflow" dir=in action=allow protocol=TCP localport=8080
```

### Bước 5: Mở Ports trên Azure NSG

Vào Azure Portal → VM → Networking → Add inbound port rule:

**Rule 1: Airflow**
- Port: 8080
- Protocol: TCP
- Source: Any (hoặc IP của bạn)
- Action: Allow
- Priority: 310

**Rule 2: MLflow**
- Port: 5000
- Protocol: TCP
- Priority: 320

**Rule 3: MinIO**
- Port: 9000, 9001
- Protocol: TCP
- Priority: 330

**Rule 4: Grafana**
- Port: 3000
- Protocol: TCP
- Priority: 340

### Bước 6: Truy Cập Services
```
Airflow:  http://40.82.143.98:8080   (admin/admin)
MLflow:   http://40.82.143.98:5000
MinIO:    http://40.82.143.98:9000
Grafana:  http://40.82.143.98:3000   (admin/admin)
```

---

## Giải Pháp 2: SSH Tunnel + Port Forwarding (Đơn Giản Hơn)

### Cách này KHÔNG cần mở ports trên Azure NSG!

### Từ máy local của bạn, tạo SSH tunnel:

```bash
# SSH vào VM với port forwarding
ssh -L 8080:localhost:8080 \
    -L 5000:localhost:5000 \
    -L 9000:localhost:9000 \
    -L 3000:localhost:3000 \
    azureuser@40.82.143.98
```

### Sau khi SSH vào VM, chạy kubectl port-forward:

```bash
# Terminal 1: Airflow
kubectl port-forward -n nexusml service/airflow-webserver-service 8080:8080 --address=0.0.0.0

# Terminal 2: MLflow
kubectl port-forward -n nexusml service/mlflow-service 5000:5000 --address=0.0.0.0

# Terminal 3: MinIO
kubectl port-forward -n nexusml service/minio-service 9000:9000 --address=0.0.0.0

# Terminal 4: Grafana
kubectl port-forward -n nexusml service/grafana-service 3000:3000 --address=0.0.0.0
```

### Giờ từ browser trên máy local:
```
Airflow:  http://localhost:8080
MLflow:   http://localhost:5000
MinIO:    http://localhost:9000
Grafana:  http://localhost:3000
```

---

## Giải Pháp 3: Install Ingress Controller (Production-Ready)

### Cài MetalLB cho Minikube
```bash
minikube addons enable metallb

# Configure IP range
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: metallb-system
  name: config
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - 172.16.0.10-172.16.0.20
EOF
```

### Cài NGINX Ingress
```bash
minikube addons enable ingress
```

### Tạo Ingress Resource
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nexusml-ingress
  namespace: nexusml
spec:
  rules:
  - host: airflow.nexusml.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: airflow-webserver-service
            port:
              number: 8080
  - host: mlflow.nexusml.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlflow-service
            port:
              number: 5000
```

### Cập nhật /etc/hosts trên máy local
```
40.82.143.98  airflow.nexusml.local
40.82.143.98  mlflow.nexusml.local
40.82.143.98  minio.nexusml.local
40.82.143.98  grafana.nexusml.local
```

---

## Khuyến Nghị

**Cho Dev/Test**: Dùng **Giải pháp 2** (SSH Tunnel)
- ✅ Đơn giản nhất
- ✅ Bảo mật (không mở ports)
- ✅ Không cần config Azure NSG

**Cho Production**: Dùng **Giải pháp 3** (Ingress)
- ✅ Chuẩn Kubernetes
- ✅ Dễ quản lý
- ✅ Support SSL/TLS

**Nếu cần public access**: Dùng **Giải pháp 1** (NodePort + NSG)
- ⚠️ Nhớ restrict source IP trong NSG rules
- ⚠️ Cân nhắc thêm authentication

---

## Script Tự Động cho Giải Pháp 2

```bash
#!/bin/bash
# File: platform/kubernetes/port-forward-all.sh

echo "Starting port forwarding for all services..."

kubectl port-forward -n nexusml service/airflow-webserver-service 8080:8080 --address=0.0.0.0 &
PID1=$!

kubectl port-forward -n nexusml service/mlflow-service 5000:5000 --address=0.0.0.0 &
PID2=$!

kubectl port-forward -n nexusml service/minio-service 9000:9000 --address=0.0.0.0 &
PID3=$!

kubectl port-forward -n nexusml service/grafana-service 3000:3000 --address=0.0.0.0 &
PID4=$!

echo "Port forwarding started!"
echo "Airflow:  http://localhost:8080"
echo "MLflow:   http://localhost:5000"
echo "MinIO:    http://localhost:9000"
echo "Grafana:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all port forwarding"

# Cleanup on exit
trap "kill $PID1 $PID2 $PID3 $PID4 2>/dev/null" EXIT

wait
```

Chạy:
```bash
bash platform/kubernetes/port-forward-all.sh
```

---

## Credentials

### Airflow
- URL: `http://[IP]:8080`
- User: `admin`
- Pass: `admin`

### Grafana (nếu đã cài)
- URL: `http://[IP]:3000`
- User: `admin`
- Pass: Xem secret hoặc mặc định `admin`

### MLflow
- URL: `http://[IP]:5000`
- Không cần authentication

### MinIO
- URL: `http://[IP]:9000` (API) hoặc `:9001` (Console)
- Access Key: `minio-access-key`
- Secret Key: `minio-secret-key`
