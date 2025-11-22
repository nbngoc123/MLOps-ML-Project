#!/bin/bash

echo "Your MLOps services are running on these URLs:"
echo "----------------------------------------------"
echo "FrontEnd:       http://localhost:7860"
echo "Backend API:    http://localhost:8000"
echo "MLflow:         http://localhost:5000"
echo "Airflow UI:     http://localhost:8080  (username: admin / password: admin)"
echo "Grafana:        http://localhost:3000  (username: admin / password: admin)"
echo "Prometheus:     http://localhost:9090"
echo "Alertmanager:   http://localhost:9093"
echo "Node Exporter:  http://localhost:9100"
echo "Loki:           http://localhost:3100"
echo "MinIO Console:  http://localhost:9000  (username: minioadmin / password: minioadmin)"
echo "----------------------------------------------"


echo "ssh -i ~/.ssh/vm-k8s.pem -L 8000:localhost:8000 -L 3000:localhost:3000 -L 5000:localhost:5000 -L 9000:localhost:9000 -L 9001:localhost:9001 -L 7860:localhost:7860 azureuser@40.82.143.98"




