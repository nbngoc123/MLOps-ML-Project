kubectl apply -f 0-config-storage.yaml
kubectl apply -f 1-minio-stack.yaml
kubectl apply -f 2-mlflow-stack.yaml
kubectl apply -f 3-backend-stack.yaml
kubectl apply -f 4-frontend-stack.yaml