from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import io
from minio import Minio
from minio.error import S3Error
import logging

log = logging.getLogger(__name__)

# --- CONFIG MINIO ---
MINIO_ENDPOINT_HOST = os.environ.get("MINIO_ENDPOINT", "minio:9000").replace("http://", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "nexusml"

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT_HOST,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

default_args = {
    'owner': 'data-team',
    'start_date': datetime(2025, 11, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    '01_data_etl_minio',
    default_args=default_args,
    schedule='0 2 * * *', # Chạy lúc 2h sáng
    catchup=False,
    description="ETL Pipeline: Extract -> Clean -> Trigger Training"
)

@task(dag=dag)
def ensure_bucket_exists():
    client = get_minio_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    
    # Tạo folder giả lập
    for folder in ["raw", "processed", "models"]:
        try:
            client.put_object(MINIO_BUCKET, f"{folder}/.keep", io.BytesIO(b""), 0)
        except: pass

@task(dag=dag)
def extract_and_combine_data():
    client = get_minio_client()
    objects = client.list_objects(MINIO_BUCKET, prefix="raw/", recursive=True)
    csv_files = [obj.object_name for obj in objects if obj.object_name.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("No CSV files found in raw/")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_temp = f"/tmp/combined_{timestamp}.csv"
    first = True
    
    for file_key in csv_files:
        response = None
        try:
            response = client.get_object(MINIO_BUCKET, file_key)
            with pd.read_csv(response, chunksize=10000) as reader:
                for chunk in reader:
                    mode = 'w' if first else 'a'
                    header = first
                    chunk.to_csv(local_temp, mode=mode, header=header, index=False)
                    first = False
        finally:
            if response: response.close()
            
    s3_dest = f"processed/combined_{timestamp}.csv"
    client.fput_object(MINIO_BUCKET, s3_dest, local_temp)
    if os.path.exists(local_temp): os.remove(local_temp)
    
    return s3_dest

@task(dag=dag)
def preprocess_data(s3_raw_key: str):
    client = get_minio_client()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_clean = f"/tmp/clean_{timestamp}.csv"
    
    response = None
    try:
        response = client.get_object(MINIO_BUCKET, s3_raw_key)
        first = True
        with pd.read_csv(response, chunksize=10000) as reader:
            for chunk in reader:
                if 'text' in chunk.columns:
                    chunk['text'] = chunk['text'].astype(str).str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
                
                mode = 'w' if first else 'a'
                header = first
                chunk.to_csv(local_clean, mode=mode, header=header, index=False)
                first = False
    finally:
        if response: response.close()

    s3_clean_key = f"processed/clean_data_{timestamp}.csv"
    client.fput_object(MINIO_BUCKET, s3_clean_key, local_clean)
    if os.path.exists(local_clean): os.remove(local_clean)
    
    return s3_clean_key

# --- FLOW ---
init = ensure_bucket_exists()
raw_file = extract_and_combine_data()
clean_file = preprocess_data(raw_file)

# Trigger DAG training và truyền đường dẫn file sạch qua configuration
trigger_ml = TriggerDagRunOperator(
    task_id='trigger_training',
    trigger_dag_id='02_model_training_minio', # ID của DAG 2
    conf={"s3_clean_key": "{{ task_instance.xcom_pull(task_ids='preprocess_data') }}"},
    dag=dag
)

init >> raw_file >> clean_file >> trigger_ml