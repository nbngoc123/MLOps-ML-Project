FROM apache/airflow:3.1.2

USER root

# Cài thêm các package hệ thống
RUN apt-get update && \
    apt-get install -y git postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .
COPY dags/ /opt/airflow/dags/

# Cài thêm 'apache-airflow-providers-fab' để có lệnh 'airflow users' và giao diện UI cũ
RUN pip install --no-cache-dir apache-airflow-providers-fab -r requirements.txt