FROM apache/airflow:3.1.2

USER root

# Cài thêm các package hệ thống nếu cần
RUN apt-get update && \
    apt-get install -y git postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements
COPY requirements.txt .

# Cài dependencies python
RUN pip install --no-cache-dir -r requirements.txt
