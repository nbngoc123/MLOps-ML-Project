FROM apache/airflow:2.10.3

USER root
RUN apt-get update && \
    apt-get install -y git postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} 
# -r requirements.txt
