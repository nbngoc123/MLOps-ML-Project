FROM apache/airflow:3.1.3

COPY requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

WORKDIR /opt/airflow