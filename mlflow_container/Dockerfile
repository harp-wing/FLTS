# mlflow_container/Dockerfile
FROM ghcr.io/mlflow/mlflow:v2.12.1

# Install psycopg2 (needed for Postgres backend)
RUN pip install "mlflow[extras]==3.1.1" psycopg2-binary awscli awscli-local
