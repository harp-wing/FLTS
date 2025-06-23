# eda_container/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir pandas matplotlib seaborn

CMD ["python", "main.py"]
