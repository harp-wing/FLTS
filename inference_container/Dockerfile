# predict_container/Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY . .
COPY ../shared ./shared
COPY ../train_container/m1.py ./train_container/m1.py

RUN pip install --no-cache-dir pandas numpy torch

CMD ["python", "main.py"]
