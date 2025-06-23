# eval_container/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir numpy matplotlib scikit-learn

CMD ["python", "main.py"]
