# eval_container/Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir numpy matplotlib scikit-learn

CMD ["python", "main.py"]
