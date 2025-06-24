# train_container/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Copy only what's needed
COPY . .
COPY ../shared ./shared

RUN pip install --no-cache-dir pandas numpy torch matplotlib seaborn scikit-learn

CMD ["python", "main.py"]
