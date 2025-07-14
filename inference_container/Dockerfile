# predict_container/Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY inference_container/. .
COPY shared ./shared
COPY train_container/m1.py ./train_container/m1.py

RUN pip install torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mlflow numpy pandas boto3

CMD ["python", "main.py"]
