FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

#https://forums.fast.ai/t/docker-image-went-from-606mb-to-2-95gb-by-adding-fastai/61280/3
ENV MODULE_NAME=app

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y gcc \
    && apt-get update && apt-get install -y ncdu \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt
COPY ./imagenet_class_index.json ./imagenet_class_index.json
COPY ./helpers.py ./helpers.py 

RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt --default-timeout=100 future

RUN rm -rf /root/.cache
