FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ENV MODULE_NAME=app

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y gcc \
    && apt-get update && apt-get install -y ncdu \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache
