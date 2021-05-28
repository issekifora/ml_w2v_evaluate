FROM python:3.6

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip \
&& pip3 install -r requirements.txt