FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8

WORKDIR /home/site/wwwroot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y