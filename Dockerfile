FROM astrocrpublic.azurecr.io/runtime:3.3-7

# install additional syatem dependencies for ml library
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    && re -rf /var/lib/apt/lists/*

USER astro


ENV MLFLOW_TRACKING_URI = http://mlflow:5001
ENV MLFLOW_S3_ENDPOINT_URL = http://minio:9000
ENV AWS_ACCESS_KEY_ID = minioadmin
ENV AWS_SECRET_ACCESSS_KEY = minioadmin
ENV AWS_DEFAULT_REGION = us-east-1

