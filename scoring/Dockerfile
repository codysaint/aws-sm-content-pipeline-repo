# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set some environment variables. 
# PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
	

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

# Copy the inference code and requirements file -- scoring > inference.py
COPY scoring /opt/program
WORKDIR /opt/program

# Specify the entrypoint command
ENTRYPOINT ["python", "./scoring/inference.py"]
