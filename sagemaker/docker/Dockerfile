# Dockerfile for SageMaker Training
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements and install Python dependencies
COPY sagemaker/docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY sagemaker/scripts/ ./

# Set Python path
ENV PYTHONPATH="/opt/ml/code:$PYTHONPATH"

# Set default command for training
ENTRYPOINT ["python", "train_sagemaker.py"]