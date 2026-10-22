# Live Subtitle Engine Dockerfile
# Optimized for CPU inference with faster-whisper and OpenVINO/ONNX

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HF_HOME=/app/models

# Install system dependencies including ffmpeg for audio/video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install core Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir openvino>=2023.2.0

# Copy application source code
COPY . /app

# Expose WebSocket ASR server port
EXPOSE 8765

# Default command: launch WebSocket ASR server
CMD ["python", "backend/app_bridge.py", "--host", "0.0.0.0", "--model", "base", "--port", "8765"]
