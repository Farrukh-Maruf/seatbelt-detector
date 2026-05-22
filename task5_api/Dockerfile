
# GPU host:  docker run -p 8000:8000 --gpus all seatbelt-detector
# CPU host:  docker run -p 8000:8000          seatbelt-detector
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Torch (CUDA build) first, then the rest. Pin to the cu124 index.
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code + weights (weights can also be mounted at runtime via -v)
COPY . .

EXPOSE 8000
# weights paths overridable via env (see main.py)
ENV YOLO_WEIGHTS=/app/task2_detection/runs/kamar_yolov8s/weights/best.pt \
    CLASSIFIER_WEIGHTS=/app/task3_classification/weights/mobilenet_v3_small.pt

CMD ["uvicorn", "task5_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
