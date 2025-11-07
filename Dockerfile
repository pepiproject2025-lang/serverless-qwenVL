FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /workspace

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 캐시 무효화용 빌드 augument
ARG CACHE_BREAKER=2025-11-07-1
RUN echo "CACHE_BREAKER=$CACHE_BREAKER"

# torch는 cu128 채널에서 설치
# torchvision/torchaudio가 필요하면 동일한 +cu128 태그로 맞춰 설치해야 함.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0+cu128

# 나머지 패키지
COPY requirements.txt .
RUN python3 -m pip install --no-chache-dir -r requirements.txt && \
    python - <<'PY'
import torch, sys
print("Torch: ", torch.__version__, "CUDA: ", torch.version.cuda)
print("CUDA available: ", torch.cuda.is_available())
try:
    print("capability: ", torch.cuda.get_device_capability())
except Exception as e:
    print("capability: N/A", e)
PY

# Runpod serverless entrypoint: handler.py with handler(event)
COPY handler.py .
CMD ["python3", "-u", "handler.py"]