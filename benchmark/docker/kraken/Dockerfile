FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/kraken/requirements.txt docker/kraken/requirements.txt
RUN pip install --no-cache-dir --timeout 1200 --retries 20 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 && \
    pip install --no-cache-dir --timeout 1200 --retries 20 \
    -r docker/kraken/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/kraken docker/kraken

CMD ["uvicorn", "docker.kraken.app:app", "--host", "0.0.0.0", "--port", "8000"]
