FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/surya/requirements.txt docker/surya/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1 && \
    pip install --no-cache-dir --timeout 300 --retries 10 \
    -r docker/surya/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/surya docker/surya

CMD ["uvicorn", "docker.surya.app:app", "--host", "0.0.0.0", "--port", "8000"]
