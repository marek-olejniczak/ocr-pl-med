FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/qwen2_5_vl/requirements.txt docker/qwen2_5_vl/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r docker/qwen2_5_vl/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/qwen2_5_vl docker/qwen2_5_vl

CMD ["uvicorn", "docker.qwen2_5_vl.app:app", "--host", "0.0.0.0", "--port", "8000"]
