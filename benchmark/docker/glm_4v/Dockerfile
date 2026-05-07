FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/glm_4v/requirements.txt docker/glm_4v/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r docker/glm_4v/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/glm_4v docker/glm_4v

CMD ["uvicorn", "docker.glm_4v.app:app", "--host", "0.0.0.0", "--port", "8000"]
