# Surya OCR training image
# Wzorowany na benchmark/docker/surya/Dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Współdzielone narzędzia benchmarku (logowanie, common)
COPY benchmark/docker/common.py benchmark/docker/common.py
COPY benchmark/docker/logging_utils.py benchmark/docker/logging_utils.py

# Zależności treningowe
COPY training/ocr/surya/requirements.txt training/ocr/surya/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r training/ocr/surya/requirements.txt

# Kod treningowy
COPY training/ocr/surya training/ocr/surya
COPY training/ocr/common training/ocr/common

# Dane treningowe (bind-mount w docker compose)
# COPY training/data training/data

# Brak ENTRYPOINT — konwersję robi convert_data.py, a trening cli.py,
# więc komendę podaje się jawnie (docker compose run ... <komenda>).
CMD ["python", "training/ocr/surya/cli.py", "--help"]
