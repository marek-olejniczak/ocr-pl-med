FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r docker/tesseract_pol/requirements.txt

CMD ["uvicorn", "docker.tesseract_pol.app:app", "--host", "0.0.0.0", "--port", "8000"]
