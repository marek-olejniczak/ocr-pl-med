FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r docker/paddleocr/requirements.txt

CMD ["uvicorn", "docker.paddleocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
