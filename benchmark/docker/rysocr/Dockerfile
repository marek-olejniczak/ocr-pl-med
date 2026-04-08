FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r docker/rysocr/requirements.txt

CMD ["uvicorn", "docker.rysocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
