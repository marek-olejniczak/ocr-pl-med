FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r docker/parseq/requirements.txt

CMD ["uvicorn", "docker.parseq.app:app", "--host", "0.0.0.0", "--port", "8000"]
