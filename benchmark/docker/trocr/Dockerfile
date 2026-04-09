FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
	--index-url https://download.pytorch.org/whl/cpu \
	torch==2.6.0 torchvision==0.21.0 && \
	pip install --no-cache-dir --timeout 300 --retries 10 \
	-r docker/trocr/requirements.txt

CMD ["uvicorn", "docker.trocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
