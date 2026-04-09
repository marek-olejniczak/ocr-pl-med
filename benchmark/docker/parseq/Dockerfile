FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/parseq/requirements.txt docker/parseq/requirements.txt
RUN pip install --no-cache-dir -r docker/parseq/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/parseq docker/parseq
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/parseq_wrapper.py modele/parseq_wrapper.py

CMD ["uvicorn", "docker.parseq.app:app", "--host", "0.0.0.0", "--port", "8000"]
