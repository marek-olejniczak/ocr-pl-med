FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/parseq/requirements.txt docker/parseq/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
	-r docker/parseq/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/parseq docker/parseq
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/parseq_wrapper.py modele/parseq_wrapper.py

CMD ["uvicorn", "docker.parseq.app:app", "--host", "0.0.0.0", "--port", "8000"]
