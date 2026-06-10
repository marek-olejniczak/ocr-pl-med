FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/paddleocr/requirements.txt docker/paddleocr/requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r docker/paddleocr/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/paddleocr docker/paddleocr
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/paddleocr_wrapper.py modele/paddleocr_wrapper.py

CMD ["uvicorn", "docker.paddleocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
