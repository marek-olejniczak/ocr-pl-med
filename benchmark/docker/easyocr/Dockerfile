FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY docker/easyocr/requirements.txt docker/easyocr/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
	-r docker/easyocr/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/easyocr docker/easyocr
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/easyocr_wrapper.py modele/easyocr_wrapper.py

CMD ["uvicorn", "docker.easyocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
