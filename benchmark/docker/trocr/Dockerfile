FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY docker/trocr/requirements.txt docker/trocr/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r docker/trocr/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/trocr docker/trocr
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/trocr_wrapper.py modele/trocr_wrapper.py

CMD ["uvicorn", "docker.trocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
