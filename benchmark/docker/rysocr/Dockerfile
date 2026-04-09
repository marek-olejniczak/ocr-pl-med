FROM python:3.11-slim

WORKDIR /app

COPY docker/rysocr/requirements.txt docker/rysocr/requirements.txt
RUN pip install --no-cache-dir -r docker/rysocr/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/rysocr docker/rysocr
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/rysocr_wrapper.py modele/rysocr_wrapper.py

CMD ["uvicorn", "docker.rysocr.app:app", "--host", "0.0.0.0", "--port", "8000"]
