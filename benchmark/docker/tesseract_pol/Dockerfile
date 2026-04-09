FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/tesseract_pol/requirements.txt docker/tesseract_pol/requirements.txt
RUN pip install --no-cache-dir -r docker/tesseract_pol/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/tesseract_pol docker/tesseract_pol
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/tesseract_pol_wrapper.py modele/tesseract_pol_wrapper.py

CMD ["uvicorn", "docker.tesseract_pol.app:app", "--host", "0.0.0.0", "--port", "8000"]
