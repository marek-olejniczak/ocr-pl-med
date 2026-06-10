FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libgl1 \
	libglib2.0-0 \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/calamari/requirements.txt docker/calamari/requirements.txt
RUN pip install --no-cache-dir -r docker/calamari/requirements.txt

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/calamari docker/calamari
COPY modele/__init__.py modele/__init__.py
COPY modele/base_wrapper.py modele/base_wrapper.py
COPY modele/calamari_wrapper.py modele/calamari_wrapper.py

CMD ["uvicorn", "docker.calamari.app:app", "--host", "0.0.0.0", "--port", "8000"]
