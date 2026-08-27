# TrOCR benchmark service
#
# Bazuje na obrazie treningowym TrOCR (training-trocr-training:latest: torch
# 2.7.1+cu118, transformers 5.0.0, peft, datasets, wandb). Polski TrOCR
# (PET3R12/trocr-base-polish-handwriting) i nasz fine-tune są zapisane z
# transformers 5.0.0 (plik processor_config.json) — na obrazie z transformers 4.x
# (dawny FROM pytorch/pytorch:2.6.0) się nie załadują.
# Delta względem obrazu treningowego jest mała (~100-200MB), co jest bezpieczne
# przy 99% pełnej partycji root /var/lib/docker.
FROM training-trocr-training:latest

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
