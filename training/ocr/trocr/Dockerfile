# TrOCR (VisionEncoderDecoderModel) training image
#
# Bazuje na obrazie treningowym Suryi (training-surya-training:latest: torch
# 2.7.1+cu118, peft, accelerate, datasets, wandb) zamiast na pytorch/pytorch —
# na serwerze /var/lib/docker siedzi na 99% pełnej partycji root, więc delta
# kilkunastu-warstwowego obrazu jest bezpieczna (~100-300MB) w przeciwieństwie
# do pełnego rebuildu (2-4GB).
#
# Jedyne czego tu brakuje względem obrazu Suryi to transformers 5.0.0: polski
# TrOCR (PET3R12/trocr-base-polish-handwriting) jest zapisany z transformers
# 5.0.0 (plik processor_config.json zamiast preprocessor_config.json) i nie
# ładuje się na transformers 4.56.1.
FROM training-surya-training:latest

WORKDIR /app

# Zależności treningowe (w większości już w bazie — patrz requirements.txt)
COPY training/ocr/trocr/requirements.txt training/ocr/trocr/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r training/ocr/trocr/requirements.txt

# Kod treningowy
COPY training/ocr/trocr training/ocr/trocr

# Brak ENTRYPOINT — trening odpala cli.py, komendę podaje się jawnie
# (docker compose run ... <komenda>).
CMD ["python", "training/ocr/trocr/cli.py", "train", "--help"]
