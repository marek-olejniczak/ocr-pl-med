# PARSeq (docTR recognition) training image
#
# Bazuje na obrazie treningowym Suryi (training-surya-training:latest: torch
# 2.7.1+cu118, wandb) zamiast na pytorch/pytorch — na serwerze /var/lib/docker
# siedzi na 99% pełnej partycji root, więc delta kilkunastu-warstwowego obrazu
# jest bezpieczna (~100-300MB) w przeciwieństwie do pełnego rebuildu (2-4GB).
#
# DocTR NIE jest transformers — to osobny model (własna pętla treningowa w
# cli.py, nie HF Trainer). Jedyne czego brakuje względem obrazu Suryi to:
#   * python-doctr (ciągnie opencv-python — PEŁNY, nie headless — który wymaga
#     systemowych bibliotek GL; stąd apt-get libgl1/libglib2.0-0),
#   * moment. trocr/transformers tu nie wchodzą.
FROM training-surya-training:latest

# Pełny opencv-python (zależność docTR) potrzebuje libGL / libglib w runtime.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Zależności treningowe (torch/torchvision/wandb już w bazie — patrz requirements.txt)
COPY training/ocr/parseq/requirements.txt training/ocr/parseq/requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -r training/ocr/parseq/requirements.txt

# Kod treningowy
COPY training/ocr/parseq training/ocr/parseq

# Brak ENTRYPOINT — trening odpala cli.py, komendę podaje się jawnie
# (docker compose run ... <komenda>).
CMD ["python", "training/ocr/parseq/cli.py", "train", "--help"]
