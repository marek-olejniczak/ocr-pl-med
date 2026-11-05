# Bazuje na obrazie treningowym (surya-ocr 0.17.1, transformers 4.56.1, torch 2.7.1,
# peft 0.20.0 — wszystko już zainstalowane) zamiast na czystym pytorch/pytorch.
#
# Dlaczego: data-root dockera leży na prawie pełnej partycji root (~2.5 GB wolnego).
# Standardowa przebudowa pisałaby 2-4 GB nowych warstw; baza z obrazu treningowego
# to delta ~100 MB (tylko zależności serwera HTTP). Wymaga wcześniejszego
# zbudowania training-surya-training na tym samym hoście.
FROM training-surya-training:latest

WORKDIR /app

COPY docker/common.py docker/common.py
COPY docker/logging_utils.py docker/logging_utils.py
COPY docker/surya docker/surya

# Obraz treningowy nie ma zależności serwera HTTP — doinstaluj minimalnie.
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "requests>=2.32.0"

CMD ["uvicorn", "docker.surya.app:app", "--host", "0.0.0.0", "--port", "8000"]
