# Surya OCR — fine-tuning

Fine-tuning modelu rozpoznawania tekstu Surya na polskich dokumentach medycznych.

## Metodologia

Surya używa zmodyfikowanej architektury Donut (GQA, MoE, UTF-16 decoding, ~650M params).
Model ładuje się przez `FoundationPredictor` z pakietu `surya-ocr`.

Fine-tuning odbywa się przez oficjalny skrypt `surya/scripts/finetune_ocr.py`,
który używa HuggingFace `Trainer`. Dodatkowo wspieramy LoRA (PEFT) dla mniejszego
zużycia VRAM.

### Format danych

Dane w formacie HuggingFace `datasets` lub lokalnym:
- Obrazy: wycinki linii tekstu (PNG/JPG)
- Metadane: JSONL z polami `file_name` i `text`

Konwersja z `training/data/raw/` → format Surya: `python cli.py convert --input ... --output ...`

### Opcje treningu

1. **Full fine-tuning** — przez oficjalny skrypt `surya/scripts/finetune_ocr.py`
2. **LoRA** — przez `--lora` flagę, mniejsze zużycie VRAM (testowane na GTX 1650 4GB)

## Szybki start

```bash
# 1. Budowa obrazu
docker compose build surya-training

# 2. Konwersja danych
docker compose run --rm surya-training python cli.py convert \
    --input data/raw --output data/processed/surya

# 3. Trening (LoRA, lokalnie/testowo)
docker compose run --rm surya-training python cli.py train \
    --data data/processed/surya/train \
    --lora --epochs 10 --batch-size 4

# 4. Predykcja (walidacja na zbiorze testowym)
docker compose run --rm surya-training python cli.py predict \
    --checkpoint results/ocr/surya/<run_id>/checkpoint \
    --data data/processed/surya/val
```

## Hiperparametry

Domyślne wartości w `config.yaml`. Wszystkie nadpisywalne przez CLI.

## Źródła

- Repo: https://github.com/datalab-to/surya
- Dokumentacja: https://surya.readthedocs.io/
- Wersja w benchmarku: `surya-ocr==0.17.1`
