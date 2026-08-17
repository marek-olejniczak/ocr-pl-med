# Surya OCR — fine-tuning (model foundation)

Fine-tuning modelu Surya (foundation, `surya-ocr==0.17.1`) na polskich dokumentach
medycznych. LoRA na dekoderze (attention-only).

## Ważne

Oficjalny `surya/scripts/finetune_ocr.py` w 0.17.1 jest **niekompatybilny** z modelem
foundation (brakuje mu `cache_position` i przeliczonych `image_embeddings`, których
wymaga `SuryaModel.forward`). Tutejszy `cli.py` poprawia to w collatorze — wzorowany
na działającym forku (zob. issue [datalab-to/surya#474](https://github.com/datalab-to/surya/issues/474)).

## Format danych

```
training/data/processed/surya/
├── train/
│   ├── metadata.jsonl   # {"file_name": "...", "text": "..."}
│   └── images/          # płaskie pliki PNG/JPG
└── val/
    ├── metadata.jsonl
    └── images/
```

Konwersję z surowych danych robi `convert_data.py` (subkomendy `ocr800k`, `handlabeled`),
**nie** `cli.py`.

## Szybki start (Docker)

```bash
# 0. (raz) uid/gid hosta, żeby kontener pisał pliki jako Ty
printf 'HUID=%s\nHGID=%s\n' "$(id -u)" "$(id -g)" > training/.env

# 1. Budowa obrazu
docker compose -f training/docker-compose.yml build surya-training

# 2. Konwersja danych (ocr800k → processed/surya)
docker compose -f training/docker-compose.yml run --rm surya-training \
    python training/ocr/surya/convert_data.py ocr800k \
    --input dataset/ocr_800k --output training/data/processed/surya

# 3. Trening (LoRA)
docker compose -f training/docker-compose.yml run --rm surya-training \
    python training/ocr/surya/cli.py train \
    --train-metadata training/data/processed/surya/train/metadata.jsonl \
    --train-images-dir training/data/processed/surya/train/images \
    --val-metadata training/data/processed/surya/val/metadata.jsonl \
    --val-images-dir training/data/processed/surya/val/images \
    --lora --report-to wandb --run-name run01 \
    --output-dir training/results/ocr/surya/run01
```

Smoke test (mała próbka, waliduje że pipeline nie crashuje):

```bash
docker compose -f training/docker-compose.yml run --rm surya-training \
    python training/ocr/surya/convert_data.py ocr800k \
    --input dataset/ocr_800k --output training/data/processed/surya --max-samples 500
# potem train z --max-steps 2 (limit kroków, żeby nie czekać na epokę)
```

## Ewaluacja

Wytrenowany adapter LoRA (katalog `adapter/` w `--output-dir`) jest walidowany przez
benchmark (`benchmark/docker/surya` ładuje go przez `lora_adapter_path`). W `cli.py`
nie ma subkomendy `predict` — za ewaluację odpowiada benchmark.

## W&B

`report-to wandb` włącza logowanie metryk (loss, lr, grad_norm) do projektu
`ocr-finetune` / entity `ocr-pl-med` (ustawiane przez env w `docker-compose.yml`).
Wymaga `WANDB_API_KEY` w środowisku hosta. Po treningu artefakty `surya-lora-best`
i `surya-lora-last` (katalogi adaptera LoRA) są logowane do W&B.

## Hiperparametry

`config.yaml` to dokumentacja domyślnych wartości; faktycznie obowiązują domyślne
z argparse w `cli.py` (wszystkie nadpisywalne przez CLI).

## Źródła

- Repo: https://github.com/datalab-to/surya
- Issue o zepsutym skrypcie: https://github.com/datalab-to/surya/issues/474
- Wersja w benchmarku: `surya-ocr==0.17.1`
