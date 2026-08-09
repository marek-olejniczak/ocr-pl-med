# Training — fine-tuning modeli OCR i detekcji linii

Fine-tuning wybranych modeli OCR/HTR na danych polskich dokumentów medycznych.

## Struktura

```
training/
├── data/                    # dane treningowe (DVC)
│   ├── raw/                 # źródłowe wycinki linii (immutable)
│   └── processed/           # po konwersji do formatu per-model
├── ocr/                     # OCR fine-tuning
│   ├── common/              # współdzielone narzędzia
│   └── <model>/             # per-model: Dockerfile, cli.py, config
├── line_detection/          # detekcja linii (placeholder)
└── results/                 # checkpointy + logi (.gitignored)
```

## Wzorzec per-model

Każdy model w `ocr/<model>/` dostaje:
- `Dockerfile` — obraz PyTorch + zależności modelu
- `cli.py` — entry point z subkomendami `train` / `predict`
- `config.yaml` — domyślne hiperparametry
- `requirements.txt` — zależności Python

Uruchomienie treningu:
```bash
docker compose run --rm <model>-training python cli.py train --data ... --epochs 10
```

## Dane

Dane treningowe są wersjonowane przez DVC (remote: DagsHub).
Żeby pobrać dane:
```bash
dvc pull
```

## Walidacja

Wytrenowane checkpointy są walidowane przez istniejący benchmark w `benchmark/`.
Patrz: `benchmark/README.md` oraz `benchmark/docker/<model>/app.py`.

## Modele

| Model | Status | Uwagi |
|---|---|---|
| Surya | 🚧 w trakcie | LoRA fine-tuning na FoundationPredictor |
