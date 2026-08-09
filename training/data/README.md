# Dane treningowe

Dane są wersjonowane przez DVC. Remote: DagsHub.

## Pobieranie

```bash
dvc pull
```

## Struktura

```
data/
├── raw/                         # źródłowe wycinki linii (obrazy + metadane)
│   ├── .gitkeep
│   └── *.dvc                    # DVC trackers
└── processed/                   # po konwersji do formatu per-model
    └── surya/
        ├── train/               # dane treningowe w formacie Surya
        │   ├── metadata.jsonl   # {"file_name": "...", "text": "..."}
        │   └── images/
        └── val/                 # dane walidacyjne
            ├── metadata.jsonl
            └── images/
```

## Format danych źródłowych (raw)

Wycinki linii tekstu — obrazy PNG/JPG + plik metadanych (CSV/JSONL) z kolumnami:
- `file_name` — nazwa pliku obrazu
- `text` — transkrypcja (ground truth)

## Konwersja do formatu per-model

Każdy model może wymagać innego formatu danych wejściowych.
Konwertery znajdują się w `training/ocr/<model>/convert_data.py`.
