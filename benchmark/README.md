# OCR Benchmark (HTR)

Lekki benchmark do porownywania modeli OCR/HTR na danych polskich.

## Co jest w projekcie

- `modele/`:
  - `base_wrapper.py` - abstrakcja `HTRModelWrapper`
  - `tesseract_pol_wrapper.py` - wrapper dla Tesseract POL
  - `rysocr_wrapper.py` - wrapper dla RysOCR (LoRA na PaddleOCR-VL)
- `src/`:
  - `data_generator.py` - loader probek i mapowanie `file_name -> image_path`
  - `metrics.py` - metryki (EMA, CER, WER, Levenshtein) + raporty
  - `evaluate.py` - glowny pipeline benchmarku
- `wyniki/` - zapisywane raporty CSV/JSON

## Szybki start

1. Zainstaluj zaleznosci:

```bash
pip install -r requirements.txt
```

2. Uruchom benchmark na Tesseract:

```bash
python src/evaluate.py --model tesseract_pol --limit 50
```

3. Uruchom benchmark na RysOCR:

```bash
python src/evaluate.py --model rysocr --limit 50
```

Uwaga: pierwsze uruchomienie `RysOCR` moze pobrac duze wagi modelu (ok. 2GB+).

## Tryb offline dla RysOCR

Po jednokrotnym pobraniu wag mozna uruchamiac offline:

```bash
python src/evaluate.py --model rysocr --rysocr-local-files-only --limit 50
```

Jesli cache jest niepelny, uruchom raz bez `--rysocr-local-files-only`.

## Argumenty przydatne dla RysOCR

- `--rysocr-adapter` - domyslnie `kacperwikiel/RysOCR`
- `--rysocr-base` - domyslnie `PaddlePaddle/PaddleOCR-VL`
- `--rysocr-device` - np. `cpu` lub `cuda`
- `--rysocr-max-new-tokens`
- `--rysocr-prompt`
- `--rysocr-local-files-only`

## Wyniki

Pipeline zapisuje:

- `wyniki/results_<timestamp>.csv`
- `wyniki/summary_<timestamp>.json`


