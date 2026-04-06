# OCR Benchmark (HTR)

Lekki benchmark do porownywania modeli OCR/HTR na danych polskich.

## Co jest w projekcie

- `modele/`:
  - `base_wrapper.py` - abstrakcja `HTRModelWrapper`
  - `tesseract_pol_wrapper.py` - wrapper dla Tesseract POL
  - `rysocr_wrapper.py` - wrapper dla RysOCR (LoRA na PaddleOCR-VL)
  - `trocr_wrapper.py` - wrapper dla TrOCR handwritten (Transformer OCR)
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

4. Uruchom RysOCR z wiekszym batchem na GPU:

```bash
python src/evaluate.py --model rysocr --rysocr-batch-size 4 --limit 50
```

5. Uruchom RysOCR z batchem i mixed precision (AMP):

```bash
python src/evaluate.py --model rysocr --rysocr-batch-size 4 --rysocr-use-amp --limit 50
```

6. Uruchom benchmark na TrOCR (wariant handwritten, lzejszy VRAM):

```bash
python src/evaluate.py --model trocr --limit 50
```

7. Uruchom TrOCR z wiekszym batchem na GPU:

```bash
python src/evaluate.py --model trocr --trocr-batch-size 8 --limit 50
```

8. Uruchom TrOCR z batchem i mixed precision (AMP):

```bash
python src/evaluate.py --model trocr --trocr-batch-size 8 --trocr-use-amp --limit 50
```

9. Uruchom TrOCR w wariancie base (mocniejsze GPU, wyzsze zuzycie VRAM):

```bash
python src/evaluate.py --model trocr --trocr-variant base --trocr-batch-size 4 --limit 50
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
- `--rysocr-batch-size` - domyslnie `2`, zwieksza throughput kosztem VRAM
- `--rysocr-use-amp` - opcjonalne mixed precision na CUDA (domyslnie wylaczone)
- `--rysocr-local-files-only`

## Argumenty przydatne dla TrOCR

- `--trocr-model-id` - domyslnie `microsoft/trocr-small-handwritten`
- `--trocr-variant` - `small` (domyslnie) lub `base`; wybiera domyslny checkpoint handwritten
- `--trocr-model-id` - opcjonalne nadpisanie checkpointu; gdy pominiete, bierze model z `--trocr-variant`
- `--trocr-device` - np. `cpu` lub `cuda`
- `--trocr-max-new-tokens`
- `--trocr-batch-size` - domyslnie `4`, zwieksza throughput kosztem VRAM
- `--trocr-use-amp` - opcjonalne mixed precision na CUDA (domyslnie wylaczone)
- `--trocr-local-files-only`

## TrOCR i jezyk polski

- Domyslny wariant TrOCR w benchmarku nie jest dotrenowany stricte na jezyku polskim.
- Nie nalezy oczekiwac stabilnego rozpoznawania polskich diakrytykow (np. ą, ć, ę, ł, ń, ó, ś, ź, ż).
- TrOCR traktuj jako punkt odniesienia dla transformera OCR, a nie model zoptymalizowany pod polskie dane.

## Wydajnosc RysOCR

- Domyslnie RysOCR dziala z `--rysocr-batch-size 2`, co zwykle lepiej wykorzystuje GPU niz inferencja pojedyncza.
- Przy ograniczonej pamieci GPU startuj od batcha 2 i stopniowo zwiekszaj (`4`, `8`) do momentu najlepszego kompromisu szybkosc/VRAM.
- `--rysocr-use-amp` przyspiesza inferencje na CUDA, ale na CPU jest ignorowane.

## Wyniki

Pipeline zapisuje:

- `wyniki/results_<timestamp>.csv`
- `wyniki/summary_<timestamp>.json`


