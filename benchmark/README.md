# OCR Benchmark (HTR)

Lekki benchmark do porownywania modeli OCR/HTR na danych polskich.

## Co jest w projekcie

- `modele/`:
  - `base_wrapper.py` - abstrakcja `HTRModelWrapper`
  - `tesseract_pol_wrapper.py` - wrapper dla Tesseract POL
  - `rysocr_wrapper.py` - wrapper dla RysOCR (LoRA na PaddleOCR-VL)
  - `trocr_wrapper.py` - wrapper dla TrOCR handwritten (Transformer OCR)
  - `paddleocr_wrapper.py` - wrapper dla PaddleOCR PP-OCRv4 (mobile/server, recognition-only)
  - `easyocr_wrapper.py` - wrapper dla EasyOCR (GPU/CPU, batching, lokalny cache wag)
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

10. Uruchom PaddleOCR PP-OCRv4 w wariancie mobile (wydajnosc):

```bash
python src/evaluate.py --model paddleocr --paddleocr-variant mobile --paddleocr-device cpu --limit 50
```

11. Uruchom PaddleOCR PP-OCRv4 w wariancie server (wyzsza precyzja):

```bash
python src/evaluate.py --model paddleocr --paddleocr-variant server --paddleocr-device cpu --limit 50
```

12. Uruchom EasyOCR na GPU (jezyk polski + angielski):

```bash
python src/evaluate.py --model easyocr --easyocr-device cuda --easyocr-langs pl,en --limit 50
```

13. Uruchom EasyOCR z lokalnym cache wag i batchingiem:

```bash
python src/evaluate.py --model easyocr --easyocr-device cuda --easyocr-batch-size 16 --easyocr-model-storage-dir modele/cache/easyocr --limit 50
```

Uwaga: PaddleOCR wymaga dodatkowo backendu PaddlePaddle.

Aktualna rekomendacja dla tego repo: profil CPU (Python 3.12).

Rekomendowane profile instalacji:
- Python 3.12 (CPU): `paddlepaddle==3.3.1` + `paddleocr/paddlex 3.4.x`
- GPU na tym hostcie: dostepne sa tylko wheel `paddlepaddle-gpu 2.6.x`, co oznacza profil legacy (`paddleocr/paddlex 2.x`) i najlepiej starszy Python (np. 3.10)

Przykladowe komendy instalacji:
- Py3.12 CPU: `python -m pip install "paddlepaddle==3.3.1" "paddleocr>=3.4.0,<3.5.0" "paddlex>=3.4.3,<3.5.0"`
- Legacy GPU (Py<=3.10): `python -m pip install "paddlepaddle-gpu==2.6.2" "paddleocr==2.10.0" "paddlex==2.1.0"`

Niezgodny mix wersji (np. `paddleocr/paddlex 3.x` z `paddle 2.x`) moze konczyc sie bledem C++ (`SIGSEGV`).

W praktyce benchmark utrzymujemy obecnie i testujemy na CPU, dlatego przy uruchomieniach PaddleOCR zalecane jest jawne `--paddleocr-device cpu`.

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

- `--trocr-variant` - `small` (domyslnie) lub `base`; wybiera domyslny checkpoint handwritten
- `--trocr-model-id` - opcjonalne nadpisanie checkpointu; gdy pominiete, bierze model z `--trocr-variant`
- `--trocr-device` - np. `cpu` lub `cuda`
- `--trocr-max-new-tokens`
- `--trocr-batch-size` - domyslnie `4`, zwieksza throughput kosztem VRAM
- `--trocr-use-amp` - opcjonalne mixed precision na CUDA (domyslnie wylaczone)
- `--trocr-local-files-only`

## Argumenty przydatne dla PaddleOCR

- `--paddleocr-variant` - `mobile` (domyslnie) lub `server`
- `--paddleocr-rec-model-name` - opcjonalne nadpisanie modelu rec; gdy pominiete, bierze model z `--paddleocr-variant`
- `--paddleocr-lang` - domyslnie `pl`; istotne glownie dla fallbacku legacy OCR
- `--paddleocr-device` - `auto`, `cpu` lub `gpu`
- `--paddleocr-use-angle-cls` - wlacza klasyfikator kata (CLS)
- `--paddleocr-rec-batch-size` - batch size dla rec (domyslnie `8`)

## Argumenty przydatne dla EasyOCR

- `--easyocr-langs` - lista jezykow rozdzielona przecinkami, domyslnie `pl,en`
- `--easyocr-device` - `auto`, `cpu` lub `cuda`
- `--easyocr-batch-size` - batch size inferencji (domyslnie `8`)
- `--easyocr-model-storage-dir` - katalog na lokalny cache wag EasyOCR

## EasyOCR i jezyk polski

- Domyslnie EasyOCR uruchamiany jest z jezykiem polskim (`pl`) oraz angielskim (`en`).
- Przy uruchomieniu na GPU (`--easyocr-device cuda`) wrapper automatycznie spadnie do CPU, gdy CUDA jest niedostepna.
- Wagi modelu sa pobierane raz i zapisywane lokalnie w `--easyocr-model-storage-dir`.
- Wrapper zaklada heterogeniczne rozmiary obrazow wejsciowych i przetwarza je bezpiecznie per-obraz (w chunkach logicznych wg `--easyocr-batch-size`).

## PaddleOCR i tryb bez detekcji dokumentu

- Wrapper PaddleOCR dziala w trybie recognition-only (`det=False`), bo benchmark operuje na wycietych slowach/liniach.
- Detektor na poziomie dokumentu/strony jest celowo wylaczony.
- W nowszych wersjach PaddleOCR backend opiera sie o `TextRecognition` (bez detekcji z definicji).

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


