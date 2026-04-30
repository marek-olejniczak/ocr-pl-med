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
  - `parseq_wrapper.py` - wrapper dla PARSeq (docTR, recognition-only)
  - `calamari_wrapper.py` - wrapper dla Calamari OCR (line-based, ensemble checkpointow)
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

14. Uruchom PARSeq (docTR) z domyslnym preprocessingiem 32x128:

```bash
python src/evaluate.py --model parseq --limit 50
```

15. Uruchom PARSeq (docTR) z alternatywnym preprocessingiem 128x128:

```bash
python src/evaluate.py --model parseq --parseq-input-size 128x128 --parseq-batch-size 8 --limit 50
```

16. Przykladowe uruchomienie z jawnie ustawionymi katalogami cache modeli:

```bash
python src/evaluate.py --model trocr --trocr-cache-dir modele/cache/trocr --limit 50
python src/evaluate.py --model rysocr --rysocr-cache-dir modele/cache/rysocr --limit 50
python src/evaluate.py --model paddleocr --paddleocr-cache-dir modele/cache/paddlex --paddleocr-device cpu --limit 50
python src/evaluate.py --model parseq --parseq-cache-dir modele/cache/parseq --limit 50
```

17. Uruchom Calamari OCR (domyslnie model `idiotikon`, rekomendowany dla polskich diakrytykow):

```bash
python src/evaluate.py --model calamari --limit 50
```

18. Uruchom Calamari z innym modelem i jawnie ustawionym cache:

```bash
python src/evaluate.py --model calamari --calamari-model uw3-modern-english --calamari-cache-dir modele/cache/calamari --limit 50
```

19. Uruchom GLM-4V (HTTP-only, GPU-first):

```bash
docker compose up -d glm_4v
python src/evaluate.py --model glm_4v --inference-mode http --limit 10
```

Uwaga: GLM-4V-9B wymaga znacznie wiecej VRAM (ok. 28-33 GB dla BF16/FP16, ~10 GB dla INT4).
Na GPU 4 GB model nie startuje (OOM), dlatego w tym repo nie zostal jeszcze przetestowany na GPU.

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
- `--rysocr-cache-dir` - katalog cache wag/modeli dla RysOCR (domyslnie `modele/cache/rysocr`)

## Argumenty przydatne dla TrOCR

- `--trocr-variant` - `small` (domyslnie) lub `base`; wybiera domyslny checkpoint handwritten
- `--trocr-model-id` - opcjonalne nadpisanie checkpointu; gdy pominiete, bierze model z `--trocr-variant`
- `--trocr-device` - np. `cpu` lub `cuda`
- `--trocr-max-new-tokens`
- `--trocr-batch-size` - domyslnie `4`, zwieksza throughput kosztem VRAM
- `--trocr-use-amp` - opcjonalne mixed precision na CUDA (domyslnie wylaczone)
- `--trocr-local-files-only`
- `--trocr-cache-dir` - katalog cache wag/modeli dla TrOCR (domyslnie `modele/cache/trocr`)

## Argumenty przydatne dla PaddleOCR

- `--paddleocr-variant` - `mobile` (domyslnie) lub `server`
- `--paddleocr-rec-model-name` - opcjonalne nadpisanie modelu rec; gdy pominiete, bierze model z `--paddleocr-variant`
- `--paddleocr-lang` - domyslnie `pl`; istotne glownie dla fallbacku legacy OCR
- `--paddleocr-device` - `auto`, `cpu` lub `gpu`
- `--paddleocr-use-angle-cls` - wlacza klasyfikator kata (CLS)
- `--paddleocr-rec-batch-size` - batch size dla rec (domyslnie `8`)
- `--paddleocr-cache-dir` - katalog cache modeli PaddleX/PaddleOCR (domyslnie `modele/cache/paddlex`)

## Argumenty przydatne dla EasyOCR

- `--easyocr-langs` - lista jezykow rozdzielona przecinkami, domyslnie `pl,en`
- `--easyocr-device` - `auto`, `cpu` lub `cuda`
- `--easyocr-batch-size` - batch size inferencji (domyslnie `8`)
- `--easyocr-model-storage-dir` - katalog na lokalny cache wag EasyOCR

## Argumenty przydatne dla PARSeq (docTR)

- `--parseq-device` - `auto`, `cpu` lub `cuda`
- `--parseq-batch-size` - batch size inferencji (domyslnie `8`)
- `--parseq-cache-dir` - katalog na lokalny cache wag PARSeq/docTR (domyslnie `modele/cache/parseq`)
- `--parseq-input-size` - preset preprocessingu resize: `32x128` (domyslnie) albo `128x128`; gdy checkpoint nie wspiera wybranego rozmiaru, wrapper zrobi fallback do rozmiaru modelu
- `--parseq-use-amp` - opcjonalne mixed precision na CUDA (domyslnie wylaczone)
- `--parseq-model-id` - opcjonalny `repo_id` Hugging Face dla niestandardowego checkpointu PARSeq
- `--parseq-local-files-only` - tryb offline, laduj tylko z lokalnego cache
- `--parseq-lang` - preferowany jezyk (informacyjnie)

## Argumenty przydatne dla Calamari OCR

- `--calamari-model` - nazwa modelu Calamari (domyslnie `idiotikon`)
- `--calamari-batch-size` - batch size inferencji (domyslnie `8`)
- `--calamari-cache-dir` - katalog na lokalny cache modeli (domyslnie `modele/cache/calamari`)
- `--calamari-local-files-only` - tryb offline, bez pobierania modeli
- `--calamari-checkpoints` - opcjonalna lista sciezek do `.ckpt` rozdzielona przecinkami
- `--calamari-device` - preferowane urzadzenie (`auto`, `cpu`, `gpu`; zalezne od backendu TF)

## Argumenty przydatne dla GLM-4V (HTTP)

- `--glm-4v-model-id` - domyslnie `zai-org/glm-4v-9b`
- `--glm-4v-device` - `auto`, `cpu` lub `cuda` (GPU-first; cpu tylko jawnie)
- `--glm-4v-dtype` - `auto`, `float16`, `bfloat16` lub `float32`
- `--glm-4v-max-new-tokens`
- `--glm-4v-prompt`
- `--glm-4v-cache-dir` - katalog cache modelu
- `--glm-4v-local-files-only` - tryb offline, tylko lokalny cache
- `--http-timeout` - wydluz, jesli pierwsze ladowanie trwa dluzej

## EasyOCR i jezyk polski

- Domyslnie EasyOCR uruchamiany jest z jezykiem polskim (`pl`) oraz angielskim (`en`).
- Przy uruchomieniu na GPU (`--easyocr-device cuda`) wrapper automatycznie spadnie do CPU, gdy CUDA jest niedostepna.
- Wagi modelu sa pobierane raz i zapisywane lokalnie w `--easyocr-model-storage-dir`.
- Wrapper zaklada heterogeniczne rozmiary obrazow wejsciowych i przetwarza je bezpiecznie per-obraz (w chunkach logicznych wg `--easyocr-batch-size`).

## Cache modeli

- Domyslnie cache wag modeli trafia do `modele/cache/*` dla wrapperow, ktore na to pozwalaja.
- EasyOCR: `modele/cache/easyocr`.
- TrOCR: `modele/cache/trocr`.
- RysOCR: `modele/cache/rysocr`.
- PaddleOCR/PaddleX: `modele/cache/paddlex`.
- PARSeq/docTR: `modele/cache/parseq`.
- Calamari: `modele/cache/calamari`.
- GLM-4V: `modele/cache/glm_4v`.

## PaddleOCR i tryb bez detekcji dokumentu

- Wrapper PaddleOCR dziala w trybie recognition-only (`det=False`), bo benchmark operuje na wycietych slowach/liniach.
- Detektor na poziomie dokumentu/strony jest celowo wylaczony.
- W nowszych wersjach PaddleOCR backend opiera sie o `TextRecognition` (bez detekcji z definicji).

## TrOCR i jezyk polski

- Domyslny wariant TrOCR w benchmarku nie jest dotrenowany stricte na jezyku polskim.
- Nie nalezy oczekiwac stabilnego rozpoznawania polskich diakrytykow (np. ą, ć, ę, ł, ń, ó, ś, ź, ż).
- TrOCR traktuj jako punkt odniesienia dla transformera OCR, a nie model zoptymalizowany pod polskie dane.

## PARSeq/docTR i jezyk polski

- PARSeq w docTR jest modelem rozpoznawania tekstu (recognition-only), bez detekcji dokumentu.
- Domyslnie uzywany jest pretrained PARSeq, ktory nie ma twardego przelacznika jezyka jak niektore inne biblioteki OCR.
- Dla jezyka polskiego jakosc diakrytykow (np. ą, ć, ę, ł, ń, ó, ś, ź, ż) zalezy od charsetu checkpointu; mozna podmienic checkpoint przez `--parseq-model-id`, jesli dostepny jest wariant lepiej wspierajacy PL.
- Preprocessing wrappera korzysta z flow docTR i resize do `32x128` (lub opcjonalnie `128x128`).

## Calamari OCR i jezyk polski

- W oficjalnych modelach Calamari nie ma dedykowanego checkpointu stricte PL.
- Najlepsza praktyczna opcja wielojezyczna pod polskie znaki to `idiotikon` (szeroki zestaw znakow lacinskich i diakrytykow).
- Dla najwyzszej jakosci PL warto rozwazyc pozniejszy fine-tuning na lokalnym korpusie.

## Uwaga srodowiskowa dla Calamari

- Calamari 2.x (kompatybilne z aktualnymi modelami release 2.1/2.2) wymaga stosu TensorFlow, ktory w praktyce najlepiej dziala na Python <= 3.11.
- W tym repo benchmark jest aktualnie utrzymywany na Python 3.12 dla pozostalych modeli; dla Calamari moze byc potrzebne osobne srodowisko (np. conda z Python 3.11).

## Wydajnosc RysOCR

- Domyslnie RysOCR dziala z `--rysocr-batch-size 2`, co zwykle lepiej wykorzystuje GPU niz inferencja pojedyncza.
- Przy ograniczonej pamieci GPU startuj od batcha 2 i stopniowo zwiekszaj (`4`, `8`) do momentu najlepszego kompromisu szybkosc/VRAM.
- `--rysocr-use-amp` przyspiesza inferencje na CUDA, ale na CPU jest ignorowane.

## Wyniki

Pipeline zapisuje:

- `wyniki/results_<timestamp>.csv`
- `wyniki/summary_<timestamp>.json`


