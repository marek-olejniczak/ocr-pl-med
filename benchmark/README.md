# OCR Benchmark (HTR)

Lightweight benchmark for comparing OCR/HTR models on Polish data.

## Project contents

- `docker/`:
  - model services (FastAPI) + Dockerfile
- `orchestrator/`:
  - HTTP client and benchmark runner
- `src/`:
  - benchmark pipeline (calls the HTTP client)
- `wyniki/` - output CSV/JSON reports

## Quick start (Docker + HTTP)

1. Start the Docker service for the chosen model:

```bash
docker compose up -d <service>
```

2. Run the benchmark via HTTP:

```bash
python src/evaluate.py --model <model> --inference-mode http --limit 50
```

3. Results are stored in `wyniki/`.

### Services and ports

| Model (--model) | Docker service | Port | GPU |
| --- | --- | --- | --- |
| easyocr | easyocr | 8001 | CPU |
| trocr | trocr | 8002 | CPU |
| paddleocr | paddleocr | 8003 | CPU |
| parseq | parseq | 8004 | CPU |
| calamari | calamari | 8005 | CPU |
| rysocr | rysocr | 8006 | CPU |
| tesseract_pol | tesseract-pol | 8007 | CPU |
| surya | surya | 8008 | CPU |
| got_ocr | got-ocr | 8009 | CPU |
| qwen2_5_vl | qwen2_5_vl | 8010 | GPU |
| kraken | kraken | 8011 | GPU |
| glm_4v | glm_4v | 8012 | GPU |

### Example runs (HTTP)

1. Run benchmark on Tesseract:

```bash
python src/evaluate.py --model tesseract_pol --inference-mode http --limit 50
```

2. Run benchmark on RysOCR:

```bash
python src/evaluate.py --model rysocr --inference-mode http --limit 50
```

3. Run RysOCR with a larger batch size on GPU:

```bash
python src/evaluate.py --model rysocr --inference-mode http --rysocr-batch-size 4 --limit 50
```

4. Run RysOCR with batch size and mixed precision (AMP):

```bash
python src/evaluate.py --model rysocr --inference-mode http --rysocr-batch-size 4 --rysocr-use-amp --limit 50
```

5. Run benchmark on TrOCR (handwritten variant, lower VRAM):

```bash
python src/evaluate.py --model trocr --inference-mode http --limit 50
```

6. Run TrOCR with a larger batch size on GPU:

```bash
python src/evaluate.py --model trocr --inference-mode http --trocr-batch-size 8 --limit 50
```

7. Run TrOCR with batch size and mixed precision (AMP):

```bash
python src/evaluate.py --model trocr --inference-mode http --trocr-batch-size 8 --trocr-use-amp --limit 50
```

8. Run TrOCR base variant (heavier GPU usage):

```bash
python src/evaluate.py --model trocr --inference-mode http --trocr-variant base --trocr-batch-size 4 --limit 50
```

9. Run PaddleOCR PP-OCRv4 mobile (faster):

```bash
python src/evaluate.py --model paddleocr --inference-mode http --paddleocr-variant mobile --paddleocr-device cpu --limit 50
```

10. Run PaddleOCR PP-OCRv4 server (higher accuracy):

```bash
python src/evaluate.py --model paddleocr --inference-mode http --paddleocr-variant server --paddleocr-device cpu --limit 50
```

11. Run EasyOCR on GPU (Polish + English):

```bash
python src/evaluate.py --model easyocr --inference-mode http --easyocr-device cuda --easyocr-langs pl,en --limit 50
```

12. Run EasyOCR with local cache and batching:

```bash
python src/evaluate.py --model easyocr --inference-mode http --easyocr-device cuda --easyocr-batch-size 16 --easyocr-model-storage-dir modele/cache/easyocr --limit 50
```

13. Run PARSeq (docTR) with default preprocessing 32x128:

```bash
python src/evaluate.py --model parseq --inference-mode http --limit 50
```

14. Run PARSeq (docTR) with preprocessing 128x128:

```bash
python src/evaluate.py --model parseq --inference-mode http --parseq-input-size 128x128 --parseq-batch-size 8 --limit 50
```

15. Example with explicit cache directories:

```bash
python src/evaluate.py --model trocr --inference-mode http --trocr-cache-dir modele/cache/trocr --limit 50
python src/evaluate.py --model rysocr --inference-mode http --rysocr-cache-dir modele/cache/rysocr --limit 50
python src/evaluate.py --model paddleocr --inference-mode http --paddleocr-cache-dir modele/cache/paddlex --paddleocr-device cpu --limit 50
python src/evaluate.py --model parseq --inference-mode http --parseq-cache-dir modele/cache/parseq --limit 50
```

16. Run Calamari OCR (default model `idiotikon`, recommended for Polish diacritics):

```bash
python src/evaluate.py --model calamari --inference-mode http --limit 50
```

17. Run Calamari with another model and explicit cache:

```bash
python src/evaluate.py --model calamari --inference-mode http --calamari-model uw3-modern-english --calamari-cache-dir modele/cache/calamari --limit 50
```

18. Run GLM-4V (HTTP-only, GPU-first):

```bash
docker compose up -d glm_4v
python src/evaluate.py --model glm_4v --inference-mode http --limit 10
```

Note: GLM-4V-9B requires much more VRAM (~28-33 GB for BF16/FP16, ~10 GB for INT4).
On 4 GB GPUs the model OOMs, so it has not been tested on GPU in this repo yet.

Note: the first run of `RysOCR` may download large model weights (~2GB+).

## Offline mode for RysOCR

After downloading weights once, you can run offline:

```bash
python src/evaluate.py --model rysocr --rysocr-local-files-only --limit 50
```

If the cache is incomplete, run once without `--rysocr-local-files-only`.

## Useful arguments for RysOCR

- `--rysocr-adapter` - default `kacperwikiel/RysOCR`
- `--rysocr-base` - default `PaddlePaddle/PaddleOCR-VL`
- `--rysocr-device` - e.g. `cpu` or `cuda`
- `--rysocr-max-new-tokens`
- `--rysocr-prompt`
- `--rysocr-batch-size` - default `2`, increases throughput at the cost of VRAM
- `--rysocr-use-amp` - optional mixed precision on CUDA (disabled by default)
- `--rysocr-local-files-only`
- `--rysocr-cache-dir` - cache directory for RysOCR (default `modele/cache/rysocr`)

## Useful arguments for TrOCR

- `--trocr-variant` - `small` (default) or `base`; selects the default handwritten checkpoint
- `--trocr-model-id` - optional override; if omitted, uses `--trocr-variant`
- `--trocr-device` - e.g. `cpu` or `cuda`
- `--trocr-max-new-tokens`
- `--trocr-batch-size` - default `4`, increases throughput at the cost of VRAM
- `--trocr-use-amp` - optional mixed precision on CUDA (disabled by default)
- `--trocr-local-files-only`
- `--trocr-cache-dir` - cache directory for TrOCR (default `modele/cache/trocr`)

## Useful arguments for PaddleOCR

- `--paddleocr-variant` - `mobile` (default) or `server`
- `--paddleocr-rec-model-name` - optional override; if omitted, uses `--paddleocr-variant`
- `--paddleocr-lang` - default `pl`; mainly for legacy OCR fallback
- `--paddleocr-device` - `auto`, `cpu`, or `gpu`
- `--paddleocr-use-angle-cls` - enables angle classifier (CLS)
- `--paddleocr-rec-batch-size` - batch size for recognition (default `8`)
- `--paddleocr-cache-dir` - cache directory for PaddleX/PaddleOCR (default `modele/cache/paddlex`)

## Useful arguments for EasyOCR

- `--easyocr-langs` - comma-separated languages, default `pl,en`
- `--easyocr-device` - `auto`, `cpu`, or `cuda`
- `--easyocr-batch-size` - batch size (default `8`)
- `--easyocr-model-storage-dir` - local cache directory for EasyOCR weights

## Useful arguments for PARSeq (docTR)

- `--parseq-device` - `auto`, `cpu`, or `cuda`
- `--parseq-batch-size` - batch size (default `8`)
- `--parseq-cache-dir` - cache directory for PARSeq/docTR (default `modele/cache/parseq`)
- `--parseq-input-size` - resize preset: `32x128` (default) or `128x128`; if unsupported, falls back to model size
- `--parseq-use-amp` - optional mixed precision on CUDA (disabled by default)
- `--parseq-model-id` - optional `repo_id` for custom PARSeq checkpoint
- `--parseq-local-files-only` - offline mode, use local cache only
- `--parseq-lang` - preferred language (informational)

## Useful arguments for Calamari OCR

- `--calamari-model` - Calamari model name (default `idiotikon`)
- `--calamari-batch-size` - batch size (default `8`)
- `--calamari-cache-dir` - cache directory (default `modele/cache/calamari`)
- `--calamari-local-files-only` - offline mode, no downloads
- `--calamari-checkpoints` - optional list of `.ckpt` paths, comma-separated
- `--calamari-device` - preferred device (`auto`, `cpu`, `gpu`; depends on TF backend)

## Useful arguments for GLM-4V (HTTP)

- `--glm-4v-model-id` - default `zai-org/glm-4v-9b`
- `--glm-4v-device` - `auto`, `cpu`, or `cuda` (GPU-first; CPU only when explicit)
- `--glm-4v-dtype` - `auto`, `float16`, `bfloat16`, or `float32`
- `--glm-4v-max-new-tokens`
- `--glm-4v-prompt`
- `--glm-4v-cache-dir` - model cache directory
- `--glm-4v-local-files-only` - offline mode, local cache only
- `--http-timeout` - extend if the first load is slow

## EasyOCR and Polish language

- By default EasyOCR runs with Polish (`pl`) and English (`en`).
- On GPU (`--easyocr-device cuda`) the wrapper falls back to CPU if CUDA is unavailable.
- Weights are downloaded once and stored under `--easyocr-model-storage-dir`.
- The wrapper assumes heterogeneous input sizes and processes per-image in safe chunks based on `--easyocr-batch-size`.

## Model cache

- Default cache locations are `modele/cache/*` where supported.
- EasyOCR: `modele/cache/easyocr`.
- TrOCR: `modele/cache/trocr`.
- RysOCR: `modele/cache/rysocr`.
- PaddleOCR/PaddleX: `modele/cache/paddlex`.
- PARSeq/docTR: `modele/cache/parseq`.
- Calamari: `modele/cache/calamari`.
- GLM-4V: `modele/cache/glm_4v`.

## PaddleOCR detection-free mode

- The PaddleOCR wrapper runs in recognition-only mode (`det=False`) because the benchmark uses cropped words/lines.
- The document/page detector is intentionally disabled.
- In newer PaddleOCR versions the backend uses `TextRecognition` (no detection by design).

## TrOCR and Polish language

- The default TrOCR variant is not fine-tuned for Polish.
- Do not expect stable Polish diacritics.
- Treat TrOCR as a transformer OCR baseline, not a Polish-optimized model.

## PARSeq/docTR and Polish language

- PARSeq in docTR is a recognition-only model, without document detection.
- The default pretrained PARSeq has no hard language switch.
- Polish diacritic quality depends on the checkpoint charset; you can swap a checkpoint via `--parseq-model-id` if available.
- Preprocessing follows docTR flow and resizes to `32x128` (or `128x128`).

## Calamari OCR and Polish language

- Official Calamari models do not include a dedicated Polish checkpoint.
- The best practical multilingual option for Polish diacritics is `idiotikon`.
- For the highest quality, consider fine-tuning on a local corpus.

## Calamari environment note

- Calamari 2.x (compatible with release 2.1/2.2 models) requires a TensorFlow stack that works best on Python <= 3.11.
- This benchmark is currently maintained on Python 3.12 for other models; Calamari may need a separate env (e.g. conda with Python 3.11).

## RysOCR performance

- Default `--rysocr-batch-size 2` typically uses GPU better than single-image inference.
- On limited VRAM, start with batch 2 and increase to `4`, `8` for the best speed/VRAM tradeoff.
- `--rysocr-use-amp` speeds up CUDA inference but is ignored on CPU.

## Results

The pipeline writes:

- `wyniki/results_<timestamp>.csv`
- `wyniki/summary_<timestamp>.json`


