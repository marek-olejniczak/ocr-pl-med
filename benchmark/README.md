# OCR Benchmark (HTR)

Lightweight benchmark for comparing OCR/HTR models on Polish data.

## Architecture

The benchmark runs models as **Docker services** (FastAPI) and communicates via HTTP.
Orchestration is handled by **AutoRunner** (`autorunner.py`), which reads an
`experiments.yaml` config, starts/stops services, runs inference, computes
metrics, and saves results.

```
experiments.yaml → AutoRunner → docker compose up → HTTP inference → metrics → wyniki/
```

Model wrappers in `modele/` are **local (in-process)** variants — a legacy approach.
Most models now run only via Docker HTTP; local wrappers are kept for reference
and may lack working dependencies.

## Quick start (AutoRunner)

1. Build Docker images for all models:

```bash
python autorunner.py --build
```

2. Enable desired models and datasets in `experiments.yaml`. Then run:

```bash
python autorunner.py
```

AutoRunner will iterate over enabled models in sequence:
start service → health check → load model → benchmark each dataset → stop service.

Results land in `wyniki/<experiment-name>_<timestamp>/`.

## Project contents

| Path | Purpose |
|---|---|
| `autorunner.py` | Main entry point — orchestrates full benchmark runs |
| `experiments.yaml` | Configuration: models, datasets, settings |
| `docker/` | FastAPI model services + Dockerfiles (one per model) |
| `docker-compose.yml` | Service definitions (ports, GPU, volumes, health checks) |
| `orchestrator/` | HTTP client (`client.py`) + benchmark runner (`benchmark.py`) |
| `src/` | Data loading (`data_generator.py`), metrics (`metrics.py`), experiment results (`experiment_results.py`) |
| `modele/` | **Legacy** local wrappers (in-process, mostly unused) |
| `wyniki/` | Output: per-run directories with raw predictions + metrics |
| `dane/` | Datasets (labels CSV + images) |
| `wandb/` | Weights & Biases logs (optional) |

## experiments.yaml reference

```yaml
experiment:
  name: "ocr-benchmark-autorun-test"          # run name, used for output dir
  project_name: "ocr-benchmark"               # W&B project
  wandb_entity: "ocr-pl-med"                  # W&B entity
  tags: ["autorun", "http"]                   # W&B tags
  hardware_note: "Nvidia GTX 1650, 4GB VRAM"  # informational

settings:
  output_dir: "wyniki"                         # output root
  export_overleaf_table: false                 # generate LaTeX tables
  auto_start_services: true                    # start/stop Docker per model
  stop_service_after_run: true
  sequential: true                             # run models one at a time
  warmup_samples: 1
  cooldown_seconds: 5                          # pause between models
  http_timeout_default: 120
  wandb_log: false                             # log metrics to W&B

datasets:
  - id: "testset_800"
    enabled: true
    labels_csv: "dane/testset/labels.csv"
    images_dir: "dane/testset/images"
    limit: null                                # null = all samples
    single_words: false                        # word-level or line-level metrics

models:
  - id: "tesseract_pol"
    service_name: "tesseract-pol"
    base_url: "http://localhost:8007"
    enabled: true
    timeout: 120
    options:
      language: "pol"
      psm: 7
      oem: 1
```

Models and datasets can be toggled on/off with `enabled: true/false`.

## Building images

Build every enabled service, or build a single one:

```bash
python autorunner.py --build
docker compose build <service-name>
```

## Services and ports

| Model ID | Docker service | Port | GPU |
|---|---|---|---|
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

Start a service manually (bypassing AutoRunner):

```bash
docker compose up -d <service>
```

## Results

Output directory structure:

```
wyniki/
  <experiment-name>_<timestamp>/
    meta.yaml                              # full config snapshot
    overleaf_tables.tex                    # LaTeX tables (optional)
    <model_id>/
      <dataset_id>/
        raw_predictions.csv                # per-sample predictions
        summary_metrics.json               # aggregated metrics + timing
```

Metrics computed: exact match accuracy (EMA), near-perfect match (NPM, ≤1 char
error), character error rate (CER), word error rate (WER), mean Levenshtein
distance.

## Model cache

Default cache directories (bind-mounted into Docker containers):

| Model | Cache path |
|---|---|
| EasyOCR | `modele/cache/easyocr` |
| TrOCR | `modele/cache/trocr` |
| RysOCR | `modele/cache/rysocr` |
| PaddleOCR/PaddleX | `modele/cache/paddlex` |
| PARSeq/docTR | `modele/cache/parseq` |
| Calamari | `modele/cache/calamari` |
| GLM-4V | `modele/cache/glm_4v` |
| Surya | `modele/cache/surya` |
| GOT-OCR | `modele/cache/got_ocr` |
| Qwen2.5-VL | `modele/cache/qwen2_5_vl` |
| Kraken | `modele/cache/kraken` |

---

## Legacy: local wrappers (`src/evaluate.py`)

The older approach runs models in-process via `modele/*_wrapper.py`:

```bash
python src/evaluate.py --model tesseract_pol --inference-mode local --limit 50
python src/evaluate.py --model rysocr --inference-mode local --rysocr-batch-size 2 --limit 50
python src/evaluate.py --model trocr --inference-mode local --limit 50
```

Only **tesseract_pol**, **rysocr**, and **trocr** have working local wrappers.
The remaining local wrappers (paddleocr, easyocr, parseq, calamari) require
additional dependencies not installed by default.

Models **surya**, **got_ocr**, **qwen2_5_vl**, **glm_4v**, and **kraken** are
**HTTP-only** — they have no local wrapper and must be run via Docker.

Local mode options (per model) remain documented in `src/evaluate.py` parser
arguments and are not duplicated here.

### Calamari environment note

Calamari 2.x requires TensorFlow on Python ≤ 3.11. This project is maintained
on Python 3.12+; Calamari may need a separate conda env.

### TrOCR and Polish language

The default TrOCR variant is not fine-tuned for Polish. Do not expect stable
Polish diacritics. Treat TrOCR as a transformer OCR baseline.

### PARSeq/docTR and Polish language

PARSeq is recognition-only, no document detection. Polish diacritic quality
depends on the checkpoint charset. Use `--parseq-model-id` to swap checkpoints.

### PaddleOCR detection-free mode

The PaddleOCR wrapper runs in recognition-only mode (`det=False`). The
document/page detector is intentionally disabled.

## W&B integration

Set `wandb_log: true` in `experiments.yaml` and export:

```bash
export WANDB_API_KEY=your_key
```
