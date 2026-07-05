# OCR pipeline app

Working demo of the full OCR pipeline for this project. You upload a document
(image or PDF), the app finds the text lines and sends each one to an OCR
model, and you get the recognized text back.

```
upload -> geometric prep -> photometric prep -> line detection
                    |                                  |
                    +-----------> crop lines <---------+
                                      |
                            per-line OCR -> text
```

## How it works

Preprocessing is split in two stages with different scopes.

The geometric stage (deskew) is global: each uploaded page (PDFs are
rasterized page by page) gets straightened first, and that straightened image
is the base for everything downstream. Skewed pages hurt both models at once,
the detector duplicates and splits lines and the OCR gets tilted crops. The
benchmark skips this stage because ground-truth boxes have to stay valid
there; the app has no such constraint.

The photometric stage (CLAHE, grid removal, ink separation) feeds the line
detector only. It matters because the detector was trained mostly on black
ink, so things like blue handwriting land far from the training distribution
without it. The detector runs on that image, but the line crops for OCR are
cut from the base image. The coordinates transfer 1:1 because this stage
touches only pixel values, never geometry. Crops go to the OCR service in
reading order and the texts come back joined into the final result.

The pipeline logic lives in `pipeline.py` and has no UI in it, gradio is just
a thin layer on top. If we ever want a nicer front (FastAPI + React), the
pipeline moves over unchanged.

## Running

```bash
# deps (once)
pip install -r requirements.txt

# an OCR service (from benchmark/)
docker compose up -d tesseract-pol

# the app (from app/)
python ui.py --weights ../best_iou_median.pt --ocr-url http://localhost:8007
```

Open http://localhost:7860, drop in a file, press Run. Test inputs: anything
from `line_benchmark/dataset/yolo_raw/images/val/` or `test/`, or any PDF.

Useful flags: `--conf` (detection threshold, default 0.25 like the benchmark),
`--device` (cpu / cuda / mps), `--no-geometric` (skip deskew),
`--no-photometric` (feed the detector the base image), `--port`. The collapsed
"Detector input" section in the UI shows the preprocessed pages, so when lines
are missing or split, open it, it usually explains why.

## Swapping models

Both models are placeholders until the benchmarks pick winners, and both are
swappable without touching the pipeline:

- Line detector: `--weights` takes any ultralytics checkpoint (YOLOv8, YOLO11,
  RT-DETR). If the line benchmark winner ends up being detectron2 or kraken,
  it gets a small HTTP wrapper in its existing container and a second adapter
  in `detectors.py`.
- OCR: every service in `benchmark/` exposes the same API, so switching the
  model means switching the URL. Ports from `benchmark/docker-compose.yml`:

  | service | port | | service | port |
  |---|---|---|---|---|
  | easyocr | 8001 | | tesseract-pol | 8007 |
  | trocr | 8002 | | surya | 8008 |
  | paddleocr | 8003 | | got-ocr | 8009 |
  | parseq | 8004 | | qwen2_5_vl | 8010 |
  | calamari | 8005 | | kraken | 8011 |
  | rysocr | 8006 | | glm_4v | 8012 |

tesseract-pol is the current default because it is the lightest one, but it
can only read print, on handwriting it produces garbage. That is the model,
not the pipeline.
