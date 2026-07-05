# Line detection benchmark

A tool for comparing models that detect lines of text in document images
(ultimately Polish handwritten medical documents). For every model we measure
quality, speed, and resource use under the same conditions, so the comparison
is fair.

## How it works

Each model lives in its own Docker container and exposes the same small API:

- `train` fits the model on our data,
- `predict` runs the model on images and writes the detected lines in COCO
  format.

Because every model writes its output in one format, a single shared
`evaluate.py` computes the metrics identically for all of them. The model does
not know how the metrics are computed, and the evaluator does not know which
model produced the output. That keeps the comparison honest and lets us add a
new model without touching the rest: you only need a Dockerfile, a `cli.py` that
honors the contract, and one entry in the config.

## Directory layout

```
line_benchmark/
  data_prep/             data preparation
    build_dataset.py     split into train / val / test
    preprocess_dataset.py  photometric image variant
    converters/          COCO -> YOLO and COCO -> PAGE XML
  docker/                one folder per model family (Dockerfile + cli.py)
    ultralytics/         YOLOv8, YOLO11, RT-DETR
    detectron2/          Faster R-CNN
    surya/               Surya (predict only)
    kraken/              Kraken (HTR line segmentation)
  evaluation/            metrics + evaluate.py (shared by all models)
  training_diagnostics/  training diagnostics and validation metrics (ultralytics)
  orchestrator/          run_experiments.py + experiments.yaml (full matrix)
  results/               checkpoints, predictions, metrics, summary.csv
```

## Models

| Model | Family | Modes |
|---|---|---|
| YOLOv8, YOLO11 | single-stage CNN detection | finetuning |
| RT-DETR | transformer detection | finetuning |
| Faster R-CNN | two-stage CNN detection | finetuning |
| Surya | document layout model | zero-shot |
| Kraken BLLA | HTR line segmentation | zero-shot + finetuning |

Zero-shot means the model detects lines without being trained on our data.
Finetuning means the model is further trained on our dataset.

## What we measure

Quality:

- mAP (the standard COCO detection metric),
- precision, recall, F1,
- line metrics: how many lines are missed, cut into pieces (split), merged into
  one (merge), and how tightly the boxes match the truth (IoU),
- ECE, which tells whether the model's confidence matches its accuracy.

Speed: time per image in milliseconds.

Resources: peak GPU memory and peak RAM.

The line metrics matter more to us than mAP alone. A missed line is text lost
for good in downstream OCR, and splitting or merging lines breaks the
recognition context. mAP does not show this well, which is why we added our own
metrics.

## How to run

Run all commands from the `line_benchmark/` directory.

1. Split the data into train / val / test:

```
python data_prep/build_dataset.py --coco <coco_file.json> --out-dir dataset/annotations
```

2. Build the YOLO layout (copies images and writes data.yaml):

```
python data_prep/converters/to_yolo.py --annotations-dir dataset/annotations \
  --images-root <image_dir> --out-dir dataset/yolo_raw --copy
```

3. Build a model image:

```
docker compose build ultralytics
```

4. A single run (train, predict, evaluate):

```
docker compose run --rm ultralytics python docker/ultralytics/cli.py train \
  --weights yolov8n.pt --data dataset/yolo_raw/data.yaml \
  --out results/checkpoints/yolov8_ft-raw --line-val --diagnostics --wandb

docker compose run --rm ultralytics python docker/ultralytics/cli.py predict \
  --weights results/checkpoints/yolov8_ft-raw/train/weights/best.pt \
  --coco dataset/annotations/instances_test.json --images-root <images> \
  --out results/predictions/yolov8_ft-raw_eval-raw

python evaluation/evaluate.py --gt dataset/annotations/instances_test.json \
  --pred results/predictions/yolov8_ft-raw_eval-raw/predictions.json \
  --exp-id yolov8_ft-raw_eval-raw --out-dir results
```

5. The whole experiment matrix at once:

```
python orchestrator/run_experiments.py --dry-run    # preview, runs nothing
python orchestrator/run_experiments.py              # execute
```

The orchestrator is idempotent: it skips stages that are already done, so after
a failure you can run it again and it fills in only the gaps. Filters:
`--only <name_fragment>`, `--stage train|predict|eval`.

## Choosing the learning rate

Before training it helps to run an LR range test (Smith's method), which shows
at what LR the loss drops fastest and where it starts to diverge:

```
docker compose run --rm ultralytics python docker/ultralytics/cli.py lr-find \
  --weights yolov8n.pt --data dataset/yolo_raw/data.yaml --out results/lr_find/yolov8n
```

The result (`lr_find.json`) holds the suggested LR. Per-model LR is set in
`experiments.yaml`.

## Results

Each experiment appends one row to `results/summary.csv` with the full set of
numbers (quality, speed, resources) and writes a full `results/metrics/<id>.json`.
Training also logs curves and prediction previews to Weights & Biases.

## Notes

- Preprocessing in the benchmark is photometric only (CLAHE, grid removal, ink
  separation, binarization). We do not rotate or crop the image, because that
  would shift the annotations relative to the pixels. Deskew and border padding
  are left for the production version, where there are no annotations to break.
- The benchmark is dataset agnostic. You point it at COCO files in the config,
  so when a larger dataset arrives you swap two paths.
- The raw and prep variants share the same annotations (preprocessing preserves
  geometry), so they differ only in the set of images.
