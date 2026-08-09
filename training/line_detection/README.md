# Line Detection — fine-tuning

Miejsce na fine-tuning modeli detekcji linii (YOLO, RT-DETR, Faster R-CNN, Kraken BLLA).

## Osoba odpowiedzialna

[TBD]

## Wzorzec

Taki sam jak `training/ocr/`: Dockerfile + cli.py (train/predict) per model.

## Istniejący kod

Kod treningowy dla modeli ultralytics/detectron2/kraken znajduje się w `line_benchmark/docker/`.
Tam też są już zdefiniowane pipeline'y treningowe (patrz: `line_benchmark/orchestrator/`).
