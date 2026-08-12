# End-to-end benchmark

Scores the whole pipeline (line detection + OCR) against page-level ground
truth, for every detector x OCR combination at once.

## Ground truth

JSONL, one record per page, boxes in COCO xywh on the original image:

```json
{"image": "pages/scan_01.png",
 "lines": [{"bbox": [341, 5, 1822, 115], "text": "Rp. Amoksycylina 500mg"}]}
```

`gt.from_coco()` converts a COCO export with per-annotation transcriptions
(plain `text` field or CVAT-style `attributes.text`).

## Running

```bash
# OCR services under test (from benchmark/)
docker compose up -d tesseract-pol trocr

# from e2e_benchmark/
python run.py --gt dataset/gt.jsonl \
    --weights ../best_iou_median.pt --ocr tesseract-pol trocr
```

Results land in `results/<name>_<timestamp>/` with `summary_metrics.json`
and `raw_predictions.csv` per combination, plus `overleaf_tables.tex` ready
to paste into the thesis. Flags: `--iou` (matching threshold, default 0.5),
`--conf`, `--imgsz`, `--device`, `--name`, `--no-photometric`.

## Metrics

Three levels in each summary:

- `document`: CER/WER/EMA on full-page text in reading order, so exactly
  what the pipeline hands to the user
- `detection`: precision/recall/F1 of line boxes at the IoU threshold
- `matched_lines`: CER/WER on IoU-matched line pairs only

The split matters when reading results: a missed line hurts `document` and
`detection` but not `matched_lines`, while a bad OCR read hurts the text
metrics and leaves `detection` alone. Text metrics come from the OCR
benchmark evaluator, box matching from line_benchmark, so the numbers are
comparable across all three benchmarks.

Geometric preprocessing is off here on purpose: GT boxes are in original
image coordinates and deskew would move the ink from under them. The
photometric stage runs like in the app (detector input only).
