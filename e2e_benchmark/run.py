"""Run the full pipeline over a GT dataset and score it end-to-end.

    python run.py --gt dataset/gt.jsonl \\
        --weights ../best_iou_median.pt --ocr tesseract-pol trocr

Scores every detector x OCR combination, writes results/<name>_<timestamp>/
with per-combo raw_predictions.csv + summary_metrics.json and
overleaf_tables.tex at the run root. No geometric preprocessing: GT boxes
are in original image coordinates and deskew would invalidate them.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "app"))

import evaluate  # noqa: E402
import gt as gt_mod  # noqa: E402
import latex  # noqa: E402
import pipeline  # noqa: E402
import preprocess  # noqa: E402
import services  # noqa: E402
from detectors import UltralyticsDetector  # noqa: E402
from ocr_client import OCRClient  # noqa: E402


def _resolve_ocr(name, registry):
    if name in registry:
        return name, registry[name]
    if name.startswith("http"):
        return name.replace("://", "_").replace("/", "_").rstrip("_"), name
    raise SystemExit(f"unknown OCR service '{name}' - one of: "
                     + ", ".join(sorted(registry)) + " or a URL")


def _write_raw(combo_dir, pages, page_evals):
    with open(combo_dir / "raw_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "gt_line", "pred_line", "iou",
                    "ground_truth", "prediction"])
        for page, pe in zip(pages, page_evals):
            matched_gt = {g: (p, iou) for g, p, iou in pe["matches"]}
            matched_pred = {p for _, p, _ in pe["matches"]}
            for g, gt_text in enumerate(pe["gt_texts"]):
                if g in matched_gt:
                    p, iou = matched_gt[g]
                    w.writerow([page.image_path.name, g, p, f"{iou:.3f}",
                                gt_text, pe["pred_texts"][p]])
                else:
                    w.writerow([page.image_path.name, g, "", "",
                                gt_text, ""])
            for p, pred_text in enumerate(pe["pred_texts"]):
                if p not in matched_pred:
                    w.writerow([page.image_path.name, "", p, "",
                                "", pred_text])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="GT JSONL (see gt.py)")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="one or more ultralytics checkpoints")
    ap.add_argument("--ocr", nargs="+", required=True,
                    help="service names from benchmark/docker-compose.yml "
                         "or full URLs")
    ap.add_argument("--name", default="e2e")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-photometric", action="store_true")
    ap.add_argument("--out", default=str(_HERE / "results"))
    args = ap.parse_args()

    pages = gt_mod.load(args.gt)
    registry = services.load_registry()
    photometric = None if args.no_photometric else preprocess.photometric

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"{args.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.yaml").write_text(yaml.safe_dump({
        "gt": str(Path(args.gt).resolve()), "pages": len(pages),
        "weights": args.weights, "ocr": args.ocr, "iou": args.iou,
        "imgsz": args.imgsz, "conf": args.conf,
        "photometric": photometric is not None}, sort_keys=False))

    table_rows = []
    for weights in args.weights:
        det_id = Path(weights).stem
        detector = UltralyticsDetector(weights, imgsz=args.imgsz,
                                       conf=args.conf, device=args.device)
        for ocr_arg in args.ocr:
            ocr_id, url = _resolve_ocr(ocr_arg, registry)
            ocr = OCRClient(url)
            try:
                ocr.health()
                ocr.load()
            except requests.RequestException:
                raise SystemExit(
                    f"{ocr_id} is not answering at {url} - start it first:\n"
                    f"cd benchmark && docker compose up -d {ocr_id}")

            print(f"[{det_id} x {ocr_id}] {len(pages)} pages")
            page_evals, started = [], time.perf_counter()
            for page in pages:
                img = Image.open(page.image_path).convert("RGB")
                results, _ = pipeline.run(img, detector, ocr,
                                          preprocess=photometric)
                page_evals.append(
                    evaluate.page_eval(page.lines, results, args.iou))
            per_page = (time.perf_counter() - started) / max(len(pages), 1)

            summary = evaluate.summarize(page_evals, per_page)
            combo_dir = run_dir / f"{det_id}__{ocr_id}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            (combo_dir / "summary_metrics.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2))
            _write_raw(combo_dir, pages, page_evals)
            table_rows.append((det_id, ocr_id, summary))

            doc = summary["metrics"]["document"]
            det = summary["metrics"]["detection"]
            print(f"  CER {doc['cer']:.4f}  WER {doc['wer']:.4f}  "
                  f"DetF1 {det['f1']:.4f}  {per_page:.2f}s/page")

    dataset_id = Path(args.gt).resolve().parent.name
    (run_dir / "overleaf_tables.tex").write_text(
        latex.e2e_table(table_rows, dataset_id))
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
