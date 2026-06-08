"""Predict CLI for Surya text-line detection (zero-shot only).

Surya detects text lines natively (modified EfficientViT segformer). Detection
finetuning is not publicly available, so this is predict-only; `train` exits
unsupported. Honors the benchmark contract: predict writes COCO results +
meta.json, so the shared evaluator scores it like any other model.

Usage:
    python cli.py predict --weights surya --coco instances_test.json \
        --images-root <root> --out results/predictions/<exp_id>
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

LINE_CATEGORY_ID = 1


def xyxy_to_coco(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def detections_to_coco(bboxes_conf, image_id):
    """[(xyxy, confidence), ...] -> COCO results records."""
    return [{"image_id": int(image_id), "category_id": LINE_CATEGORY_ID,
             "bbox": xyxy_to_coco(b), "score": float(c)}
            for b, c in bboxes_conf]


def speed_stats(speeds_ms):
    if not speeds_ms:
        return {"ms_per_image_mean": 0.0, "ms_per_image_median": 0.0}
    return {"ms_per_image_mean": float(statistics.fmean(speeds_ms)),
            "ms_per_image_median": float(statistics.median(speeds_ms))}


def cmd_train(args):
    sys.exit("surya: text-line detection finetuning is not publicly available "
             "(predict-only / zero-shot model)")


def cmd_predict(args):
    from PIL import Image
    from surya.detection import DetectionPredictor

    det = DetectionPredictor()
    coco = json.loads(Path(args.coco).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions, speeds = [], []
    for img in coco["images"]:
        path = Path(args.images_root) / img["file_name"]
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        t0 = time.perf_counter()
        result = det([image])[0]
        speeds.append((time.perf_counter() - t0) * 1000.0)
        bboxes_conf = [
            (b.bbox, b.confidence if b.confidence is not None else 1.0)
            for b in result.bboxes
            if (b.confidence if b.confidence is not None else 1.0) >= args.conf
        ]
        predictions.extend(detections_to_coco(bboxes_conf, img["id"]))

    (out / "predictions.json").write_text(json.dumps(predictions))

    import surya
    meta = {"model": "surya-detection", "weights": str(args.weights),
            "device": str(args.device), "conf": args.conf,
            "n_images": len(coco["images"]),
            "n_predictions": len(predictions),
            **speed_stats(speeds),
            "versions": {"surya": getattr(surya, "__version__", "unknown"),
                         "python": platform.python_version()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(predictions)} predictions for {len(coco['images'])} images")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--weights", default="surya")   # ignored; surya loads its own
    p.add_argument("--coco", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.0)   # keep all; eval thresholds
    p.add_argument("--imgsz", type=int, default=640)    # accepted, surya self-sizes
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
