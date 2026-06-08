"""Train (ketos segtrain) / predict (blla baseline segmentation) CLI for Kraken.

Kraken is the most version-sensitive benchmark model: the segmentation API
changed across 4.x / 5.x / 7.x. This targets kraken ~5.2 (kraken.blla.segment +
vgsl.TorchVGSLModel; Segmentation.lines[*].boundary polygons). VERIFY the
segment call and the ketos segtrain flags against the installed version when
building the image - they are the most likely thing to need adjustment.

Kraken outputs polygons; we take each line boundary's bounding rectangle so the
shared evaluator sees COCO bboxes like every other model. Kraken gives no
per-line detection score, so score = 1.0 (note: ECE is degenerate for Kraken).
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

LINE_CATEGORY_ID = 1


def polygon_to_bbox(points):
    """Polygon [(x,y), ...] -> COCO xywh of its bounding rectangle."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    return [float(x0), float(y0), float(max(xs) - x0), float(max(ys) - y0)]


def lines_to_coco(boundaries, image_id):
    """List of boundary polygons -> COCO results (score 1.0; kraken has none)."""
    return [{"image_id": int(image_id), "category_id": LINE_CATEGORY_ID,
             "bbox": polygon_to_bbox(b), "score": 1.0}
            for b in boundaries if b]


def speed_stats(speeds_ms):
    if not speeds_ms:
        return {"ms_per_image_mean": 0.0, "ms_per_image_median": 0.0}
    return {"ms_per_image_mean": float(statistics.fmean(speeds_ms)),
            "ms_per_image_median": float(statistics.median(speeds_ms))}


def cmd_train(args):
    # ketos segtrain is the stable interface (CLI), so we shell out rather than
    # bind to a churning Python API. PAGE XML produced by to_pagexml.py.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xmls = sorted(str(p) for p in Path(args.data).glob("*.xml"))
    if not xmls:
        raise SystemExit(f"no PAGE XML in {args.data} (run to_pagexml first)")
    cmd = ["ketos", "segtrain", "-f", "page",
           "-o", str(out / "model"),
           "--epochs", str(args.epochs),
           "-r", str(args.lr0),
           "--device", args.device or "cuda",
           *xmls]
    print("running:", " ".join(cmd[:8]), f"... ({len(xmls)} xml)")
    subprocess.run(cmd, check=True)   # writes out/model_best.mlmodel
    print(f"best checkpoint: {out / 'model_best.mlmodel'}")


def cmd_predict(args):
    from PIL import Image
    from kraken import blla
    from kraken.lib import vgsl
    from tqdm import tqdm

    model = None
    if args.weights and args.weights not in ("blla", "default"):
        model = vgsl.TorchVGSLModel.load_model(args.weights)

    coco = json.loads(Path(args.coco).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions, speeds = [], []
    for img in tqdm(coco["images"], desc="kraken predict"):
        path = Path(args.images_root) / img["file_name"]
        try:
            im = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        t0 = time.perf_counter()
        seg = blla.segment(im, model=model) if model else blla.segment(im)
        speeds.append((time.perf_counter() - t0) * 1000.0)
        boundaries = [getattr(ln, "boundary", None) for ln in seg.lines]
        predictions.extend(lines_to_coco(boundaries, img["id"]))

    (out / "predictions.json").write_text(json.dumps(predictions))

    import kraken
    meta = {"model": "kraken-blla", "weights": str(args.weights),
            "device": str(args.device),
            "n_images": len(coco["images"]),
            "n_predictions": len(predictions),
            "note": "score=1.0 (kraken segmentation has no per-line confidence)",
            **speed_stats(speeds),
            "versions": {"kraken": getattr(kraken, "__version__", "unknown"),
                         "python": platform.python_version()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(predictions)} predictions for {len(coco['images'])} images")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--weights", default="blla")   # base model for segtrain
    t.add_argument("--data", required=True)        # dir of PAGE XML
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--lr0", type=float, default=0.0002)   # -> ketos -r
    t.add_argument("--imgsz", type=int, default=640)   # accepted, unused
    t.add_argument("--batch", type=int, default=1)     # accepted, unused
    t.add_argument("--device", default=None)
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--weights", default="blla")
    p.add_argument("--coco", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.0)  # accepted, unused
    p.add_argument("--imgsz", type=int, default=640)   # accepted, unused
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
