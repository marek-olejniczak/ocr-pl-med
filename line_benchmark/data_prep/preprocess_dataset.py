"""Generate the photometric ("prep") variant of a dataset.

Only pixel-value operations run (CLAHE, grid removal, ink separation,
binarization). Geometry is untouched, so the same COCO annotations stay
valid for both raw and prep variants. Output mirrors the input tree
relative to --images-root.

Usage (from line_benchmark/):
    python data_prep/preprocess_dataset.py \
        --images-root ../../dataset --out-dir dataset/prep --workers 8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from preprocessing import DocumentPreprocessor, IMAGE_EXTS

# geometry-preserving config: GT boxes must stay valid on the output
PHOTOMETRIC_CONFIG = {"deskew": False, "border_px": 0, "alpha_crop": False}


def process_one(job):
    src, dst, force = job
    if dst.exists() and not force:
        return "skip"
    orig = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if orig is None:
        print(f"  cannot read: {src}")
        return "error"
    result = DocumentPreprocessor(PHOTOMETRIC_CONFIG).preprocess(str(src))
    if result.shape[:2] != orig.shape[:2]:
        raise RuntimeError(f"geometry changed for {src.name}: "
                           f"{orig.shape[:2]} -> {result.shape[:2]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), result)
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root, out = Path(args.images_root), Path(args.out_dir)
    images = [p for p in sorted(root.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTS]
    jobs = [(p, out / p.relative_to(root), args.force) for p in images]
    print(f"Processing {len(jobs)} images with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(process_one, jobs, chunksize=4))
    print(f"done: {results.count('ok')} processed, "
          f"{results.count('skip')} skipped, {results.count('error')} errors")


if __name__ == "__main__":
    main()
