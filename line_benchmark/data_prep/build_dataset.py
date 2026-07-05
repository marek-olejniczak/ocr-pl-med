"""Split COCO annotations into deterministic train/val/test sets.

Two modes:
    python data_prep/build_dataset.py --coco master.json --out-dir dataset/annotations
    python data_prep/build_dataset.py --train-coco gen.json --test-coco manual.json \
        --out-dir dataset/annotations
"""

import argparse
import json
import random
from pathlib import Path

SEED = 0


def split_image_ids(image_ids, val_frac, test_frac, seed=SEED):
    """Deterministic shuffle-split; input order must not matter."""
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    test = ids[:n_test]
    val = ids[n_test:n_test + n_val]
    train = ids[n_test + n_val:]
    return train, val, test


def subset_coco(coco, image_ids):
    ids = set(image_ids)
    return {
        **coco,
        "images": [i for i in coco["images"] if i["id"] in ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }


def _write(coco, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco))
    print(f"  {path.name}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", help="single master file -> train/val/test")
    ap.add_argument("--train-coco", help="explicit train file")
    ap.add_argument("--test-coco", help="explicit test file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    if args.coco and not (args.train_coco or args.test_coco):
        coco = json.loads(Path(args.coco).read_text())
        ids = [i["id"] for i in coco["images"]]
        train, val, test = split_image_ids(ids, args.val_frac,
                                           args.test_frac, args.seed)
        _write(subset_coco(coco, train), out / "instances_train.json")
        _write(subset_coco(coco, val), out / "instances_val.json")
        _write(subset_coco(coco, test), out / "instances_test.json")
    elif args.train_coco and args.test_coco:
        coco = json.loads(Path(args.train_coco).read_text())
        ids = [i["id"] for i in coco["images"]]
        train, val, _ = split_image_ids(ids, args.val_frac,
                                        test_frac=0.0, seed=args.seed)
        _write(subset_coco(coco, train), out / "instances_train.json")
        _write(subset_coco(coco, val), out / "instances_val.json")
        test = json.loads(Path(args.test_coco).read_text())
        _write(test, out / "instances_test.json")
    else:
        ap.error("use either --coco, or both --train-coco and --test-coco")


if __name__ == "__main__":
    main()
