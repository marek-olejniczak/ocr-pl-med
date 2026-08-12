"""Split COCO annotations into deterministic train/val/test sets.

Two modes:
    python data_prep/build_dataset.py --coco master.json --out-dir dataset/annotations
    python data_prep/build_dataset.py --train-coco gen.json --test-coco manual.json \
        --out-dir dataset/annotations

With --group the split runs over template pages instead of single images: a
generated set holds hundreds of variants of the same page, so splitting by
image would put near-identical pages in train and test. The group key is the
file name without its trailing variant index (KARTA_PACJENTA_1_p01_0007.jpg
-> KARTA_PACJENTA_1_p01).
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 0
VARIANT_RE = re.compile(r"_\d+$")


def group_key(file_name):
    return VARIANT_RE.sub("", Path(file_name).stem)


def split_groups(images, val_frac, test_frac, seed=SEED, test_groups=None):
    """Group-aware split. Returns image ids per split.

    test_groups pins the test split to named groups (e.g. the pages that also
    exist as real scans); the rest is shuffled into train/val."""
    groups = defaultdict(list)
    for img in images:
        groups[group_key(img["file_name"])].append(img["id"])
    names = sorted(groups)

    if test_groups:
        unknown = sorted(set(test_groups) - set(names))
        if unknown:
            raise ValueError("unknown test groups: " + ", ".join(unknown))
        pinned = set(test_groups)
        test = [n for n in names if n in pinned]
        rest = [n for n in names if n not in pinned]
    else:
        rest = list(names)
        random.Random(seed).shuffle(rest)
        n_test = round(len(names) * test_frac)
        test, rest = rest[:n_test], rest[n_test:]

    random.Random(seed + 1).shuffle(rest)
    n_val = round(len(names) * val_frac)
    val, train = rest[:n_val], rest[n_val:]

    def ids(chosen):
        return [i for n in chosen for i in groups[n]]

    print(f"  groups: {len(train)} train, {len(val)} val, {len(test)} test "
          f"(of {len(names)})")
    return ids(train), ids(val), ids(test)


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
    ap.add_argument("--group", action="store_true",
                    help="split by template page, not by image")
    ap.add_argument("--test-groups",
                    help="file with one group name per line -> the test split")
    args = ap.parse_args(argv)

    pinned = None
    if args.test_groups:
        pinned = [ln.strip() for ln
                  in Path(args.test_groups).read_text().splitlines()
                  if ln.strip()]

    out = Path(args.out_dir)
    if args.coco and not (args.train_coco or args.test_coco):
        coco = json.loads(Path(args.coco).read_text())
        if args.group or pinned:
            train, val, test = split_groups(coco["images"], args.val_frac,
                                            args.test_frac, args.seed, pinned)
        else:
            ids = [i["id"] for i in coco["images"]]
            train, val, test = split_image_ids(ids, args.val_frac,
                                               args.test_frac, args.seed)
        _write(subset_coco(coco, train), out / "instances_train.json")
        _write(subset_coco(coco, val), out / "instances_val.json")
        _write(subset_coco(coco, test), out / "instances_test.json")
    elif args.train_coco and args.test_coco:
        coco = json.loads(Path(args.train_coco).read_text())
        if args.group:
            train, val, _ = split_groups(coco["images"], args.val_frac,
                                         test_frac=0.0, seed=args.seed)
        else:
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
