"""Deterministic per-template subsamples of train and val for HPO trials.

Writes, for --train-per-group 100 --val-per-group 67:
    dataset/hpo/instances_train_g100.json     COCO, for detectron2
    dataset/hpo/instances_val_g67.json
    dataset/yolo_raw/hpo_train_g100.txt       image lists for ultralytics, next
    dataset/yolo_raw/hpo_val_g67.txt          to images/: its loader resolves
    dataset/yolo_raw/data_hpo_g100_g67.yaml   only ./ lines against the list

Usage (from line_benchmark/):
    python hpo/prepare.py --train-per-group 100 --val-per-group 67
"""

import argparse
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from data_prep.build_dataset import subset_coco  # noqa: E402
from data_prep.converters.to_pagexml import subsample  # noqa: E402


def coco_subset(coco_path, per_group):
    coco = json.loads(Path(coco_path).read_text())
    keep = subsample(coco["images"], per_group)
    return subset_coco(coco, [i["id"] for i in keep])


def image_list(coco, split):
    return "".join(f"./images/{split}/{Path(i['file_name']).name}\n"
                   for i in coco["images"])


def prepare(ann_dir, yolo_dir, out_dir, train_per_group, val_per_group):
    ann_dir, yolo_dir, out_dir = Path(ann_dir), Path(yolo_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_t, tag_v = f"g{train_per_group}", f"g{val_per_group}"

    train = coco_subset(ann_dir / "instances_train.json", train_per_group)
    val = coco_subset(ann_dir / "instances_val.json", val_per_group)
    paths = {
        "train_coco": out_dir / f"instances_train_{tag_t}.json",
        "val_coco": out_dir / f"instances_val_{tag_v}.json",
        "train_txt": yolo_dir / f"hpo_train_{tag_t}.txt",
        "val_txt": yolo_dir / f"hpo_val_{tag_v}.txt",
        "data_yaml": yolo_dir / f"data_hpo_{tag_t}_{tag_v}.yaml",
    }
    paths["train_coco"].write_text(json.dumps(train))
    paths["val_coco"].write_text(json.dumps(val))
    paths["train_txt"].write_text(image_list(train, "train"))
    paths["val_txt"].write_text(image_list(val, "val"))
    paths["data_yaml"].write_text(
        f"train: {paths['train_txt'].name}\n"
        f"val: {paths['val_txt'].name}\n"
        "names:\n"
        "  0: line\n")
    missing = [p for p in (yolo_dir / ln[2:].strip()
                           for ln in paths["train_txt"].read_text().splitlines()[:50])
               if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} listed images are not in {yolo_dir}, "
                         f"e.g. {missing[0]} - run to_yolo first")
    return paths, len(train["images"]), len(val["images"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", default="dataset/annotations")
    ap.add_argument("--yolo-dir", default="dataset/yolo_raw")
    ap.add_argument("--out-dir", default="dataset/hpo")
    ap.add_argument("--train-per-group", type=int, default=100)
    ap.add_argument("--val-per-group", type=int, default=67)
    args = ap.parse_args(argv)
    paths, n_train, n_val = prepare(args.ann_dir, args.yolo_dir, args.out_dir,
                                    args.train_per_group, args.val_per_group)
    print(f"train {n_train} stron, val {n_val} stron")
    for key, p in paths.items():
        print(f"  {key:10s} {p}")


if __name__ == "__main__":
    main()
