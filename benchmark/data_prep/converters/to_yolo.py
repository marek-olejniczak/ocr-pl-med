"""Convert COCO split files into an ultralytics YOLO dataset layout.

Extracted from yolo_training.ipynb (convert_coco_to_yolo) and adapted:
splits come pre-made from build_dataset.py (the notebook shuffled without
a seed), images are symlinked instead of copied (--copy to revert).

Creates:
    out/images/{split}/   symlinks (or copies) to source images
    out/labels/{split}/   one txt per image: "0 cx cy w h" (normalized)
    out/data.yaml

Usage (from benchmark/):
    python data_prep/converters/to_yolo.py --annotations-dir dataset/annotations \
        --images-root ../../dataset --out-dir dataset/yolo_raw
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def _clip01(v):
    return min(max(v, 0.0), 1.0)


def coco_to_yolo_lines(annotations, img_w, img_h):
    lines = []
    for a in annotations:
        x, y, w, h = a["bbox"]
        cx, cy = _clip01((x + w / 2) / img_w), _clip01((y + h / 2) / img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} "
                     f"{_clip01(w / img_w):.6f} {_clip01(h / img_h):.6f}")
    return lines


def convert_coco_to_yolo(coco_path, images_root, out_dir, split, copy=False):
    coco = json.loads(Path(coco_path).read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    img_out = Path(out_dir) / "images" / split
    lbl_out = Path(out_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stems = set()
    for img in coco["images"]:
        rel = Path(img["file_name"])
        if rel.stem in stems:
            raise ValueError(f"duplicate stem: {rel.stem} "
                             "(YOLO pairs images/labels by stem)")
        stems.add(rel.stem)

        src = (Path(images_root) / rel).resolve()
        dst = img_out / rel.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

        lines = coco_to_yolo_lines(anns[img["id"]],
                                   img["width"], img["height"])
        (lbl_out / f"{rel.stem}.txt").write_text(
            "\n".join(lines) + "\n" if lines else "")
    return len(coco["images"])


def write_data_yaml(out_dir):
    # No `path:` key on purpose: ultralytics then resolves train/val/test
    # against the data.yaml's own directory, so the layout works wherever the
    # tree is mounted (host venv AND the /benchmark bind mount in Docker).
    # Baking an absolute path here breaks the moment host and container paths
    # differ.
    (Path(out_dir) / "data.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: line\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations-dir", required=True,
                    help="dir with instances_{train,val,test}.json")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    for split in SPLITS:
        coco_path = Path(args.annotations_dir) / f"instances_{split}.json"
        if not coco_path.exists():
            print(f"  {split}: no {coco_path.name}, skipping")
            continue
        n = convert_coco_to_yolo(coco_path, args.images_root, out,
                                 split, args.copy)
        print(f"  {split}: {n} images")
    write_data_yaml(out)


if __name__ == "__main__":
    main()
