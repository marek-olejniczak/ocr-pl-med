"""Load form-template pages for the dataset generator.

A templates directory (built by build_templates.py) contains one COCO
annotations.json plus one subdirectory per page:

    templates/
      annotations.json
      <stem>/
        <stem>_blank.png     # all fill-in fields cleared (always present)
        <stem>_partial.png   # only f-fields keep their real handwriting (optional)

Labels: p (printed), t (text), n (number), mix (letters+digits),
f (real handwriting present in the original scan).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALID_LABELS = {"p", "n", "t", "f", "mix"}


@dataclass
class TemplatePage:
    name: str
    blank_path: Path
    partial_path: Optional[Path]
    fields: list[dict]  # {"label", "x_min", "y_min", "x_max", "y_max"}


def load_templates(templates_dir: Path) -> list[TemplatePage]:
    """Read annotations.json and resolve per-page image paths.

    Pages whose _blank.png is missing are skipped with a warning.
    Annotations with labels outside VALID_LABELS are ignored.
    """
    ann_path = templates_dir / "annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    fields_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        label = cat_names.get(ann["category_id"], "")
        if label not in VALID_LABELS:
            continue
        x, y, w, h = ann["bbox"]
        fields_by_image.setdefault(ann["image_id"], []).append({
            "label": label,
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x + w),
            "y_max": int(y + h),
        })

    pages: list[TemplatePage] = []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        page_dir = templates_dir / stem
        blank = page_dir / f"{stem}_blank.png"
        if not blank.exists():
            print(f"  SKIP template '{stem}': {blank} not found", file=sys.stderr)
            continue
        partial = page_dir / f"{stem}_partial.png"
        pages.append(TemplatePage(
            name=stem,
            blank_path=blank,
            partial_path=partial if partial.exists() else None,
            fields=fields_by_image.get(im["id"], []),
        ))
    return pages
