"""Page-level ground truth: JSONL, one record per page.

    {"image": "pages/scan_01.png",
     "lines": [{"bbox": [x, y, w, h], "text": "Rp. Amoksycylina 500mg"}]}

bbox is COCO xywh in original image coordinates, image paths resolve
relative to the JSONL file.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GTLine:
    bbox: tuple  # x, y, w, h
    text: str


@dataclass
class GTPage:
    image_path: Path
    lines: list


def page_text(lines):
    ordered = sorted(lines, key=lambda l: l.bbox[1] + l.bbox[3] / 2)
    return "\n".join(l.text for l in ordered)


def load(jsonl_path):
    root = Path(jsonl_path).resolve().parent
    pages = []
    with open(jsonl_path) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if "image" not in rec or "lines" not in rec:
                raise ValueError(
                    f"{jsonl_path}:{lineno}: record needs 'image' and 'lines'")
            lines = [GTLine(tuple(l["bbox"]), l["text"]) for l in rec["lines"]]
            pages.append(GTPage(root / rec["image"], lines))
    return pages


def from_coco(coco_path, out_jsonl, text_key="text"):
    """COCO with per-annotation transcriptions -> GT JSONL.
    Returns the number of annotations without a transcription."""
    coco = json.loads(Path(coco_path).read_text())
    by_image = {img["id"]: (img["file_name"], []) for img in coco["images"]}
    skipped = 0
    for ann in coco["annotations"]:
        text = ann.get(text_key,
                       (ann.get("attributes") or {}).get(text_key))
        if text is None:
            skipped += 1
            continue
        by_image[ann["image_id"]][1].append(
            {"bbox": list(ann["bbox"]), "text": text})
    with open(out_jsonl, "w") as f:
        for file_name, lines in by_image.values():
            f.write(json.dumps({"image": file_name, "lines": lines},
                               ensure_ascii=False) + "\n")
    return skipped
