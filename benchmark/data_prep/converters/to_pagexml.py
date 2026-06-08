"""Convert COCO line annotations to PAGE XML for Kraken (ketos segtrain).

Kraken trains on baselines + boundary polygons, not bboxes. Our GT has only
bboxes, so we synthesise a baseline as a horizontal line at `baseline_frac` of
the box height (text baselines sit below the glyphs) and use the bbox rectangle
as the boundary. This is an approximation - document it as a limitation in the
thesis.

Usage (from benchmark/):
    python data_prep/converters/to_pagexml.py --coco instances_train.json \
        --out-dir dataset/pagexml/train
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def baseline_from_bbox(bbox, frac=0.75):
    """Horizontal baseline at `frac` of box height (left edge -> right edge)."""
    x, y, w, h = bbox
    yb = int(round(y + frac * h))
    return [(int(round(x)), yb), (int(round(x + w)), yb)]


def _bbox_polygon(bbox):
    x, y, w, h = bbox
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _pts(points):
    return " ".join(f"{int(round(px))},{int(round(py))}" for px, py in points)


def coco_image_to_pagexml(img, anns):
    """One image's COCO annotations -> PAGE XML string (single TextRegion)."""
    ET.register_namespace("", PAGE_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts")
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page",
                         imageFilename=Path(img["file_name"]).name,
                         imageWidth=str(img["width"]),
                         imageHeight=str(img["height"]))
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", id="r0")
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords",
                  points=_pts([(0, 0), (img["width"], 0),
                               (img["width"], img["height"]),
                               (0, img["height"])]))
    for i, a in enumerate(anns):
        line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", id=f"l{i}")
        ET.SubElement(line, f"{{{PAGE_NS}}}Coords",
                      points=_pts(_bbox_polygon(a["bbox"])))
        ET.SubElement(line, f"{{{PAGE_NS}}}Baseline",
                      points=_pts(baseline_from_bbox(a["bbox"])))
    return ET.tostring(root, encoding="unicode")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    coco = json.loads(Path(args.coco).read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for img in coco["images"]:
        xml = coco_image_to_pagexml(img, anns[img["id"]])
        (out / f"{Path(img['file_name']).stem}.xml").write_text(xml)
    print(f"{len(coco['images'])} PAGE XML files -> {out}")


if __name__ == "__main__":
    main()
