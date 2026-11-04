"""Convert COCO line annotations to PAGE XML for Kraken (ketos segtrain).

Kraken trains on baselines + boundary polygons, not bboxes. Our GT has only
bboxes, so we synthesise a baseline as a horizontal line at `baseline_frac` of
the box height (text baselines sit below the glyphs) and use the bbox rectangle
as the boundary. This is an approximation - document it as a limitation in the
thesis.

Usage (from line_benchmark/):
    python data_prep/converters/to_pagexml.py --coco instances_train.json \
        --out-dir dataset/pagexml/train [--per-group 46]
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
VARIANT_RE = re.compile(r"_\d+$")


def group_key(file_name):
    return VARIANT_RE.sub("", Path(file_name).stem)


def subsample(images, per_group):
    """At most `per_group` variants of each template page.

    Fewer variants of every layout beats every variant of a few layouts: the
    test split holds unseen template pages, so layout coverage is what carries
    over. Deterministic - the first N by file name."""
    if not per_group:
        return images
    groups = defaultdict(list)
    for img in images:
        groups[group_key(img["file_name"])].append(img)
    keep = []
    for key in sorted(groups):
        keep += sorted(groups[key], key=lambda i: i["file_name"])[:per_group]
    return keep


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


def image_ref(file_name, images_root=None, out_dir=None):
    """Kraken resolves imageFilename against the XML's own directory, so the
    bare basename COCO carries finds nothing once the XML lives in a folder of
    its own. Relative, never absolute - the tree is bind-mounted at a different
    path inside the container."""
    if not images_root or not out_dir:
        return Path(file_name).name
    return os.path.relpath(os.path.abspath(os.path.join(images_root, file_name)),
                           os.path.abspath(out_dir))


def coco_image_to_pagexml(img, anns, ref=None):
    """One image's COCO annotations -> PAGE XML string (single TextRegion)."""
    ET.register_namespace("", PAGE_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts")
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page",
                         imageFilename=ref or Path(img["file_name"]).name,
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
    ap.add_argument("--images-root",
                    help="where the images live; without it imageFilename is "
                         "the bare name, which only works if the XML sits in "
                         "the same folder as the image")
    ap.add_argument("--per-group", type=int, default=0,
                    help="cap variants per template page (0 = all); kraken "
                         "trains on a subsample, segtrain is far slower than "
                         "the detectors")
    args = ap.parse_args(argv)

    coco = json.loads(Path(args.coco).read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    images = subsample(coco["images"], args.per_group)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for img in images:
        ref = image_ref(img["file_name"], args.images_root, out)
        xml = coco_image_to_pagexml(img, anns[img["id"]], ref)
        (out / f"{Path(img['file_name']).stem}.xml").write_text(xml)
    print(f"{len(images)} PAGE XML files -> {out} "
          f"({len(coco['images'])} in the source)")


if __name__ == "__main__":
    main()
