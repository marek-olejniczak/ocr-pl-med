"""Build the final real test set (notes + patient cards) in the test/real/v1 layout.

Sources:
    --v1        test/real/v1 package (images of the original 114 note pages,
                text_corrections_v1.csv)
    --v13       test/real/v1.3 package or just its manifest.csv + lines.csv
                (GT for the 114 v1 pages)
    --labeled   labeler dataset dirs (annotations.csv + <stem>/<file> images):
                patient cards, Zyly Glebokie, Nerw Twarzowy

Tags in labels:
    #ignore      not a line (bar, signature, figure); COCO iscrowd=1, no crop
    #unreadable  real line, text unknown; detection GT, no crop
    [?]          illegible fragment inside a line; detection GT, no crop
Old tags of the v1.x notes are mapped to these.

Usage (from the repo root):
    python line_benchmark/data_prep/build_real_test.py --v1 real_test_v1 --v13 v13 \
        --labeled medyk_labeling/dataset medyk_labeling/dataset_zyly \
                  medyk_labeling/dataset_jakub --out real_test_v2
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# docs_20k train/val templates (KP2) and the KP1_p12 form (KP3 s.94): kept out of test
CARD_DEV = {("2", 3), ("2", 9), ("2", 11), ("2", 17), ("2", 18), ("2", 23), ("2", 24), ("3", 94)}
NAME_RES = [
    (re.compile(r"^Kopia KARTA PACJENTA (\d)_page(\d+)\.png$"), "karta_pacjenta_{}", "card", "scan", 0),
    (re.compile(r"^Żyły Głębokie_page(\d+)\.png$"), "zyly_glebokie", "notes", "tablet", 1),
    (re.compile(r"^Nerw Twarzowy_page(\d+)\.png$"), "nerw_twarzowy", "notes", "tablet", 1),
]
FIELDS = ["page_id", "document", "page", "split", "kind", "in_v1", "source_type", "grid_paper",
          "width", "height", "n_lines", "original_name", "label_group", "sha256"]


def normalize(text):
    text = text.strip()
    if text in ("[BLANK]", "[NIECZYTELNE]", "[NIECZTYELNE]"):
        return "#unreadable"
    return re.sub(r"\[(NIECZYTELNE|NIECZTYELNE|BLANK)\]", "[?]", text)


def ocr_usable(text):
    return not text.startswith("#") and "[?]" not in text


def v1_pages(v1, v13):
    """114 original note pages: v1.3 GT, tags normalized, Marek's text fixes kept."""
    v1_manifest = {r["page_id"]: r for r in csv.DictReader(open(v1 / "manifest.csv", encoding="utf-8"))}
    lines = defaultdict(list)
    for r in csv.DictReader(open(v13 / "lines.csv", encoding="utf-8")):
        if r["page_id"] in v1_manifest:
            lines[r["page_id"]].append([normalize(r["text"]),
                                        *(int(float(r[k])) for k in ("x_min", "y_min", "x_max", "y_max"))])
    assert set(lines) == set(v1_manifest), set(v1_manifest) ^ set(lines)

    log = []
    for c in csv.DictReader(open(v1 / "annotations/text_corrections_v1.csv", encoding="utf-8")):
        page = c["line_id"].rsplit("_l", 1)[0]
        if any(c["text_new"] in ln[0] for ln in lines[page]):
            continue
        # Jakub re-transcribed some lines (e.g. added the latin name), so match inside the line
        hits = [ln for ln in lines[page] if c["text_old"] in ln[0]]
        if len(hits) == 1:
            hits[0][0] = hits[0][0].replace(c["text_old"], c["text_new"])
            log.append((page, c["text_old"], c["text_new"], "reapplied"))
        else:
            log.append((page, c["text_old"], c["text_new"], f"not found ({len(hits)} matches)"))

    pages = []
    for pid, m in v1_manifest.items():
        src = v1 / "images" / m["document"] / f"{pid}.png"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == m["sha256"], pid
        pages.append(({**m, "kind": "notes", "in_v1": 1}, src, lines[pid]))
    return pages, log


def labeled_pages(dirs):
    pages = []
    for d in dirs:
        rows = defaultdict(list)
        for r in csv.DictReader(open(d / "annotations.csv", encoding="utf-8")):
            rows[r["filename"]].append([r["label"].strip(),
                                        *(int(float(r[k])) for k in ("x_min", "y_min", "x_max", "y_max"))])
        for name, lines in rows.items():
            for rx, doc, kind, source_type, grid in NAME_RES:
                m = rx.match(name)
                if m:
                    break
            else:
                raise SystemExit(f"unknown page name: {name}")
            *card, page = m.groups()
            document = doc.format(*card)
            split = "dev" if card and (card[0], int(page)) in CARD_DEV else "test"
            src = d / Path(name).stem / name
            assert src.exists(), src
            meta = {"page_id": f"{document}_p{page}", "document": document, "page": int(page),
                    "split": split, "kind": kind, "in_v1": 0, "source_type": source_type,
                    "grid_paper": grid, "original_name": name, "label_group": ""}
            pages.append((meta, src, lines))
    return pages


def write(pages, out):
    if out.exists():
        raise SystemExit(f"{out} exists, refusing to overwrite")
    (out / "annotations").mkdir(parents=True)
    coco = {"info": {"description": "real test set: handwritten notes and patient cards"},
            "images": [], "annotations": [], "categories": [{"id": 1, "name": "text_line"}]}
    rows, crops, problems = [], [], []
    ids = [m["page_id"] for m, _, _ in pages]
    assert len(ids) == len(set(ids)), "duplicate page ids"
    for img_id, (m, src, lines) in enumerate(sorted(pages, key=lambda p: p[0]["page_id"]), 1):
        pid, doc = m["page_id"], m["document"]
        for sub in ("images", "crops"):
            (out / sub / doc).mkdir(parents=True, exist_ok=True)
        dst = out / "images" / doc / f"{pid}.png"
        img = Image.open(src).convert("RGB")
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            img.save(dst)
        W, H = img.size
        m.update(width=W, height=H, n_lines=len(lines),
                 sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
        coco["images"].append({"id": img_id, "file_name": f"images/{doc}/{pid}.png",
                               "width": W, "height": H, "page_id": pid})
        for i, (text, x0, y0, x1, y1) in enumerate(lines, 1):
            lid = f"{pid}_l{i:03d}"
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0 or not text:
                problems.append((lid, text, x0, y0, x1, y1))
                continue
            rows.append([lid, pid, x0, y0, x1, y1, text, m["original_name"], ""])
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id,
                                        "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                                        "area": (x1 - x0) * (y1 - y0),
                                        "iscrowd": int(text == "#ignore"), "text": text, "line_id": lid})
            if ocr_usable(text):
                img.crop((x0, y0, x1, y1)).save(out / "crops" / doc / f"{lid}.jpg", quality=95)
                crops.append([f"crops/{doc}/{lid}.jpg", text, lid])

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted((m for m, _, _ in pages), key=lambda m: m["page_id"]))
    with open(out / "annotations/lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "page_id", "x_min", "y_min", "x_max", "y_max", "text", "orig_file", "orig_crop"])
        w.writerows(rows)
    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "label", "line_id"])
        w.writerows(crops)
    json.dump(coco, open(out / "annotations/instances.json", "w", encoding="utf-8"), ensure_ascii=False)
    return rows, crops, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v13", type=Path, required=True)
    ap.add_argument("--labeled", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    notes, log = v1_pages(args.v1, args.v13)
    pages = notes + labeled_pages(args.labeled)
    rows, crops, problems = write(pages, args.out)
    with open(args.out / "annotations/text_fixes_reapplied.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "text_old", "text_new", "status"])
        w.writerows(log)

    by = defaultdict(lambda: [0, 0])
    for m, _, lines in pages:
        by[(m["kind"], m["split"], m["in_v1"])][0] += 1
        by[(m["kind"], m["split"], m["in_v1"])][1] += len(lines)
    for k, (n, nl) in sorted(by.items()):
        print(f"{k}: {n} pages, {nl} boxes")
    print(f"{len(rows)} boxes, {len(crops)} OCR crops; text fixes: "
          f"{sum(s == 'reapplied' for *_, s in log)} reapplied, "
          f"{[r for r in log if r[3] != 'reapplied']} not found")
    print("problems:", problems)


if __name__ == "__main__":
    main()
