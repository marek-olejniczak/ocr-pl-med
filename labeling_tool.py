"""
OCR Annotation Labeling Tool — Flask + HTML5 Canvas (offline, no tkinter)
Run: python labeling_tool.py
Opens browser at http://127.0.0.1:5000
"""

import csv
import io
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, jsonify, request, send_file, abort
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

APP_DIR = Path(__file__).parent.resolve()
DATASET_DIR = APP_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)
COCO_PATH = DATASET_DIR / "annotations.json"

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ── State ──────────────────────────────────────────────────────────
# Stores uploaded images in memory for the session
image_store = {}  # id -> {"path": temp_path, "name": original_name, "data": bytes}
image_order = []  # list of ids in upload order
current_id = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    global current_id
    files = request.files.getlist("images")
    if not files:
        return jsonify({"ok": False, "error": "No files"})

    new_ids = []
    for f in files:
        if not f.filename:
            continue
        fname = f.filename
        data = f.read()
        if fname.lower().endswith(".pdf"):
            if fitz is None:
                return jsonify({"ok": False, "error": "PyMuPDF not installed — run: pip install PyMuPDF"})
            try:
                pages = _pdf_to_pngs(data)
            except Exception as e:
                return jsonify({"ok": False, "error": f"Failed to read {fname}: {e}"})
            base = Path(fname).stem
            pad = max(2, len(str(len(pages))))
            for i, page_png in enumerate(pages, 1):
                fid = uuid4().hex[:12]
                page_name = f"{base}_p{str(i).zfill(pad)}.png"
                image_store[fid] = {"name": page_name, "data": page_png}
                new_ids.append(fid)
        else:
            fid = uuid4().hex[:12]
            image_store[fid] = {"name": fname, "data": data}
            new_ids.append(fid)

    if not new_ids:
        return jsonify({"ok": False, "error": "No valid images"})

    # Replace previous set
    image_order.clear()
    image_order.extend(new_ids)
    current_id = new_ids[0]

    return jsonify({"ok": True, "images": _all_image_info(), "current": current_id})


@app.route("/api/image/<fid>")
def serve_image(fid):
    entry = image_store.get(fid)
    if not entry:
        abort(404)
    return send_file(io.BytesIO(entry["data"]), mimetype="image/png")


@app.route("/api/set-current", methods=["POST"])
def set_current():
    global current_id
    fid = request.json.get("id")
    if fid not in image_store:
        return jsonify({"ok": False})
    current_id = fid
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def save():
    data = request.json
    fid = data.get("id")
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"ok": False, "error": "Image not found"})

    annotations = data.get("annotations", [])
    name = entry["name"]
    base = Path(name).stem
    ext = Path(name).suffix or ".jpg"

    # Create image folder: dataset/<image_name>/
    image_dir = DATASET_DIR / base
    image_dir.mkdir(exist_ok=True)

    # Save copy of original image
    img = Image.open(io.BytesIO(entry["data"]))
    img.load()
    img.save(image_dir / f"{base}{ext}")

    # Create lines folder and export cropped regions
    lines_dir = image_dir / "lines"
    lines_dir.mkdir(exist_ok=True)
    # Clear old lines
    for old in lines_dir.iterdir():
        old.unlink()
    for i, ann in enumerate(annotations, 1):
        crop_path = lines_dir / f"line{i}.jpg"
        cropped = img.crop((ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]))
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        cropped.save(crop_path)

    # Append/update single COCO annotations.json at dataset root
    _save_coco(name, img.width, img.height, annotations)

    # Persist timing for this image, if provided
    time_seconds = data.get("time_seconds")
    started_at = data.get("started_at")
    if time_seconds is not None:
        _save_timing(name, time_seconds, started_at)

    return jsonify({"ok": True, "coco": str(COCO_PATH), "crops": len(annotations)})


def _pdf_to_pngs(pdf_bytes, dpi=200):
    """Render each PDF page to PNG bytes."""
    out = []
    zoom_factor = dpi / 72.0
    matrix = fitz.Matrix(zoom_factor, zoom_factor)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out.append(pix.tobytes("png"))
    return out


def _save_timing(filename, time_seconds, started_at):
    """Append/update a row in dataset/timings.csv (one row per image)."""
    timings_path = DATASET_DIR / "timings.csv"
    rows = []
    if timings_path.exists():
        with open(timings_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["filename"] != filename:
                    rows.append(row)

    saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows.append({
        "filename": filename,
        "time_seconds": f"{float(time_seconds):.2f}",
        "started_at": started_at or "",
        "saved_at": saved_at,
    })

    with open(timings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "time_seconds", "started_at", "saved_at"])
        writer.writeheader()
        writer.writerows(rows)


def _load_coco():
    if COCO_PATH.exists():
        with open(COCO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"images": [], "annotations": [], "categories": []}


def _write_coco(coco):
    with open(COCO_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def _category_id(coco, label):
    for cat in coco["categories"]:
        if cat["name"] == label:
            return cat["id"]
    new_id = max((c["id"] for c in coco["categories"]), default=0) + 1
    coco["categories"].append({"id": new_id, "name": label})
    return new_id


def _save_coco(filename, width, height, annotations):
    """Update annotations.json — replace annotations for this image, keep others."""
    coco = _load_coco()

    image = next((im for im in coco["images"] if im["file_name"] == filename), None)
    if image is None:
        image = {"id": max((im["id"] for im in coco["images"]), default=0) + 1,
                 "file_name": filename}
        coco["images"].append(image)
    image["width"] = width
    image["height"] = height

    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != image["id"]]
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    for i, ann in enumerate(annotations, 1):
        w = ann["x_max"] - ann["x_min"]
        h = ann["y_max"] - ann["y_min"]
        coco["annotations"].append({
            "id": next_ann_id,
            "image_id": image["id"],
            "category_id": _category_id(coco, ann["label"]),
            "bbox": [ann["x_min"], ann["y_min"], w, h],
            "area": w * h,
            "iscrowd": 0,
            "crop": f"line{i}.jpg",
        })
        next_ann_id += 1

    _write_coco(coco)


def _migrate_csv_to_coco():
    """One-time: convert legacy annotations.csv to COCO annotations.json."""
    csv_path = DATASET_DIR / "annotations.csv"
    if COCO_PATH.exists() or not csv_path.exists():
        return
    by_image = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row["filename"], []).append(row)
    for filename, rows in by_image.items():
        img_path = DATASET_DIR / Path(filename).stem / filename
        if img_path.exists():
            with Image.open(img_path) as img:
                width, height = img.size
        else:
            # Image copy missing — fall back to bbox extents
            width = max(int(r["x_max"]) for r in rows)
            height = max(int(r["y_max"]) for r in rows)
        annotations = [{
            "label": r["label"],
            "x_min": int(r["x_min"]), "y_min": int(r["y_min"]),
            "x_max": int(r["x_max"]), "y_max": int(r["y_max"]),
        } for r in rows]
        _save_coco(filename, width, height, annotations)
    print(f"Migrated {csv_path.name} -> {COCO_PATH.name} ({len(by_image)} images)")


@app.route("/api/load-annotations/<fid>")
def load_annotations(fid):
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"annotations": []})
    name = entry["name"]
    coco = _load_coco()
    annotations = []
    image = next((im for im in coco["images"] if im["file_name"] == name), None)
    if image is not None:
        labels = {c["id"]: c["name"] for c in coco["categories"]}
        for a in coco["annotations"]:
            if a["image_id"] != image["id"]:
                continue
            x, y, w, h = a["bbox"]
            annotations.append({
                "label": labels.get(a["category_id"], ""),
                "x_min": int(x), "y_min": int(y),
                "x_max": int(x + w), "y_max": int(y + h),
            })
    return jsonify({"annotations": annotations})


def _all_image_info():
    result = []
    for fid in image_order:
        entry = image_store[fid]
        img = Image.open(io.BytesIO(entry["data"]))
        w, h = img.size
        result.append({"id": fid, "name": entry["name"], "width": w, "height": h})
    return result


def main():
    _migrate_csv_to_coco()
    url = "http://127.0.0.1:5000"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Labeling tool running at {url}")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
