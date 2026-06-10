"""
OCR Annotation Labeling Tool — Flask + HTML5 Canvas (offline, no tkinter)
Run: python labeling_tool.py
Opens browser at http://127.0.0.1:5000
"""

import csv
import io
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

    # Append/update single annotations.csv at dataset root
    _save_csv(base, name, annotations)

    # Persist timing for this image, if provided
    time_seconds = data.get("time_seconds")
    started_at = data.get("started_at")
    if time_seconds is not None:
        _save_timing(name, time_seconds, started_at)

    return jsonify({"ok": True, "csv": str(DATASET_DIR / "annotations.csv"), "crops": len(annotations)})


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


def _save_csv(base, filename, annotations):
    """Update the single annotations.csv — replace rows for this image, keep others."""
    csv_path = DATASET_DIR / "annotations.csv"
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["filename"] != filename:
                    existing_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "x_min", "y_min", "x_max", "y_max", "crop"])
        for row in existing_rows:
            writer.writerow([row["filename"], row["label"], row["x_min"], row["y_min"], row["x_max"], row["y_max"], row.get("crop", "")])
        for i, ann in enumerate(annotations, 1):
            crop_name = f"line{i}.jpg"
            writer.writerow([filename, ann["label"], ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"], crop_name])


@app.route("/api/load-annotations/<fid>")
def load_annotations(fid):
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"annotations": []})
    name = entry["name"]
    csv_path = DATASET_DIR / "annotations.csv"
    annotations = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["filename"] == name:
                    annotations.append({
                        "label": row["label"],
                        "x_min": int(row["x_min"]), "y_min": int(row["y_min"]),
                        "x_max": int(row["x_max"]), "y_max": int(row["y_max"]),
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
    url = "http://127.0.0.1:5000"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Labeling tool running at {url}")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
