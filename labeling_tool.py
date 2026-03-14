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
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, jsonify, request, send_file, abort
from PIL import Image

APP_DIR = Path(__file__).parent.resolve()
ANNOTATIONS_DIR = APP_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

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
        fid = uuid4().hex[:12]
        data = f.read()
        image_store[fid] = {"name": f.filename, "data": data}
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

    # Save CSV
    csv_path = ANNOTATIONS_DIR / f"{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "x_min", "y_min", "x_max", "y_max"])
        for ann in annotations:
            writer.writerow([
                name, ann["label"],
                ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"],
            ])

    # Export cropped regions
    crops_dir = ANNOTATIONS_DIR / f"{base}_crops"
    crops_dir.mkdir(exist_ok=True)
    img = Image.open(io.BytesIO(entry["data"]))
    img.load()
    label_counts = {}
    for ann in annotations:
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in ann["label"])
        safe = safe.strip().replace(" ", "_") or "unlabeled"
        count = label_counts.get(safe, 0)
        label_counts[safe] = count + 1
        suffix = f"_{count}" if count > 0 else ""
        crop_path = crops_dir / f"{base}_{safe}{suffix}.png"
        cropped = img.crop((ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]))
        cropped.save(crop_path)

    return jsonify({"ok": True, "csv": str(csv_path), "crops": len(annotations)})


@app.route("/api/load-annotations/<fid>")
def load_annotations(fid):
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"annotations": []})
    base = Path(entry["name"]).stem
    csv_path = ANNOTATIONS_DIR / f"{base}.csv"
    annotations = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
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
