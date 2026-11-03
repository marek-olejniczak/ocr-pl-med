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
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF
from flask import Flask, render_template, jsonify, request, send_file, abort
from PIL import Image

import auto_annotate

APP_DIR = Path(__file__).parent.resolve()

# PDF rendering: each page is rasterized at the native resolution of its
# scanned image (never below PDF_RENDER_DPI), capped so the long side does
# not exceed PDF_MAX_SIDE_PX — hundreds of pages are kept in RAM.
PDF_RENDER_DPI = 200
PDF_MAX_DPI = 600
PDF_MAX_SIDE_PX = int(os.environ.get("PDF_MAX_SIDE_PX", "4000"))
DATASET_DIR = APP_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
# Skany notatek w PDF potrafią ważyć kilkaset MB. Limit można zmienić
# zmienną środowiskową MAX_UPLOAD_MB.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "2048"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def too_large(_err):
    return jsonify({
        "ok": False,
        "error": f"Plik za duży (limit {MAX_UPLOAD_MB} MB). "
                 "Uruchom z większym limitem: MAX_UPLOAD_MB=4096 python labeling_tool.py",
    }), 413

# ── State ──────────────────────────────────────────────────────────
# Stores uploaded images in memory for the session
image_store = {}  # id -> {"path": temp_path, "name": original_name, "data": bytes}
image_order = []  # list of ids in upload order
current_id = None

# Auto-annotation (HTR) background jobs: job_id -> {"progress":, "cancel":}
_jobs = {}
_jobs_lock = threading.Lock()


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
    rotations = _load_rotations()
    for f in files:
        if not f.filename:
            continue
        data = f.read()
        if Path(f.filename).suffix.lower() == ".pdf":
            try:
                new_ids.extend(_ingest_pdf(f.filename, data, rotations))
            except Exception as e:  # noqa: BLE001
                return jsonify({"ok": False, "error": f"Cannot read PDF {f.filename}: {e}"})
        else:
            fid = uuid4().hex[:12]
            rotation = rotations.get(f.filename, 0)
            image_store[fid] = {
                "name": f.filename,
                "data": _rotate_bytes(data, rotation),
                "rotation": rotation,
            }
            new_ids.append(fid)

    if not new_ids:
        return jsonify({"ok": False, "error": "No valid images"})

    # Replace previous set
    image_order.clear()
    image_order.extend(new_ids)
    current_id = new_ids[0]

    return jsonify({"ok": True, "images": _all_image_info(), "current": current_id})


def _ingest_pdf(filename, pdf_bytes, rotations=None):
    """Rasterize each PDF page to a PNG image and register it in the store.

    Returns the list of new image ids. Pages are named
    ``<pdf_stem>_page01.png``, ``<pdf_stem>_page02.png``, ... so they sort
    naturally and are traceable back to their source document. Pages rotated
    and saved earlier are rotated the same way again (see rotations.json).
    """
    rotations = rotations or {}
    base = Path(filename).stem
    ids = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pad = max(2, len(str(doc.page_count)))
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=_page_render_dpi(page))
            png_bytes = pix.tobytes("png")
            fid = uuid4().hex[:12]
            name = f"{base}_page{str(i + 1).zfill(pad)}.png"
            rotation = rotations.get(name, 0)
            image_store[fid] = {
                "name": name,
                "data": _rotate_bytes(png_bytes, rotation),
                "rotation": rotation,
            }
            ids.append(fid)
    finally:
        doc.close()
    return ids


# ── Page rotation ───────────────────────────────────────────────────
# Orientation of saved pages is remembered in dataset/rotations.json
# (page name -> clockwise degrees), so re-uploading the same PDF gives the
# same orientation as the stored copy and the saved boxes still line up.

def _rotations_path():
    return DATASET_DIR / "rotations.json"


def _load_rotations():
    try:
        with open(_rotations_path(), encoding="utf-8") as f:
            return {k: int(v) % 360 for k, v in json.load(f).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _remember_rotation(name, rotation):
    rotations = _load_rotations()
    if rotation % 360:
        rotations[name] = rotation % 360
    elif name in rotations:
        del rotations[name]
    else:
        return
    with open(_rotations_path(), "w", encoding="utf-8") as f:
        json.dump(rotations, f, ensure_ascii=False, indent=2, sort_keys=True)


def _rotate_bytes(data, degrees):
    """Rotate image bytes clockwise by a multiple of 90 degrees."""
    degrees %= 360
    if not degrees:
        return data
    img = Image.open(io.BytesIO(data))
    img.load()
    rotated = img.rotate(-degrees, expand=True)  # PIL rotates counter-clockwise
    buf = io.BytesIO()
    rotated.save(buf, format="PNG")
    return buf.getvalue()


@app.route("/api/rotate", methods=["POST"])
def rotate_page():
    data = request.get_json(silent=True) or {}
    entry = image_store.get(data.get("id"))
    if not entry:
        return jsonify({"ok": False, "error": "Image not found"}), 404
    try:
        degrees = int(data.get("degrees"))
    except (TypeError, ValueError):
        degrees = 0
    if degrees not in (90, -90, 180):
        return jsonify({"ok": False, "error": "degrees must be 90, -90 or 180"}), 400

    entry["data"] = _rotate_bytes(entry["data"], degrees)
    entry["rotation"] = (entry.get("rotation", 0) + degrees) % 360
    with Image.open(io.BytesIO(entry["data"])) as img:
        width, height = img.size
    return jsonify({"ok": True, "width": width, "height": height})


def _page_render_dpi(page):
    """DPI matching the scanned image on the page, so nothing is downsampled.

    Uses the largest embedded image that covers a meaningful part of the page.
    Pages without images (vector/text PDFs) fall back to PDF_RENDER_DPI.
    """
    rect = page.rect
    page_area = rect.width * rect.height
    dpi = float(PDF_RENDER_DPI)

    for info in page.get_image_info():
        bbox = fitz.Rect(info["bbox"])
        if bbox.is_empty or bbox.width * bbox.height < 0.3 * page_area:
            continue
        # max/max is robust to images placed with 90° rotation
        native = max(info["width"], info["height"]) / (max(bbox.width, bbox.height) / 72)
        dpi = max(dpi, native)

    dpi = min(dpi, PDF_MAX_DPI)
    long_side_in = max(rect.width, rect.height) / 72
    if long_side_in > 0:
        dpi = min(dpi, PDF_MAX_SIDE_PX / long_side_in)
    return max(1, int(round(dpi)))


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

    # Orientation is persisted only together with the image copy and CSV,
    # so the stored files always agree with each other.
    _remember_rotation(name, entry.get("rotation", 0))

    return jsonify({"ok": True, "csv": str(DATASET_DIR / "annotations.csv"), "crops": len(annotations)})


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
        writer.writerow(["filename", "label", "x_min", "y_min", "x_max", "y_max", "crop", "source"])
        for row in existing_rows:
            writer.writerow([row["filename"], row["label"], row["x_min"], row["y_min"], row["x_max"], row["y_max"], row.get("crop", ""), row.get("source") or "manual"])
        for i, ann in enumerate(annotations, 1):
            crop_name = f"line{i}.jpg"
            writer.writerow([filename, ann["label"], ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"], crop_name, ann.get("source") or "manual"])


@app.route("/api/load-annotations/<fid>")
def load_annotations(fid):
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"annotations": []})
    name = entry["name"]
    csv_path = DATASET_DIR / "annotations.csv"
    sx, sy = _annotation_scale(name, entry["data"])
    annotations = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["filename"] == name:
                    annotations.append({
                        "label": row["label"],
                        "x_min": round(int(row["x_min"]) * sx), "y_min": round(int(row["y_min"]) * sy),
                        "x_max": round(int(row["x_max"]) * sx), "y_max": round(int(row["y_max"]) * sy),
                        "source": row.get("source") or "manual",
                    })
    return jsonify({"annotations": annotations})


def _annotation_scale(name, current_data):
    """Scale factors from saved annotation coordinates to the current image.

    Saved coordinates refer to the copy stored in dataset/<stem>/<name>. If the
    same page is now rendered at a different resolution (e.g. after the PDF
    rendering change), boxes are rescaled instead of drifting off the text.
    """
    saved = DATASET_DIR / Path(name).stem / name
    if not saved.exists():
        return 1.0, 1.0
    try:
        with Image.open(saved) as old, Image.open(io.BytesIO(current_data)) as new:
            (ow, oh), (nw, nh) = old.size, new.size
    except Exception:  # noqa: BLE001 — unreadable copy: keep coordinates as-is
        return 1.0, 1.0
    if not ow or not oh:
        return 1.0, 1.0
    return nw / ow, nh / oh


# ── Auto-annotation (HTR) ──────────────────────────────────────────

@app.route("/api/htr-status")
def htr_status():
    """Czy silnik rozpoznawania jest gotowy (model pobrany, serwer zywy)."""
    try:
        recognizer = auto_annotate.get_recognizer()
        ok, message = recognizer.available()
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "backend": "?", "message": str(e)})
    return jsonify({
        "ok": ok,
        "backend": recognizer.name,
        "model": auto_annotate.HTR_MODEL,
        "message": message,
    })


@app.route("/api/auto-annotate/<fid>", methods=["POST"])
def start_auto_annotate(fid):
    """Uruchamia pre-anotacje w tle; zwraca id zadania do odpytywania."""
    entry = image_store.get(fid)
    if not entry:
        return jsonify({"ok": False, "error": "Image not found"}), 404

    payload = request.get_json(silent=True) or {}
    recognizer = auto_annotate.get_recognizer(payload.get("backend"))
    ready, message = recognizer.available()
    if not ready:
        return jsonify({"ok": False, "error": message}), 503

    img = Image.open(io.BytesIO(entry["data"]))
    img.load()

    job_id = uuid4().hex[:12]
    progress = auto_annotate.AutoAnnotateProgress()
    with _jobs_lock:
        _jobs[job_id] = {"progress": progress, "cancel": False}

    def cancelled():
        job = _jobs.get(job_id)
        return bool(job and job["cancel"])

    def worker():
        try:
            auto_annotate.auto_annotate(img, recognizer, progress, cancelled)
        except Exception as e:  # noqa: BLE001
            progress.stage = "error"
            progress.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job": job_id})


@app.route("/api/auto-annotate/job/<job_id>")
def auto_annotate_progress(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Unknown job"}), 404
    return jsonify({"ok": True, **job["progress"].as_dict()})


@app.route("/api/auto-annotate/job/<job_id>/cancel", methods=["POST"])
def cancel_auto_annotate(job_id):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["cancel"] = True
    return jsonify({"ok": True})


@app.route("/api/recognize-box", methods=["POST"])
def recognize_box():
    """Odczytuje tekst z pojedynczej ramki.

    Używane po ręcznej korekcie ramki — poprawione kadrowanie zwykle
    wystarcza, by model odczytał linię bezbłędnie.
    """
    data = request.get_json(silent=True) or {}
    entry = image_store.get(data.get("id"))
    if not entry:
        return jsonify({"ok": False, "error": "Image not found"}), 404

    box = data.get("box") or {}
    try:
        x_min, y_min = int(box["x_min"]), int(box["y_min"])
        x_max, y_max = int(box["x_max"]), int(box["y_max"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid box"}), 400

    recognizer = auto_annotate.get_recognizer(data.get("backend"))
    ready, message = recognizer.available()
    if not ready:
        return jsonify({"ok": False, "error": message}), 503

    img = Image.open(io.BytesIO(entry["data"]))
    img.load()
    margin = auto_annotate.CROP_MARGIN
    crop = img.crop((
        max(0, x_min - margin),
        max(0, y_min - margin),
        min(img.width, x_max + margin),
        min(img.height, y_max + margin),
    ))

    try:
        label = recognizer.recognize(crop)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "label": label})


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
