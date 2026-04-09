"""Web app for interactive augmentation preview.

Provides a UI with text input and slider controls for each augmentation
parameter. Renders a live preview image on each parameter change.
"""

import base64
import io
import random
import sys
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vocabulary import Vocabulary
from renderer import render_text, find_fonts
from char_renderer import render_text_per_char
from transforms import (
    AugmentConfig,
    CharTransformConfig,
    LineTransformConfig,
    PaperTransformConfig,
    ScanTransformConfig,
    TransformPipeline,
)

app = Flask(__name__)

# Load resources once at startup
RESOURCE_DIR = str(Path(__file__).parent.parent / "resources")
FONT_DIR = str(Path(__file__).parent.parent / "resources" / "fonts")
vocab = Vocabulary(RESOURCE_DIR)
fonts = find_fonts(FONT_DIR)
font_names = {Path(f).name: f for f in fonts}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Augmentation Preview</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }
        h1 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #fff;
        }
        .layout {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 24px;
        }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .section {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 14px;
        }
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .section h3 {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
        }
        .toggle {
            appearance: none;
            width: 36px;
            height: 20px;
            background: #333;
            border-radius: 10px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }
        .toggle:checked { background: #4a9eff; }
        .toggle::after {
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            background: #fff;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s;
        }
        .toggle:checked::after { transform: translateX(16px); }
        .param {
            margin-bottom: 8px;
        }
        .param:last-child { margin-bottom: 0; }
        .param label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #aaa;
            margin-bottom: 3px;
        }
        .param label .val {
            color: #4a9eff;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        input[type="range"] {
            width: 100%;
            height: 4px;
            appearance: none;
            background: #333;
            border-radius: 2px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 14px;
            height: 14px;
            background: #4a9eff;
            border-radius: 50%;
            cursor: pointer;
        }
        .text-input {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 14px;
        }
        .text-input input, .text-input select {
            width: 100%;
            padding: 8px 10px;
            background: #111;
            border: 1px solid #333;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .text-input select { cursor: pointer; }
        .text-input input:focus, .text-input select:focus {
            outline: none;
            border-color: #4a9eff;
        }
        .btn-row {
            display: flex;
            gap: 8px;
        }
        .btn {
            flex: 1;
            padding: 8px;
            border: 1px solid #333;
            background: #222;
            color: #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.15s;
        }
        .btn:hover { background: #2a2a2a; border-color: #4a9eff; }
        .btn.primary { background: #4a9eff; border-color: #4a9eff; color: #fff; }
        .btn.primary:hover { background: #3a8eef; }
        .preview-area {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
        }
        .preview-area img {
            max-width: 100%;
            image-rendering: auto;
            border: 1px solid #333;
            border-radius: 4px;
        }
        .preview-label {
            font-size: 11px;
            color: #666;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .preview-pair {
            display: flex;
            flex-direction: column;
            gap: 16px;
            align-items: center;
            width: 100%;
        }
        .preview-item {
            text-align: center;
        }
        .loading {
            color: #666;
            font-size: 14px;
        }
        .font-size-row {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .font-size-row input[type="range"] { flex: 1; }
        .font-size-row .val {
            color: #4a9eff;
            font-weight: 600;
            font-size: 12px;
            min-width: 30px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Augmentation Preview</h1>
        <div class="layout">
            <div class="controls">
                <div class="text-input">
                    <input type="text" id="text" value="Pacjent: Jan Kowalski" placeholder="Type text...">
                    <select id="font">
                        {% for name in font_names %}
                        <option value="{{ name }}">{{ name }}</option>
                        {% endfor %}
                    </select>
                    <div class="font-size-row">
                        <input type="range" id="font_size" min="24" max="96" value="48">
                        <span class="val" id="font_size_val">48</span>
                    </div>
                    <div class="btn-row" style="margin-top: 8px;">
                        <button class="btn" onclick="randomText()">Random Text</button>
                        <button class="btn" onclick="randomPhrase()">Random Phrase</button>
                        <button class="btn primary" onclick="render()">Render</button>
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">
                        <h3>Character</h3>
                        <input type="checkbox" class="toggle" id="char_enabled" checked onchange="render()">
                    </div>
                    <div class="param">
                        <label>Rotation <span class="val" id="char_rotation_val">5.0</span>&deg;</label>
                        <input type="range" id="char_rotation" min="0" max="20" step="0.5" value="5" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Scale min <span class="val" id="char_scale_min_val">0.92</span></label>
                        <input type="range" id="char_scale_min" min="0.7" max="1.0" step="0.01" value="0.92" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Scale max <span class="val" id="char_scale_max_val">1.08</span></label>
                        <input type="range" id="char_scale_max" min="1.0" max="1.3" step="0.01" value="1.08" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Stroke thicken <input type="checkbox" id="char_stroke" checked onchange="render()"></label>
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">
                        <h3>Line / Word</h3>
                        <input type="checkbox" class="toggle" id="line_enabled" checked onchange="render()">
                    </div>
                    <div class="param">
                        <label>Baseline wander <span class="val" id="line_wander_val">3.0</span>px</label>
                        <input type="range" id="line_wander" min="0" max="15" step="0.5" value="3" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Spacing jitter <span class="val" id="line_spacing_val">2.0</span>px</label>
                        <input type="range" id="line_spacing" min="0" max="10" step="0.5" value="2" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Slant <span class="val" id="line_slant_val">12.0</span>&deg;</label>
                        <input type="range" id="line_slant" min="0" max="30" step="0.5" value="12" oninput="updateVal(this); render()">
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">
                        <h3>Paper</h3>
                        <input type="checkbox" class="toggle" id="paper_enabled" checked onchange="render()">
                    </div>
                    <div class="param">
                        <label>Texture <input type="checkbox" id="paper_texture" checked onchange="render()"></label>
                    </div>
                    <div class="param">
                        <label>Yellowing <span class="val" id="paper_yellow_val">0.10</span></label>
                        <input type="range" id="paper_yellow" min="0" max="0.5" step="0.01" value="0.1" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Coffee stain prob <span class="val" id="paper_coffee_val">0.05</span></label>
                        <input type="range" id="paper_coffee" min="0" max="1" step="0.01" value="0.05" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Fold mark prob <span class="val" id="paper_fold_val">0.10</span></label>
                        <input type="range" id="paper_fold" min="0" max="1" step="0.01" value="0.1" oninput="updateVal(this); render()">
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">
                        <h3>Scan</h3>
                        <input type="checkbox" class="toggle" id="scan_enabled" checked onchange="render()">
                    </div>
                    <div class="param">
                        <label>Noise sigma <span class="val" id="scan_noise_val">5.0</span></label>
                        <input type="range" id="scan_noise" min="0" max="30" step="0.5" value="5" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Blur radius <span class="val" id="scan_blur_val">0.50</span></label>
                        <input type="range" id="scan_blur" min="0" max="3" step="0.1" value="0.5" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Brightness var <span class="val" id="scan_bright_val">0.10</span></label>
                        <input type="range" id="scan_bright" min="0" max="0.5" step="0.01" value="0.1" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>Page rotation <span class="val" id="scan_rotation_val">1.5</span>&deg;</label>
                        <input type="range" id="scan_rotation" min="0" max="10" step="0.5" value="1.5" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>JPEG quality min <span class="val" id="scan_jpeg_min_val">60</span></label>
                        <input type="range" id="scan_jpeg_min" min="10" max="95" step="1" value="60" oninput="updateVal(this); render()">
                    </div>
                    <div class="param">
                        <label>JPEG quality max <span class="val" id="scan_jpeg_max_val">95</span></label>
                        <input type="range" id="scan_jpeg_max" min="50" max="100" step="1" value="95" oninput="updateVal(this); render()">
                    </div>
                </div>
            </div>

            <div class="preview-area" id="preview">
                <div class="preview-pair">
                    <div class="preview-item">
                        <div class="preview-label">Clean</div>
                        <img id="img_clean" src="">
                    </div>
                    <div class="preview-item">
                        <div class="preview-label">Augmented</div>
                        <img id="img_aug" src="">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let debounceTimer = null;

        function updateVal(el) {
            const valEl = document.getElementById(el.id + '_val');
            if (valEl) {
                const v = parseFloat(el.value);
                valEl.textContent = Number.isInteger(v) ? v : v.toFixed(v < 1 ? 2 : 1);
            }
        }

        // Initialize all value displays
        document.querySelectorAll('input[type="range"]').forEach(el => updateVal(el));

        function getParams() {
            return {
                text: document.getElementById('text').value,
                font: document.getElementById('font').value,
                font_size: parseInt(document.getElementById('font_size').value),
                char_enabled: document.getElementById('char_enabled').checked,
                char_rotation: parseFloat(document.getElementById('char_rotation').value),
                char_scale_min: parseFloat(document.getElementById('char_scale_min').value),
                char_scale_max: parseFloat(document.getElementById('char_scale_max').value),
                char_stroke: document.getElementById('char_stroke').checked,
                line_enabled: document.getElementById('line_enabled').checked,
                line_wander: parseFloat(document.getElementById('line_wander').value),
                line_spacing: parseFloat(document.getElementById('line_spacing').value),
                line_slant: parseFloat(document.getElementById('line_slant').value),
                paper_enabled: document.getElementById('paper_enabled').checked,
                paper_texture: document.getElementById('paper_texture').checked,
                paper_yellow: parseFloat(document.getElementById('paper_yellow').value),
                paper_coffee: parseFloat(document.getElementById('paper_coffee').value),
                paper_fold: parseFloat(document.getElementById('paper_fold').value),
                scan_enabled: document.getElementById('scan_enabled').checked,
                scan_noise: parseFloat(document.getElementById('scan_noise').value),
                scan_blur: parseFloat(document.getElementById('scan_blur').value),
                scan_bright: parseFloat(document.getElementById('scan_bright').value),
                scan_rotation: parseFloat(document.getElementById('scan_rotation').value),
                scan_jpeg_min: parseInt(document.getElementById('scan_jpeg_min').value),
                scan_jpeg_max: parseInt(document.getElementById('scan_jpeg_max').value),
            };
        }

        function render() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(doRender, 150);
        }

        async function doRender() {
            const params = getParams();
            try {
                const res = await fetch('/render', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params),
                });
                const data = await res.json();
                document.getElementById('img_clean').src = 'data:image/png;base64,' + data.clean;
                document.getElementById('img_aug').src = 'data:image/png;base64,' + data.augmented;
            } catch (e) {
                console.error(e);
            }
        }

        async function randomText() {
            const res = await fetch('/random_text');
            const data = await res.json();
            document.getElementById('text').value = data.text;
            render();
        }

        async function randomPhrase() {
            const res = await fetch('/random_phrase');
            const data = await res.json();
            document.getElementById('text').value = data.text;
            render();
        }

        // Listen for Enter in text input
        document.getElementById('text').addEventListener('keydown', e => {
            if (e.key === 'Enter') render();
        });

        // Render on load
        render();
    </script>
</body>
</html>
"""


def img_to_base64(img) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE, font_names=sorted(font_names.keys()))


@app.route("/render", methods=["POST"])
def render_endpoint():
    """Render text with the given augmentation parameters."""
    p = request.json

    text = p.get("text", "Przykład tekstu")
    font_file = font_names.get(p.get("font", ""), fonts[0] if fonts else "")
    font_size = int(p.get("font_size", 48))

    config = AugmentConfig(
        char=CharTransformConfig(
            enabled=p.get("char_enabled", True),
            rotation_max_deg=float(p.get("char_rotation", 5)),
            scale_min=float(p.get("char_scale_min", 0.92)),
            scale_max=float(p.get("char_scale_max", 1.08)),
            stroke_variation=p.get("char_stroke", True),
        ),
        line=LineTransformConfig(
            enabled=p.get("line_enabled", True),
            baseline_wander_amplitude=float(p.get("line_wander", 3)),
            spacing_jitter_px=float(p.get("line_spacing", 2)),
            slant_max_deg=float(p.get("line_slant", 12)),
        ),
        paper=PaperTransformConfig(
            enabled=p.get("paper_enabled", True),
            texture=p.get("paper_texture", True),
            yellowing_intensity=float(p.get("paper_yellow", 0.1)),
            coffee_stain_prob=float(p.get("paper_coffee", 0.05)),
            fold_mark_prob=float(p.get("paper_fold", 0.1)),
        ),
        scan=ScanTransformConfig(
            enabled=p.get("scan_enabled", True),
            noise_sigma=float(p.get("scan_noise", 5)),
            blur_radius=float(p.get("scan_blur", 0.5)),
            brightness_variation=float(p.get("scan_bright", 0.1)),
            rotation_max_deg=float(p.get("scan_rotation", 1.5)),
            jpeg_quality_min=int(p.get("scan_jpeg_min", 60)),
            jpeg_quality_max=int(p.get("scan_jpeg_max", 95)),
        ),
    )

    pipeline = TransformPipeline(config)

    # Clean render
    clean_img, _ = render_text(text, font_file, font_size)

    # Augmented render
    aug_img, _ = render_text_per_char(text, font_file, font_size, config=config)
    aug_img = pipeline.apply(aug_img)

    return jsonify({
        "clean": img_to_base64(clean_img),
        "augmented": img_to_base64(aug_img),
    })


@app.route("/random_text")
def random_text_endpoint():
    """Return a random text sample."""
    text, _ = vocab.get_random_text()
    return jsonify({"text": text})


@app.route("/random_phrase")
def random_phrase_endpoint():
    """Return a random phrase."""
    text, _ = vocab.get_random_phrase()
    return jsonify({"text": text})


if __name__ == "__main__":
    print("Starting augmentation preview server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
