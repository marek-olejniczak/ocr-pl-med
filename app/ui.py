"""Gradio front for the document -> text pipeline.

Start at least one OCR service first (from benchmark/):
    docker compose up -d tesseract-pol

Then (from app/):
    python ui.py --weights ../best_iou_median.pt

The OCR model is picked from a dropdown in the UI (every service from
benchmark/docker-compose.yml, with a live up/down marker). The line
detector comes from --weights and takes any ultralytics checkpoint
(YOLOv8 / YOLO11 / RT-DETR from the line benchmark).
"""

import argparse

import gradio as gr
import requests
from PIL import ImageDraw

import documents
import pipeline
import preprocess
import services
from detectors import UltralyticsDetector
from ocr_client import OCRClient


def annotate(image, results):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, r in enumerate(results, 1):
        x, y, w, h = r.bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x + 3, y - 14), str(i), fill="red")
    return out


def build_app(detector, registry, default_service,
              geometric_fn=None, photometric_fn=None):
    clients = {}   # service name -> OCRClient
    warmed = set()  # services already /load-ed in this process

    def get_ocr(name):
        url = registry[name]
        if name not in clients:
            clients[name] = OCRClient(url)
        ocr = clients[name]
        try:
            ocr.health()
        except requests.RequestException:
            raise gr.Error(
                f"{name} is not answering at {url} - start it first:\n"
                f"cd benchmark && docker compose up -d {name}")
        if name not in warmed:
            ocr.load()
            warmed.add(name)
        return ocr

    def service_choices():
        return [(f"{'●' if services.probe(url) else '○'} {name}", name)
                for name, url in registry.items()]

    def refresh(current):
        return gr.Dropdown(choices=service_choices(), value=current)

    def process(file, model_name):
        if file is None:
            raise gr.Error("Upload a document (image or PDF) first.")
        try:
            pages = documents.load_pages(file)
        except ValueError as e:
            raise gr.Error(str(e))
        ocr = get_ocr(model_name)

        annotated, prep_views, texts, table = [], [], [], []
        for pno, page in enumerate(pages, 1):
            # geometric stage is global: its output is the base image for
            # detection, crops and display alike
            base = geometric_fn(page) if geometric_fn else page
            # photometric stage feeds the detector only; run it once and
            # reuse for display (bind as default arg - a bare closure would
            # capture the loop variable)
            det_input = photometric_fn(base) if photometric_fn else None
            reuse = ((lambda _img, d=det_input: d)
                     if det_input is not None else None)
            try:
                results, text = pipeline.run(base, detector, ocr,
                                             preprocess=reuse)
            except requests.RequestException as e:
                raise gr.Error(f"OCR service unreachable: {e}")
            annotated.append(annotate(base, results))
            if det_input is not None:
                prep_views.append(det_input)
            texts.append(text)
            table += [[pno, i, f"{r.score:.2f}", r.text]
                      for i, r in enumerate(results, 1)]
        return annotated, prep_views or None, "\n\n".join(texts), table

    with gr.Blocks(title="OCR pipeline") as demo:
        gr.Markdown("# Document OCR pipeline\n"
                    "upload -> preprocessing -> line detection -> per-line OCR")
        with gr.Row():
            inp = gr.File(label="Document (image or PDF)",
                          file_types=[".pdf", *sorted(documents.IMAGE_EXTS)])
            with gr.Column():
                model_dd = gr.Dropdown(choices=service_choices(),
                                       value=default_service,
                                       label="OCR model (● running)")
                refresh_btn = gr.Button("Refresh status", size="sm")
        btn = gr.Button("Run", variant="primary")
        outp = gr.Gallery(label="Detected lines", columns=2)
        text = gr.Textbox(label="Recognized text", lines=10)
        table = gr.Dataframe(headers=["page", "line", "conf", "text"],
                             label="Per-line results")
        with gr.Accordion("Detector input (preprocessed)", open=False):
            prep_view = gr.Gallery(label="What the detector sees", columns=2)
        refresh_btn.click(refresh, inputs=model_dd, outputs=model_dd)
        btn.click(process, inputs=[inp, model_dd],
                  outputs=[outp, prep_view, text, table])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="../best_iou_median.pt")
    ap.add_argument("--ocr-url", default="http://localhost:8007",
                    help="default/extra OCR service URL; matched against the "
                         "compose registry, unknown URLs show up as 'custom'")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-geometric", action="store_true",
                    help="skip deskew (the global stage)")
    ap.add_argument("--no-photometric", action="store_true",
                    help="feed the detector the base image instead of the "
                         "photometric variant")
    args = ap.parse_args()

    detector = UltralyticsDetector(args.weights, imgsz=args.imgsz,
                                   conf=args.conf, device=args.device)
    try:
        registry = services.load_registry()
    except OSError:
        print("WARNING: benchmark/docker-compose.yml not found - "
              "only --ocr-url is offered")
        registry = {}
    default = next((n for n, u in registry.items() if u == args.ocr_url), None)
    if default is None:
        registry["custom"] = args.ocr_url
        default = "custom"
    if not services.probe(registry[default]):
        print(f"WARNING: no OCR service at {registry[default]} - "
              "start one before running the pipeline, or pick another "
              "model in the UI")

    geometric_fn = None if args.no_geometric else preprocess.geometric
    photometric_fn = None if args.no_photometric else preprocess.photometric
    build_app(detector, registry, default, geometric_fn,
              photometric_fn).launch(server_port=args.port)


if __name__ == "__main__":
    main()
