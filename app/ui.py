"""Gradio front for the document -> text pipeline.

Start an OCR service first (from benchmark/):
    docker compose up -d tesseract-pol

Then (from app/):
    python ui.py --weights ../best_iou_median.pt --ocr-url http://localhost:8007

Swapping models: --weights takes any ultralytics checkpoint (YOLOv8 / YOLO11 /
RT-DETR from the line benchmark), --ocr-url takes any service from the OCR
benchmark since they all share one contract.
"""

import argparse

import gradio as gr
import requests
from PIL import ImageDraw

import documents
import pipeline
import preprocess
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


def build_app(detector, ocr, geometric_fn=None, photometric_fn=None):
    def process(file):
        if file is None:
            raise gr.Error("Upload a document (image or PDF) first.")
        try:
            pages = documents.load_pages(file)
        except ValueError as e:
            raise gr.Error(str(e))

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
            outp = gr.Gallery(label="Detected lines", columns=2)
        btn = gr.Button("Run", variant="primary")
        text = gr.Textbox(label="Recognized text", lines=10)
        table = gr.Dataframe(headers=["page", "line", "conf", "text"],
                             label="Per-line results")
        with gr.Accordion("Detector input (preprocessed)", open=False):
            prep_view = gr.Gallery(label="What the detector sees", columns=2)
        btn.click(process, inputs=inp, outputs=[outp, prep_view, text, table])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="../best_iou_median.pt")
    ap.add_argument("--ocr-url", default="http://localhost:8007")
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
    ocr = OCRClient(args.ocr_url)
    geometric_fn = None if args.no_geometric else preprocess.geometric
    photometric_fn = None if args.no_photometric else preprocess.photometric
    try:
        ocr.health()
        ocr.load()
    except requests.RequestException:
        print(f"WARNING: no OCR service at {args.ocr_url} - "
              "start one before running the pipeline (see module docstring)")

    build_app(detector, ocr, geometric_fn,
              photometric_fn).launch(server_port=args.port)


if __name__ == "__main__":
    main()
