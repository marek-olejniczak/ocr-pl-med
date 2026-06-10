"""Convert a PDF form to PNG image(s) for use with the labeling tool.

Uses pypdfium2 (no external dependencies — works on Windows out of the box).

Usage:
    python pdf_to_png.py path/to/form.pdf [--dpi 200] [--output-dir forms/]
"""

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF pages to PNG images.")
    parser.add_argument("pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="Render resolution (default: 200)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="forms",
        help="Output directory (default: forms/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    base = pdf_path.stem
    scale = args.dpi / 72.0  # PDF points → pixels

    for i, page in enumerate(pdf, 1):
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil()
        if len(pdf) == 1:
            out_path = out_dir / f"{base}.png"
        else:
            out_path = out_dir / f"{base}_p{i}.png"
        img.save(out_path)
        print(f"  Saved {out_path} ({img.width}x{img.height})")

    print(f"Done. {len(pdf)} page(s) converted to {out_dir}/")


if __name__ == "__main__":
    main()
