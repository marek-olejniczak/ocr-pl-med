"""Input loading: images and PDFs both become lists of PIL pages.

PDFs are rasterized page by page - works the same for scanned PDFs (embedded
images) and digital-native ones. Digital PDFs with a text layer could skip
OCR entirely via text extraction, but medical documents are scans, so that
path is out of scope.
"""

from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_DPI = 200  # ~1650x2300 px for A4, plenty for detection at imgsz 1024


def load_pages(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # pymupdf

        doc = fitz.open(path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=PDF_DPI)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height),
                                         pix.samples))
        doc.close()
        return pages
    if suffix in IMAGE_EXTS:
        return [Image.open(path).convert("RGB")]
    raise ValueError(f"unsupported file type: {suffix or path.name}")
