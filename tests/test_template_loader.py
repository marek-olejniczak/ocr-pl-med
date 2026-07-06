import json
from pathlib import Path

from PIL import Image

from template_loader import load_templates, TemplatePage, VALID_LABELS


def _make_templates_dir(tmp_path: Path) -> Path:
    """Two pages: page_a has an f-field and a _partial, page_b has neither."""
    coco = {
        "images": [
            {"id": 1, "file_name": "page_a.png", "width": 200, "height": 100},
            {"id": 2, "file_name": "page_b.png", "width": 200, "height": 100},
            {"id": 3, "file_name": "missing.png", "width": 200, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 20]},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [10, 40, 80, 20]},
            {"id": 3, "image_id": 2, "category_id": 3, "bbox": [5, 5, 50, 15]},
            {"id": 4, "image_id": 2, "category_id": 2, "bbox": [5, 30, 50, 15]},
            {"id": 5, "image_id": 2, "category_id": 5, "bbox": [5, 60, 50, 15]},
        ],
        "categories": [
            {"id": 1, "name": "p"}, {"id": 2, "name": "n"}, {"id": 3, "name": "t"},
            {"id": 4, "name": "f"}, {"id": 5, "name": "mix"},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    for stem, with_partial in [("page_a", True), ("page_b", False)]:
        d = tmp_path / stem
        d.mkdir()
        Image.new("RGB", (200, 100), "white").save(d / f"{stem}_blank.png")
        if with_partial:
            Image.new("RGB", (200, 100), "white").save(d / f"{stem}_partial.png")
    # "missing" page dir intentionally absent — loader must skip it
    return tmp_path


def test_load_templates(tmp_path):
    pages = load_templates(_make_templates_dir(tmp_path))
    assert len(pages) == 2  # missing.png skipped
    by_name = {p.name: p for p in pages}

    a = by_name["page_a"]
    assert a.blank_path.name == "page_a_blank.png"
    assert a.partial_path is not None and a.partial_path.name == "page_a_partial.png"
    assert [f["label"] for f in a.fields] == ["p", "f"]
    # COCO xywh -> corner coords
    assert a.fields[0] == {"label": "p", "x_min": 10, "y_min": 10, "x_max": 90, "y_max": 30}

    b = by_name["page_b"]
    assert b.partial_path is None
    assert sorted(f["label"] for f in b.fields) == ["mix", "n", "t"]


def test_valid_labels_constant():
    assert VALID_LABELS == {"p", "n", "t", "f", "mix"}
