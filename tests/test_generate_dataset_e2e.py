import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def _make_mini_templates(tmp_path: Path) -> Path:
    tdir = tmp_path / "templates"
    tdir.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "mini.png", "width": 800, "height": 400}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 20, 300, 30]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [40, 80, 600, 45]},
            {"id": 3, "image_id": 1, "category_id": 4, "bbox": [40, 160, 500, 40]},
        ],
        "categories": [
            {"id": 1, "name": "p"}, {"id": 2, "name": "n"}, {"id": 3, "name": "t"},
            {"id": 4, "name": "f"}, {"id": 5, "name": "mix"},
        ],
    }
    (tdir / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    page = tdir / "mini"
    page.mkdir()
    Image.new("RGB", (800, 400), (252, 250, 247)).save(page / "mini_blank.png")
    Image.new("RGB", (800, 400), (252, 250, 247)).save(page / "mini_partial.png")
    return tdir


def test_generator_end_to_end(tmp_path):
    tdir = _make_mini_templates(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "generate_yolo_dataset.py"),
         "--templates-dir", str(tdir),
         "--output-dir", str(out),
         "--variants-per-form", "6",
         "--font-dir", str(REPO / "resources" / "fonts"),
         "--resource-dir", str(REPO / "resources"),
         "--seed", "123",
         # Must not append a row to the repo's real registry.
         "--registry", "none"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr

    coco = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 6
    sources = {a["source"] for a in coco["annotations"]}
    assert "printed" in sources and "synthetic" in sources
    for a in coco["annotations"]:
        assert "text" in a and "source" in a
        # synthetic fills and stamp lines are the two sources we know the
        # exact wording of; printed/handwritten lines are boxes without a
        # transcription
        if a["source"] in ("synthetic", "stamp"):
            assert isinstance(a["text"], str) and a["text"]
        else:
            assert a["text"] is None

    with open(out / "ground_truth.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and set(rows[0].keys()) == {
        "filename", "label", "x_min", "y_min", "x_max", "y_max", "source", "text"}

    metas = sorted((out / "metadata").glob("*.json"))
    assert len(metas) == 6
    bases = {json.loads(m.read_text(encoding="utf-8"))["base"] for m in metas}
    assert bases == {"blank", "partial"}  # 6 variants, ~50/50 split with seed 123
    meta0 = json.loads(metas[0].read_text(encoding="utf-8"))
    assert "empty_field_prob" in meta0
    assert meta0["scan_augmentation"]["profile"] in (
        "clean_color", "grayscale", "photocopy")


def test_generator_does_not_touch_the_repo_registry(tmp_path):
    """A run must never append to docs/datasets.md unless asked to.

    The registry is committed to the repo; a test run that silently adds a row
    to it turns every test invocation into a working-tree change.
    """
    registry = REPO / "docs" / "datasets.md"
    before = registry.read_text(encoding="utf-8") if registry.exists() else None

    tdir = _make_mini_templates(tmp_path)
    out = tmp_path / "out_registry_check"
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "generate_yolo_dataset.py"),
         "--templates-dir", str(tdir),
         "--output-dir", str(out),
         "--variants-per-form", "1",
         "--font-dir", str(REPO / "resources" / "fonts"),
         "--resource-dir", str(REPO / "resources"),
         "--seed", "4",
         "--registry", "none"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr

    after = registry.read_text(encoding="utf-8") if registry.exists() else None
    assert after == before, "the run modified the repo's registry"
    # The card itself still lands next to the data.
    assert (out / "dataset_card.json").exists()


def test_registry_can_be_redirected(tmp_path):
    """--registry PATH writes the row somewhere harmless instead."""
    tdir = _make_mini_templates(tmp_path)
    out = tmp_path / "out_redirect"
    registry = tmp_path / "own_registry.md"
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "generate_yolo_dataset.py"),
         "--templates-dir", str(tdir),
         "--output-dir", str(out),
         "--variants-per-form", "1",
         "--font-dir", str(REPO / "resources" / "fonts"),
         "--resource-dir", str(REPO / "resources"),
         "--seed", "4",
         "--dataset-name", "redirected",
         "--registry", str(registry)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert registry.exists()
    assert "| redirected |" in registry.read_text(encoding="utf-8")
