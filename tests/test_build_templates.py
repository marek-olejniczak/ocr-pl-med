import csv
import json
from pathlib import Path

from PIL import Image

from build_templates import build, map_old_label


def test_generic_labels():
    assert map_old_label("printed") == "p"
    assert map_old_label("text") == "t"
    assert map_old_label("number") == "n"


def test_legacy_field_labels():
    # numeric-ish legacy labels -> n
    for lbl in ["pesel", "pesel_grid", "full_date", "phone_num", "icd_10",
                "age", "day_and_month", "last_2_digits_year"]:
        assert map_old_label(lbl) == "n", lbl
    # text-ish legacy labels -> t
    for lbl in ["city", "name_and_surname", "address", "diagnosis",
                "approval", "hospital", "full_signature"]:
        assert map_old_label(lbl) == "t", lbl


def test_junk_labels_rejected():
    assert map_old_label("JĄDRA") is None
    assert map_old_label("label") is None
    assert map_old_label("") is None


def test_case_insensitive():
    assert map_old_label("  Printed ") == "p"


def test_build_merges_new_and_old_sources(tmp_path):
    """End-to-end test of build(): file copying, filtering, and re-id'ing."""
    # --- Arrange: miniature new-dataset layout ---
    new_dataset = tmp_path / "new_dataset"
    page_dir = new_dataset / "page1"
    page_dir.mkdir(parents=True)

    blank_img = Image.new("RGB", (100, 200), "white")
    blank_img.save(page_dir / "page1_blank.png")
    blank_img.save(page_dir / "page1_partial.png")
    # Files that must NOT be copied to the output template folder.
    blank_img.save(page_dir / "original.png")
    lines_dir = page_dir / "lines"
    lines_dir.mkdir()
    blank_img.save(lines_dir / "line_0.png")

    new_annotations = {
        "images": [
            {"id": 10, "file_name": "page1.png", "width": 100, "height": 200},
        ],
        "annotations": [
            # Valid annotation -> kept.
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4]},
            # Degenerate bbox (zero width) -> must be skipped.
            {"id": 2, "image_id": 10, "category_id": 1, "bbox": [5, 5, 0, 10]},
        ],
        "categories": [{"id": 1, "name": "p"}],
    }
    with open(new_dataset / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(new_annotations, f)

    # --- Arrange: miniature old CSV + images layout ---
    old_images = tmp_path / "old_images"
    old_images.mkdir()
    Image.new("RGB", (50, 60), "white").save(old_images / "old_page.png")

    old_csv = tmp_path / "old_annotations.csv"
    with open(old_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "label", "x_min", "y_min", "x_max", "y_max", "crop"]
        )
        # Valid row -> kept.
        writer.writerow(["old_page.png", "printed", "10", "10", "40", "30", ""])
        # Junk label -> excluded.
        writer.writerow(["old_page.png", "JĄDRA", "10", "10", "40", "30", ""])
        # Degenerate bbox (x_max == x_min) -> excluded.
        writer.writerow(["old_page.png", "text", "20", "20", "20", "50", ""])

    output = tmp_path / "templates"

    # --- Act ---
    build(new_datasets=[new_dataset], old_images=old_images, old_csv=old_csv,
          output=output)

    # --- Assert: only _blank/_partial copied for the new page ---
    new_out_dir = output / "page1"
    assert sorted(p.name for p in new_out_dir.iterdir()) == [
        "page1_blank.png",
        "page1_partial.png",
    ]

    # --- Assert: old PNG lands as <stem>/<stem>_blank.png ---
    old_out_dir = output / "old_page"
    assert (old_out_dir / "old_page_blank.png").exists()

    # --- Assert: annotations.json content ---
    with open(output / "annotations.json", "r", encoding="utf-8") as f:
        coco = json.load(f)

    assert [c["id"] for c in coco["categories"]] == [1, 2, 3, 4, 5]
    assert [c["name"] for c in coco["categories"]] == ["p", "n", "t", "f", "mix"]

    assert [im["id"] for im in coco["images"]] == [1, 2]

    # Only the two valid annotations survive: the new-dataset degenerate bbox
    # and the old-CSV junk label + degenerate bbox rows are all excluded.
    assert len(coco["annotations"]) == 2
    bboxes = {tuple(a["bbox"]) for a in coco["annotations"]}
    assert (1, 2, 3, 4) in bboxes
    assert (10, 10, 30, 20) in bboxes


def _make_coco_dataset(root: Path, page: str, category_order: list[str]) -> None:
    """Write a one-page COCO dataset whose category ids follow the given order."""
    page_dir = root / page
    page_dir.mkdir(parents=True)
    Image.new("RGB", (80, 120), "white").save(page_dir / f"{page}_blank.png")
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(category_order)]
    label_id = next(c["id"] for c in categories if c["name"] == "t")
    payload = {
        "images": [{"id": 7, "file_name": f"{page}.png", "width": 80, "height": 120}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": label_id, "bbox": [4, 5, 20, 10]}
        ],
        "categories": categories,
    }
    with open(root / "annotations.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_multiple_datasets_merge_and_match_categories_by_name(tmp_path):
    """Two source PDFs merge, even with different category id orderings."""
    first = tmp_path / "pdf_one"
    second = tmp_path / "pdf_two"
    _make_coco_dataset(first, "alpha", ["p", "n", "t", "f", "mix"])
    _make_coco_dataset(second, "beta", ["mix", "p", "t", "n", "f"])

    old_images = tmp_path / "old_images"
    old_images.mkdir()
    old_csv = tmp_path / "old.csv"
    with open(old_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            ["filename", "label", "x_min", "y_min", "x_max", "y_max", "crop"]
        )

    output = tmp_path / "templates"
    build(new_datasets=[first, second], old_images=old_images, old_csv=old_csv,
          output=output)

    assert (output / "alpha" / "alpha_blank.png").exists()
    assert (output / "beta" / "beta_blank.png").exists()

    with open(output / "annotations.json", "r", encoding="utf-8") as f:
        coco = json.load(f)

    assert [im["file_name"] for im in coco["images"]] == ["alpha.png", "beta.png"]
    assert [im["id"] for im in coco["images"]] == [1, 2]
    assert [a["id"] for a in coco["annotations"]] == [1, 2]
    # Both pages labelled "t" despite the second dataset numbering it differently.
    t_id = next(c["id"] for c in coco["categories"] if c["name"] == "t")
    assert {a["category_id"] for a in coco["annotations"]} == {t_id}


def test_duplicate_page_name_is_skipped_not_overwritten(tmp_path):
    """A page name may only be claimed once, so data cannot be clobbered."""
    first = tmp_path / "pdf_one"
    second = tmp_path / "pdf_two"
    _make_coco_dataset(first, "same_name", ["p", "n", "t", "f", "mix"])
    _make_coco_dataset(second, "same_name", ["p", "n", "t", "f", "mix"])

    old_images = tmp_path / "old_images"
    old_images.mkdir()
    old_csv = tmp_path / "old.csv"
    with open(old_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            ["filename", "label", "x_min", "y_min", "x_max", "y_max", "crop"]
        )

    output = tmp_path / "templates"
    build(new_datasets=[first, second], old_images=old_images, old_csv=old_csv,
          output=output)

    with open(output / "annotations.json", "r", encoding="utf-8") as f:
        coco = json.load(f)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
