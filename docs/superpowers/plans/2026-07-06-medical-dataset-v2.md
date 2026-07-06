# Medical Dataset v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Przestawić generator syntetycznych formularzy na nowe szablony COCO (labele p/f/t/n/mix, podkłady `_blank`/`_partial`), wypełniać pola prawdziwym polskim słownictwem medycznym z transkrypcją w ground truth, dodać wielolinijkowe pola / sumienność per formularz / 3 profile skanu, i posprzątać repo.

**Architecture:** Nowy moduł `template_loader.py` czyta ujednolicony katalog `templates/` (budowany jednorazowo przez `build_templates.py` z nowego datasetu + migracji starych CSV-labeli). Nowy moduł `field_content.py` generuje treść pól z `Vocabulary` z dopasowaniem do szerokości. `fill_form.py` traci pseudo-filler i legacy-labele, zyskuje wielolinijkowe wypełnianie, per-form empty-rate i profile skanu. `generate_yolo_dataset.py` orkiestruje: wybór podkładu, COCO z polami `text`/`source`.

**Tech Stack:** Python 3.13, Pillow, numpy, openpyxl (drug xlsx), pytest (nowy dev-dep). Windows/PowerShell — komendy podane dla Git Basha (repo działa z obu).

**Spec:** `docs/superpowers/specs/2026-07-06-medical-dataset-v2-design.md`

## Global Constraints

- Rotacji strony NIE ma i nie wraca — usuwamy parametr `enable_rotation` / flagę `--rotate` (geometria obrazu nigdy nie jest zmieniana po wypełnieniu).
- Bez pogrubiania kresek (zostaje wyłączone — `FONTS_NEEDING_THICKENING` pusty).
- Commity bez stopki `Co-Authored-By` (preferencja użytkownika).
- Stałe bez zmian: `MIN_FONT_SIZE=14`, `MAX_FONT_SIZE=40`, `FORM_FONT_SIZE_RANGE=(26,40)`, `PEN_FADE_PROB=0.15`, `V_OVERFLOW_FRAC=0.18`, `INK_BLACK=(20,20,28)`, `INK_BLUE=(28,42,120)`, cel wypełnienia szerokości 30–100%, hard cap 0.97.
- Wszystkie testy uruchamiane z roota repo: `python -m pytest tests/ -v` (conftest dodaje `src/` do sys.path).
- Pliki źródłowe: docstringi i komentarze po angielsku (jak istniejący kod).
- `dataset_30zPierwszegoPDF/` jest źródłem tylko do odczytu — niczego w nim nie zmieniamy.

---

### Task 1: Porządek w repo + szkielet testów

**Files:**
- Delete: `yolo_dataset/`, `yolo_dataset.zip`, `yolo_test/`, `training/`, `versions/`, `font_preview.png`, `SKIEROWANIE_DO_SZPITALA.pdf`, `annotations.csv`, `dataset/`
- Create: `.gitignore`, `tests/conftest.py`
- Modify: `requirements.txt` (dopisać `pytest`)

**Interfaces:**
- Consumes: nic
- Produces: działający `python -m pytest tests/` z roota; `output/` i `templates/` ignorowane przez git

- [ ] **Step 1: Usuń zatwierdzone artefakty (żaden nie jest śledzony w gicie)**

```bash
cd /c/Users/tomek/Desktop/text-gen
rm -rf yolo_dataset yolo_dataset.zip yolo_test training versions dataset
rm -f font_preview.png SKIEROWANIE_DO_SZPITALA.pdf annotations.csv
```

- [ ] **Step 2: Wypisz śledzone .pyc z gita**

```bash
git rm -r --cached src/__pycache__
```

Expected: lista `rm 'src/__pycache__/...'` (co najmniej char_renderer, transforms, vocabulary).

- [ ] **Step 3: Utwórz `.gitignore`**

```gitignore
__pycache__/
*.pyc
output/
templates/
*.zip
.pytest_cache/
```

(`templates/` jest w ignore, bo buduje się lokalnie skryptem z `dataset_30zPierwszegoPDF/` + `forms/segmentation/` — nie bloatujemy repo obrazami.)

- [ ] **Step 4: Dodaj pytest**

Dopisz linię `pytest` na końcu `requirements.txt`, potem:

```bash
pip install pytest
```

- [ ] **Step 5: Utwórz `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

- [ ] **Step 6: Sanity check pytest**

Run: `python -m pytest tests/ -v`
Expected: `no tests ran` (exit code 5) — kolektor działa, testów jeszcze nie ma.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt tests/conftest.py
git commit -m "Repo cleanup: remove stale generation artifacts, add .gitignore and pytest scaffold"
```

---

### Task 2: `template_loader.py` — czytanie katalogu szablonów COCO

**Files:**
- Create: `src/template_loader.py`
- Test: `tests/test_template_loader.py`

**Interfaces:**
- Consumes: strukturę katalogu `templates/` opisaną w specu (annotations.json COCO + `<stem>/<stem>_blank.png` [+ `_partial.png`])
- Produces:
  - `VALID_LABELS = {"p", "n", "t", "f", "mix"}`
  - `@dataclass TemplatePage: name: str, blank_path: Path, partial_path: Optional[Path], fields: list[dict]` — field dict: `{"label": str, "x_min": int, "y_min": int, "x_max": int, "y_max": int}`
  - `load_templates(templates_dir: Path) -> list[TemplatePage]`

- [ ] **Step 1: Napisz failujący test**

`tests/test_template_loader.py`:

```python
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
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_template_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'template_loader'`

- [ ] **Step 3: Napisz `src/template_loader.py`**

```python
"""Load form-template pages for the dataset generator.

A templates directory (built by build_templates.py) contains one COCO
annotations.json plus one subdirectory per page:

    templates/
      annotations.json
      <stem>/
        <stem>_blank.png     # all fill-in fields cleared (always present)
        <stem>_partial.png   # only f-fields keep their real handwriting (optional)

Labels: p (printed), t (text), n (number), mix (letters+digits),
f (real handwriting present in the original scan).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALID_LABELS = {"p", "n", "t", "f", "mix"}


@dataclass
class TemplatePage:
    name: str
    blank_path: Path
    partial_path: Optional[Path]
    fields: list[dict]  # {"label", "x_min", "y_min", "x_max", "y_max"}


def load_templates(templates_dir: Path) -> list[TemplatePage]:
    """Read annotations.json and resolve per-page image paths.

    Pages whose _blank.png is missing are skipped with a warning.
    Annotations with labels outside VALID_LABELS are ignored.
    """
    ann_path = templates_dir / "annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    fields_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        label = cat_names.get(ann["category_id"], "")
        if label not in VALID_LABELS:
            continue
        x, y, w, h = ann["bbox"]
        fields_by_image.setdefault(ann["image_id"], []).append({
            "label": label,
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x + w),
            "y_max": int(y + h),
        })

    pages: list[TemplatePage] = []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        page_dir = templates_dir / stem
        blank = page_dir / f"{stem}_blank.png"
        if not blank.exists():
            print(f"  SKIP template '{stem}': {blank} not found", file=sys.stderr)
            continue
        partial = page_dir / f"{stem}_partial.png"
        pages.append(TemplatePage(
            name=stem,
            blank_path=blank,
            partial_path=partial if partial.exists() else None,
            fields=fields_by_image.get(im["id"], []),
        ))
    return pages
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_template_loader.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/template_loader.py tests/test_template_loader.py
git commit -m "Add COCO template loader (p/f/t/n/mix labels, _blank/_partial bases)"
```

---

### Task 3: `build_templates.py` — budowa `templates/` (kopiowanie nowych + migracja starych)

**Files:**
- Create: `src/build_templates.py`
- Test: `tests/test_build_templates.py`

**Interfaces:**
- Consumes: `dataset_30zPierwszegoPDF/annotations.json` (COCO p/n/t/f/mix) + podfoldery; `forms/segmentation/images/*.png` + stary CSV `filename,label,x_min,y_min,x_max,y_max,crop`
- Produces:
  - katalog `templates/` w formacie czytanym przez `load_templates()` z Task 2
  - `map_old_label(label: str) -> Optional[str]` — mapowanie starych labeli na p/t/n
  - CLI: `python src/build_templates.py --new-dataset dataset_30zPierwszegoPDF --old-images forms/segmentation/images --old-csv "C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv" --output templates`

- [ ] **Step 1: Napisz failujący test mapowania labeli**

`tests/test_build_templates.py`:

```python
from build_templates import map_old_label


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
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_build_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_templates'`

- [ ] **Step 3: Napisz `src/build_templates.py`**

```python
"""Build the unified templates/ directory for the dataset generator.

Merges two sources into one COCO annotations.json + per-page folders:

1. The new labeled dataset (dataset_30zPierwszegoPDF): COCO annotations with
   p/n/t/f/mix categories and per-page folders containing _blank/_partial
   PNGs. Only _blank/_partial images are copied (originals and lines/ stay
   behind — they contain unmasked or redundant data).

2. Old templates (forms/segmentation/images) labeled via the flat CSV from
   labeling_tool.py. Labels are mapped printed->p, text->t, number->n (plus
   legacy field-specific labels). The template PNG is already an unfilled
   form, so it becomes its own _blank.

Usage:
    python src/build_templates.py \
        --new-dataset dataset_30zPierwszegoPDF \
        --old-images forms/segmentation/images \
        --old-csv "C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv" \
        --output templates
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "p"},
    {"id": 2, "name": "n"},
    {"id": 3, "name": "t"},
    {"id": 4, "name": "f"},
    {"id": 5, "name": "mix"},
]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}

# Old generic labels
_OLD_GENERIC = {"printed": "p", "text": "t", "number": "n"}
# Legacy field-specific labels from early labeling sessions
_OLD_NUMBERISH = {
    "pesel", "pesel_grid", "date", "full_date", "date_of_birth", "phone",
    "phone_num", "telefon", "icd_10", "icd10", "icd", "icd_code", "age",
    "lat", "year", "rok", "day_and_month", "last_2_digits_year",
}
_OLD_TEXTISH = {
    "city", "miasto", "name", "name_and_surname", "patient_name", "address",
    "adres", "diagnosis", "rozpoznanie", "approval", "hospital",
    "hospital_name", "szpital", "full_signature", "doctor", "doctor_name",
    "lekarz", "department", "oddzial",
}


def map_old_label(label: str) -> Optional[str]:
    """Map an old CSV label to the new p/t/n scheme; None for junk labels."""
    l = label.strip().lower()
    if l in _OLD_GENERIC:
        return _OLD_GENERIC[l]
    if l in _OLD_NUMBERISH:
        return "n"
    if l in _OLD_TEXTISH:
        return "t"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified templates/ directory.")
    parser.add_argument("--new-dataset", type=str, default="dataset_30zPierwszegoPDF")
    parser.add_argument("--old-images", type=str, default="forms/segmentation/images")
    parser.add_argument("--old-csv", type=str,
                        default="C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv")
    parser.add_argument("--output", type=str, default="templates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_out: dict = {"images": [], "annotations": [], "categories": CATEGORIES}
    next_image_id = 1
    next_ann_id = 1

    # --- 1) New dataset: copy _blank/_partial + re-id annotations ---
    new_root = Path(args.new_dataset)
    with open(new_root / "annotations.json", "r", encoding="utf-8") as f:
        src = json.load(f)
    src_cats = {c["id"]: c["name"] for c in src["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for a in src["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    n_new_pages = 0
    for im in src["images"]:
        stem = Path(im["file_name"]).stem
        src_dir = new_root / stem
        blank_src = src_dir / f"{stem}_blank.png"
        if not blank_src.exists():
            print(f"  SKIP {stem}: no _blank.png in source", file=sys.stderr)
            continue
        dst_dir = out_dir / stem
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(blank_src, dst_dir / blank_src.name)
        partial_src = src_dir / f"{stem}_partial.png"
        if partial_src.exists():
            shutil.copy2(partial_src, dst_dir / partial_src.name)

        image_id = next_image_id
        next_image_id += 1
        coco_out["images"].append({
            "id": image_id,
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        })
        for a in anns_by_image.get(im["id"], []):
            name = src_cats.get(a["category_id"])
            if name not in CAT_ID:
                continue
            x, y, w, h = a["bbox"]
            coco_out["annotations"].append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": CAT_ID[name],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            next_ann_id += 1
        n_new_pages += 1

    # --- 2) Old templates: CSV labels -> p/t/n, template PNG becomes _blank ---
    old_images = Path(args.old_images)
    rows_by_file: dict[str, list[dict]] = {}
    with open(args.old_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_file.setdefault(row["filename"].strip(), []).append(row)

    n_old_pages = 0
    n_skipped_labels = 0
    for filename, rows in sorted(rows_by_file.items()):
        img_src = old_images / filename
        if not img_src.exists():
            print(f"  SKIP old '{filename}': not in {old_images}", file=sys.stderr)
            continue
        stem = Path(filename).stem
        dst_dir = out_dir / stem
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(img_src, dst_dir / f"{stem}_blank.png")
        with Image.open(img_src) as img:
            width, height = img.size

        image_id = next_image_id
        next_image_id += 1
        coco_out["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        for row in rows:
            label = map_old_label(row["label"])
            if label is None:
                n_skipped_labels += 1
                continue
            x1, y1 = int(row["x_min"]), int(row["y_min"])
            x2, y2 = int(row["x_max"]), int(row["y_max"])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            coco_out["annotations"].append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": CAT_ID[label],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            next_ann_id += 1
        n_old_pages += 1

    with open(out_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False)

    print(f"Done. {n_new_pages} new + {n_old_pages} migrated pages -> {out_dir}/")
    print(f"  {len(coco_out['annotations'])} annotations total")
    if n_skipped_labels:
        print(f"  Skipped {n_skipped_labels} rows with unknown labels")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_build_templates.py -v`
Expected: 4 passed

- [ ] **Step 5: Zbuduj templates/ naprawdę**

```bash
cd /c/Users/tomek/Desktop/text-gen
python src/build_templates.py
```

Expected: `Done. 27 new + 13 migrated pages -> templates/`, ~2320 anotacji (1770 nowych + ~553 starych), `Skipped 2 rows with unknown labels` (JĄDRA, label). Uwaga: stary CSV ma też wiersz dla `anatomia_zmysly_1.png` z pojedynczym junk-labelem — jeśli strona wyjdzie z 0 anotacji, to oczekiwane (loader i tak ją obsłuży, a generator wygeneruje tylko czysty podkład — nieszkodliwe; liczba stron może być 14 zamiast 13).

- [ ] **Step 6: Zweryfikuj loaderem**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from template_loader import load_templates
pages = load_templates(Path('templates'))
n_partial = sum(1 for p in pages if p.partial_path)
print(len(pages), 'pages,', n_partial, 'with _partial')
print(sum(len(p.fields) for p in pages), 'fields')
"
```

Expected: ~40 stron, ~19 z `_partial` (strony KARTA z polami f), ~2320 pól.

- [ ] **Step 7: Commit**

```bash
git add src/build_templates.py tests/test_build_templates.py
git commit -m "Add build_templates: merge new COCO dataset with migrated legacy CSV labels"
```

---

### Task 4: `field_content.py` — treść medyczna dopasowana do szerokości

**Files:**
- Create: `src/field_content.py`
- Test: `tests/test_field_content.py`

**Interfaces:**
- Consumes: `Vocabulary.get_random_text(category) -> (str, str)` z `src/vocabulary.py`
- Produces:
  - `generate_field_content(kind: str, vocab: Vocabulary, measure: Callable[[str], float], bbox_w: int) -> str` — kind ∈ {"t","f","n","mix"}; `measure(text)` zwraca szerokość renderu w px (caller wstrzykuje font+stretch+tracking); wynik mieści się w `0.97*bbox_w`, celuje w 30–100% szerokości
  - `_shrink_to_fit(text: str, measure, cap: float) -> str` (używane też w teście)

- [ ] **Step 1: Napisz failujący test**

`tests/test_field_content.py`:

```python
import random

import pytest

from field_content import generate_field_content, _shrink_to_fit


class FakeVocab:
    """Deterministic stand-in for Vocabulary — no resource loading."""

    SAMPLES = {
        "diagnosis": "nadciśnienie tętnicze",
        "icd_description": "Ostre zakażenie górnych dróg oddechowych",
        "drug": "Amoksycylina",
        "dosage": "2x dziennie",
        "abbreviation": "Rp.",
        "department": "Oddział Kardiologii",
        "hospital_name": "Szpital Miejski w Radomiu",
        "approval": "Pilne badanie",
        "patient_name": "Anna Kowalska",
        "doctor_name": "lek. Jan Nowak",
        "address": "ul. Polna 12, 26-600 Radom",
        "city": "Radom",
        "pesel": "44051401359",
        "date": "12.03.2021",
        "phone": "601 202 303",
        "icd_code": "J45.0",
    }

    def get_random_text(self, category):
        return (self.SAMPLES[category], category)


def char_measure(text: str) -> float:
    return len(text) * 10.0  # 10 px per char


def test_shrink_word_wise_then_char_wise():
    assert _shrink_to_fit("abc def ghi", char_measure, cap=80.0) == "abc def"
    # single word wider than cap -> trimmed char by char
    assert _shrink_to_fit("abcdefghij", char_measure, cap=50.0) == "abcde"
    assert _shrink_to_fit("abc", char_measure, cap=5.0) == ""


@pytest.mark.parametrize("kind", ["t", "f", "n", "mix"])
def test_content_fits_hard_cap(kind):
    random.seed(7)
    vocab = FakeVocab()
    for _ in range(200):
        bbox_w = random.randint(60, 1200)
        text = generate_field_content(kind, vocab, char_measure, bbox_w)
        assert char_measure(text) <= bbox_w * 0.97 + 1e-6


def test_text_kind_uses_vocab_words():
    random.seed(3)
    vocab = FakeVocab()
    seen = " ".join(
        generate_field_content("t", vocab, char_measure, 900) for _ in range(50)
    )
    # real vocabulary content, not pseudo-Polish syllables
    known_fragments = ["nadciśnienie", "Amoksycylina", "Kowalska", "Oddział",
                       "Radom", "zakażenie", "Rp.", "Nowak", "Pilne", "Szpital"]
    assert any(frag in seen for frag in known_fragments)


def test_number_kind_is_digit_based():
    random.seed(11)
    vocab = FakeVocab()
    for _ in range(50):
        text = generate_field_content("n", vocab, char_measure, 600)
        if text:
            digits = sum(ch.isdigit() for ch in text)
            assert digits >= len(text) * 0.5, text


def test_mix_kind_has_letters_and_digits():
    random.seed(5)
    vocab = FakeVocab()
    combined = " ".join(
        generate_field_content("mix", vocab, char_measure, 700) for _ in range(50)
    )
    assert any(ch.isdigit() for ch in combined)
    assert any(ch.isalpha() for ch in combined)
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_field_content.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'field_content'`

- [ ] **Step 3: Napisz `src/field_content.py`**

```python
"""Field-content generation for form filling.

Replaces the old pseudo-Polish syllable filler: fields are filled with real
Polish medical vocabulary (the same images later train the OCR model, so the
written content is a training label, not throwaway filler).

Content per field kind:
    t / f -> ~70% medical terms (diagnoses, drugs, dosages, abbreviations,
             departments, ICD descriptions), ~30% personal data (names,
             addresses, cities) — real forms contain both
    n     -> realistic digit content: PESEL, dates, phones, NIP, digit runs
    mix   -> ICD-10 codes, dose strings (2x500mg), document numbers

Width fitting: the caller provides a `measure(text) -> px` callable (usually
font.getlength adjusted for glyph stretch and letter tracking). Content is
grown unit by unit toward a random target of 30-100% of the field width and
hard-capped at 97% so augmentation jitter can't push ink past the field edge.
"""

import random
from typing import Callable

from vocabulary import Vocabulary

# Probability that a text field draws medical content (vs personal data)
MEDICAL_TEXT_PROB = 0.7

_TEXT_MEDICAL = [
    "diagnosis", "icd_description", "drug", "dosage",
    "abbreviation", "department", "hospital_name", "approval",
]
_TEXT_PERSONAL = ["patient_name", "doctor_name", "address", "city"]

# Fill target: fraction of field width the written line should span
FILL_FRAC_MIN = 0.30
FILL_FRAC_MAX = 1.00
HARD_CAP_FRAC = 0.97


def _sample_text_unit(vocab: Vocabulary) -> str:
    pool = _TEXT_MEDICAL if random.random() < MEDICAL_TEXT_PROB else _TEXT_PERSONAL
    text, _ = vocab.get_random_text(random.choice(pool))
    return text


def _random_nip() -> str:
    digits = "".join(random.choice("0123456789") for _ in range(10))
    if random.random() < 0.5:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:8]}-{digits[8:]}"
    return digits


def _random_digit_run() -> str:
    return "".join(random.choice("0123456789") for _ in range(random.randint(1, 6)))


def _sample_number_unit(vocab: Vocabulary) -> str:
    r = random.random()
    if r < 0.20:
        return vocab.get_random_text("pesel")[0]
    if r < 0.45:
        return vocab.get_random_text("date")[0]
    if r < 0.65:
        return vocab.get_random_text("phone")[0]
    if r < 0.75:
        return _random_nip()
    return _random_digit_run()


def _random_doc_number() -> str:
    letters = "".join(random.choice("ABCDEFGHJKLMNPRSTUWXYZ") for _ in range(3))
    return letters + "".join(random.choice("0123456789") for _ in range(7))


def _random_dose() -> str:
    n = random.choice([1, 2, 3])
    amount = random.choice([50, 75, 100, 250, 500, 1000])
    return f"{n}x{amount}mg"


def _sample_mix_unit(vocab: Vocabulary) -> str:
    r = random.random()
    if r < 0.40:
        return vocab.get_random_text("icd_code")[0]
    if r < 0.70:
        return _random_dose()
    return _random_doc_number()


_UNIT_SAMPLERS = {
    "n": _sample_number_unit,
    "mix": _sample_mix_unit,
    # "t" and "f" both fall through to text units
}


def _shrink_to_fit(text: str, measure: Callable[[str], float], cap: float) -> str:
    """Trim text to fit under `cap` px: whole words first, then characters."""
    words = text.split(" ")
    while len(words) > 1 and measure(" ".join(words)) > cap:
        words.pop()
    text = " ".join(words)
    while text and measure(text) > cap:
        text = text[:-1].rstrip()
    return text


def generate_field_content(
    kind: str,
    vocab: Vocabulary,
    measure: Callable[[str], float],
    bbox_w: int,
) -> str:
    """Build field content spanning ~30-100% of the field width.

    Args:
        kind: Field label — "t"/"f" (text), "n" (digits), "mix".
        vocab: Loaded Vocabulary.
        measure: Callable returning the rendered pixel width of a string
            (must already account for glyph stretch and letter tracking).
        bbox_w: Field width in pixels.

    Returns:
        Content string whose measured width is <= 0.97 * bbox_w
        (empty string if even one character doesn't fit).
    """
    sampler = _UNIT_SAMPLERS.get(kind, _sample_text_unit)
    target = bbox_w * random.uniform(FILL_FRAC_MIN, FILL_FRAC_MAX)
    hard_cap = bbox_w * HARD_CAP_FRAC

    text = _shrink_to_fit(sampler(vocab), measure, hard_cap)
    if not text:
        return ""

    while True:
        candidate = text + " " + sampler(vocab)
        w = measure(candidate)
        if w > hard_cap:
            break
        text = candidate
        if w >= target:
            break
    return text
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_field_content.py -v`
Expected: 8 passed (3 shrink + 4 parametrized cap + 3 content)

- [ ] **Step 5: Commit**

```bash
git add src/field_content.py tests/test_field_content.py
git commit -m "Add medical field-content generator with width fitting"
```

---

### Task 5: `transforms.py` — efekty grayscale/ksero

**Files:**
- Modify: `src/transforms.py` (dopisać na końcu sekcji efektów skanu, po `jpeg_artifacts` ~linia 505)
- Test: `tests/test_scan_effects.py`

**Interfaces:**
- Consumes: nic nowego (PIL, numpy już importowane w transforms.py)
- Produces:
  - `to_grayscale(img: Image.Image) -> Image.Image` (RGB→L→RGB, rozmiar bez zmian)
  - `photocopy_contrast(img: Image.Image, low: int = 100, high: int = 190) -> Image.Image`
  - `salt_pepper_noise(img: Image.Image, amount: float = 0.005) -> Image.Image`
  - `toner_streak(img: Image.Image) -> Image.Image`

- [ ] **Step 1: Napisz failujący test**

`tests/test_scan_effects.py`:

```python
import numpy as np
from PIL import Image

from transforms import to_grayscale, photocopy_contrast, salt_pepper_noise, toner_streak


def _gradient_img(w=200, h=100):
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.merge("RGB", [Image.fromarray(arr)] * 3)


def test_to_grayscale_keeps_size_and_mode():
    img = Image.new("RGB", (120, 80), (28, 42, 120))  # blue ink color
    out = to_grayscale(img)
    assert out.size == (120, 80)
    assert out.mode == "RGB"
    r, g, b = out.getpixel((10, 10))
    assert r == g == b  # actually gray


def test_photocopy_contrast_crushes_extremes():
    out = photocopy_contrast(_gradient_img(), low=100, high=190)
    arr = np.array(out.convert("L"))
    assert arr[:, :70].max() == 0      # below low -> pure black
    assert arr[:, 160:].min() == 255   # above high -> pure white
    mid = arr[:, 100:140]
    assert 0 < mid.mean() < 255        # midtones ramp, not binarized


def test_salt_pepper_changes_some_pixels():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    out = salt_pepper_noise(img, amount=0.01)
    arr = np.array(out)
    n_black = (arr == 0).all(axis=2).sum()
    n_white = (arr == 255).all(axis=2).sum()
    assert n_black + n_white > 0
    assert n_black + n_white < 100 * 100 * 0.05  # sparse, not destroyed


def test_toner_streak_touches_narrow_band_only():
    img = Image.new("RGB", (300, 100), (200, 200, 200))
    out = toner_streak(img)
    diff_cols = (np.array(out) != np.array(img)).any(axis=(0, 2))
    assert 0 < diff_cols.sum() <= 300 * 0.05  # a narrow vertical band changed
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_scan_effects.py -v`
Expected: FAIL — `ImportError: cannot import name 'to_grayscale'`

- [ ] **Step 3: Dopisz efekty w `src/transforms.py`** (po `jpeg_artifacts`, przed `class TransformPipeline`)

```python
def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert to grayscale but keep RGB mode (network scanners default to mono)."""
    return img.convert("L").convert("RGB")


def photocopy_contrast(img: Image.Image, low: int = 100, high: int = 190) -> Image.Image:
    """Photocopier tone response: shadows crush to black, highlights burn to white.

    Grayscale values <= low become 0, >= high become 255, in-between values
    ramp linearly. Repeatedly-copied documents look exactly like this — thin
    pen strokes partially drop out, paper texture disappears.
    """
    gray = img.convert("L")
    span = max(1, high - low)
    lut = [
        0 if v <= low else 255 if v >= high else int((v - low) * 255 / span)
        for v in range(256)
    ]
    return gray.point(lut).convert("RGB")


def salt_pepper_noise(img: Image.Image, amount: float = 0.005) -> Image.Image:
    """Sparse black/white pixel dropout typical for photocopies (not smooth Gaussian)."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    n = int(h * w * amount)
    if n == 0:
        return img
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    half = n // 2
    arr[ys[:half], xs[:half]] = 0
    arr[ys[half:], xs[half:]] = 255
    return Image.fromarray(arr)


def toner_streak(img: Image.Image) -> Image.Image:
    """One narrow vertical band of lightened/darkened toner (dirty copier drum)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    band_w = max(2, int(w * random.uniform(0.004, 0.015)))
    x0 = random.randint(0, max(0, w - band_w))
    delta = random.uniform(-40.0, 25.0)  # usually a darker streak
    arr[:, x0:x0 + band_w] = np.clip(arr[:, x0:x0 + band_w] + delta, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_scan_effects.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/transforms.py tests/test_scan_effects.py
git commit -m "Add grayscale and photocopy scan effects (contrast crush, salt-pepper, toner streak)"
```

---

### Task 6: Profile skanu w `fill_form.py`

**Files:**
- Modify: `src/fill_form.py` — funkcja `apply_scan_augmentation` (linie ~544–600) + import z transforms
- Test: `tests/test_scan_profiles.py`

**Interfaces:**
- Consumes: `to_grayscale`, `photocopy_contrast`, `salt_pepper_noise`, `toner_streak` z Task 5; istniejące `uneven_brightness`, `gaussian_noise`, `gaussian_blur`, `jpeg_artifacts`
- Produces:
  - `SCAN_PROFILES: list[tuple[str, float]] = [("clean_color", 0.45), ("grayscale", 0.35), ("photocopy", 0.20)]`
  - `pick_scan_profile() -> str`
  - **Nowa sygnatura:** `apply_scan_augmentation(form: Image.Image) -> tuple[Image.Image, dict]` — bez rotacji, bez `enable_rotation`, bez zwracania kąta/rozmiaru. `meta["profile"]` zawiera nazwę profilu. (Wywołanie w `fill_single_form` naprawia Task 7 — do tego czasu `fill_single_form` jest chwilowo niespójny, dlatego Task 6 i 7 commitują się razem dopiero po Task 7? NIE — patrz Step 5: w tym tasku od razu poprawiamy też miejsce wywołania, żeby commit był spójny.)

- [ ] **Step 1: Napisz failujący test**

`tests/test_scan_profiles.py`:

```python
import random
from collections import Counter

import numpy as np
from PIL import Image

from fill_form import SCAN_PROFILES, pick_scan_profile, apply_scan_augmentation


def test_profile_names_and_weights():
    names = [n for n, _ in SCAN_PROFILES]
    assert names == ["clean_color", "grayscale", "photocopy"]
    assert abs(sum(w for _, w in SCAN_PROFILES) - 1.0) < 1e-9


def test_pick_distribution_roughly_matches_weights():
    random.seed(42)
    counts = Counter(pick_scan_profile() for _ in range(4000))
    assert 0.38 < counts["clean_color"] / 4000 < 0.52
    assert 0.28 < counts["grayscale"] / 4000 < 0.42
    assert 0.14 < counts["photocopy"] / 4000 < 0.26


def _form():
    img = Image.new("RGB", (400, 300), (250, 248, 245))
    img.paste((28, 42, 120), (50, 100, 250, 130))  # a blue "text" block
    return img


def test_apply_returns_image_and_meta_same_size():
    random.seed(1)
    out, meta = apply_scan_augmentation(_form())
    assert out.size == (400, 300)          # geometry untouched
    assert meta["profile"] in ("clean_color", "grayscale", "photocopy")
    assert "rotation_angle_deg" not in meta


def test_grayscale_and_photocopy_kill_color():
    random.seed(0)
    for _ in range(30):
        out, meta = apply_scan_augmentation(_form())
        arr = np.array(out).astype(int)
        max_chroma = np.abs(arr[:, :, 0] - arr[:, :, 2]).max()
        if meta["profile"] in ("grayscale", "photocopy"):
            assert max_chroma <= 25, meta["profile"]  # JPEG may leak a bit of chroma
        else:
            assert max_chroma > 25  # blue ink survives in clean_color
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_scan_profiles.py -v`
Expected: FAIL — `ImportError: cannot import name 'SCAN_PROFILES'`

- [ ] **Step 3: Przepisz `apply_scan_augmentation` w `src/fill_form.py`**

Zamień cały import z transforms (linie 31–41) na:

```python
from transforms import (
    AugmentConfig,
    TransformPipeline,
    WordStyle,
    gaussian_noise,
    gaussian_blur,
    uneven_brightness,
    jpeg_artifacts,
    to_grayscale,
    photocopy_contrast,
    salt_pepper_noise,
    toner_streak,
)
```

(`page_rotation` i `rotate_bbox` znikają z importów — rotacja usunięta na stałe.)

Zamień całą funkcję `apply_scan_augmentation` (wraz z docstringiem) na:

```python
# Scan profiles: how a filled paper form typically enters the system.
# Weights chosen with the user (2026-07): no phone-photo profile, no rotation.
SCAN_PROFILES: list[tuple[str, float]] = [
    ("clean_color", 0.45),
    ("grayscale", 0.35),
    ("photocopy", 0.20),
]


def pick_scan_profile() -> str:
    """Pick a scan profile name according to SCAN_PROFILES weights."""
    names, weights = zip(*SCAN_PROFILES)
    return random.choices(names, weights=weights, k=1)[0]


def apply_scan_augmentation(form: Image.Image) -> tuple[Image.Image, dict]:
    """Apply scan simulation to a fully-filled form using a random profile.

    Profiles model the three ways documents reach the system:
        clean_color — office scanner, mild noise/blur, color kept
        grayscale   — same scanner in mono mode (the common default)
        photocopy   — repeatedly-copied document: crushed contrast,
                      salt-pepper dropout, occasional toner streak

    All effects are photometric only — geometry (and thus every ground-truth
    bbox) is untouched. Page rotation was removed for good: axis-aligned
    boxes degrade on rotated text and the detector copes with skew anyway.

    Returns:
        Tuple of (degraded RGB image, metadata dict incl. "profile").
    """
    profile = pick_scan_profile()
    meta: dict = {"profile": profile}

    if profile in ("clean_color", "grayscale"):
        brightness_var = random.uniform(0.05, 0.15)
        noise_sigma = random.uniform(2.0, 6.0)
        blur_radius = random.uniform(0.3, 0.8)
        jpeg_q = (80, 92)
        form = uneven_brightness(form, brightness_var)
        form = gaussian_noise(form, noise_sigma)
        form = gaussian_blur(form, blur_radius)
        if profile == "grayscale":
            form = to_grayscale(form)
        meta.update({
            "noise_sigma": round(noise_sigma, 2),
            "blur_radius": round(blur_radius, 2),
            "brightness_variation": round(brightness_var, 3),
        })
    else:  # photocopy
        low = random.randint(80, 115)
        high = random.randint(175, 205)
        sp_amount = random.uniform(0.002, 0.008)
        blur_radius = random.uniform(0.2, 0.5)
        jpeg_q = (75, 90)
        form = to_grayscale(form)
        form = photocopy_contrast(form, low=low, high=high)
        form = salt_pepper_noise(form, amount=sp_amount)
        has_streak = random.random() < 0.35
        if has_streak:
            form = toner_streak(form)
        form = gaussian_blur(form, blur_radius)
        meta.update({
            "contrast_low": low,
            "contrast_high": high,
            "salt_pepper_amount": round(sp_amount, 4),
            "blur_radius": round(blur_radius, 2),
            "toner_streak": has_streak,
        })

    form = jpeg_artifacts(form, *jpeg_q)
    meta["jpeg_quality_range"] = list(jpeg_q)
    return form, meta
```

- [ ] **Step 4: Napraw miejsce wywołania w `fill_single_form`** (żeby moduł się importował i commit był spójny)

W `fill_single_form` zamień blok (linie ~1059–1078):

```python
    # Apply scan/photo simulation; rotation may change the canvas size,
    # in which case we must remap each tight bbox into the final coords.
    scan_meta = None
    rotation_angle = None
    pre_rot_size = form.size

    if apply_scan:
        form, scan_meta, rotation_angle, pre_rot_size = apply_scan_augmentation(
            form, enable_rotation=enable_rotation
        )
        if rotation_angle is not None and abs(rotation_angle) > 1e-6:
            post_rot_size = form.size
            for rec in tight_records:
                rec["bbox"] = list(
                    rotate_bbox(tuple(rec["bbox"]), rotation_angle, pre_rot_size, post_rot_size)
                )
            for rec in printed_records:
                rec["bbox"] = list(
                    rotate_bbox(tuple(rec["bbox"]), rotation_angle, pre_rot_size, post_rot_size)
                )
```

na:

```python
    # Scan simulation is photometric only — bboxes stay valid as-is.
    scan_meta = None
    if apply_scan:
        form, scan_meta = apply_scan_augmentation(form)
```

Usuń też parametr `enable_rotation: bool = False` z sygnatury `fill_single_form` i jego opis z docstringa. W `generate_yolo_dataset.py` usuń TYMCZASOWO przekazanie `enable_rotation=args.rotate` (sam argparse `--rotate` usunie Task 8 w całości): zamień linię `enable_rotation=args.rotate,` na nic (skasuj ją).

- [ ] **Step 5: Uruchom testy — mają przejść (całość, nie tylko nowe)**

Run: `python -m pytest tests/ -v`
Expected: wszystkie przechodzą (4 nowe + wcześniejsze)

- [ ] **Step 6: Commit**

```bash
git add src/fill_form.py src/generate_yolo_dataset.py tests/test_scan_profiles.py
git commit -m "Scan profiles: clean color / grayscale / photocopy, page rotation removed for good"
```

---

### Task 7: Przebudowa `fill_single_form` — nowe labele, pola f, multi-line, per-form emptiness, treść medyczna

**Files:**
- Modify: `src/fill_form.py` (duża zmiana — patrz szczegóły)
- Test: `tests/test_fill_single_form.py`

**Interfaces:**
- Consumes: `generate_field_content(kind, vocab, measure, bbox_w)` z Task 4; `TemplatePage.fields` format z Task 2 (`{"label": "p|t|n|f|mix", "x_min", ...}`)
- Produces — **nowa sygnatura**:

```python
def fill_single_form(
    form_path: Path,
    fields: list[dict],            # TemplatePage.fields
    vocab: Vocabulary,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    apply_scan: bool,
    skip_f_fields: bool = False,   # True when form_path is the _partial base
    empty_field_range: tuple[float, float] = (0.0, 0.40),
) -> dict
```

  Zwraca dict: `image`, `records` (JEDNA lista: `{"label", "source", "text", "bbox"}` gdzie source ∈ printed|synthetic|handwritten; text=None dla printed/handwritten), `font`, `text_style`, `ink_color`, `scan_augmentation`, `empty_field_prob` (wylosowana wartość per formularz), `multiline_fields` (licznik, do metadata).
- Produces: `plan_line_slots(bbox_h: int, font_px: int) -> list[int]` — pomocnicza, czysta (poza random), y-offsety linii względem góry bboxa; `[]` = pole nie kwalifikuje się na multi-line.

**Do usunięcia z `fill_form.py` (martwy kod po tej zmianie):**
`LABEL_TO_CATEGORY`, `_FILLER_CONSONANTS/_FILLER_CONS_RARE/_FILLER_VOWELS/_FILLER_VOWELS_RARE`, `_random_filler_word`, `_random_filler_number_group`, `generate_filler_text`, `_NUMBER_SUBCAT_MAX_LEN`, `_TEXT_SUBCAT_MAX_LEN`, `estimate_char_capacity`, `fit_font_size`, `_eligible_subcategories`, `pick_fitting_content`, `fill_pesel_grid`, `_NUMBERISH_CATEGORIES`, `EMPTY_FIELD_PROB`, `PESEL_GRID_ASPECT_RATIO` → zastępuje `GRID_ASPECT_RATIO = 6.0`, oraz CLI (`parse_args`, `load_annotations`, `main`, blok `if __name__`) — jedynym wejściem zostaje `generate_yolo_dataset.py`. Zaktualizuj docstring modułu (opisuje CSV/CLI, którego już nie ma).

- [ ] **Step 1: Napisz failujący test**

`tests/test_fill_single_form.py`:

```python
import random
from pathlib import Path

import pytest
from PIL import Image

from fill_form import fill_single_form, plan_line_slots, MIN_FONT_SIZE
from vocabulary import Vocabulary
from renderer import find_fonts
from fill_form import EXCLUDED_FONTS

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(REPO / "resources"))


@pytest.fixture(scope="module")
def font_path():
    fonts = [f for f in find_fonts(str(REPO / "resources/fonts"))
             if Path(f).name not in EXCLUDED_FONTS]
    assert fonts, "no usable fonts in resources/fonts"
    return fonts[0]


FIELDS = [
    {"label": "p",   "x_min": 40, "y_min": 20,  "x_max": 400, "y_max": 50},
    {"label": "t",   "x_min": 40, "y_min": 80,  "x_max": 700, "y_max": 125},
    {"label": "n",   "x_min": 40, "y_min": 160, "x_max": 500, "y_max": 200},
    {"label": "mix", "x_min": 40, "y_min": 240, "x_max": 400, "y_max": 280},
    {"label": "f",   "x_min": 40, "y_min": 320, "x_max": 600, "y_max": 360},
    # tall description box -> multi-line candidate
    {"label": "t",   "x_min": 40, "y_min": 420, "x_max": 760, "y_max": 620},
]


def _blank_form(tmp_path):
    p = tmp_path / "form_blank.png"
    Image.new("RGB", (800, 700), (252, 250, 247)).save(p)
    return p


def _run(tmp_path, vocab, font_path, **kwargs):
    from transforms import AugmentConfig
    config = AugmentConfig()
    config.paper.enabled = False
    config.scan.enabled = False
    return fill_single_form(
        form_path=_blank_form(tmp_path),
        fields=FIELDS,
        vocab=vocab,
        font_path=font_path,
        config=config,
        pipeline=None,
        apply_scan=False,
        **kwargs,
    )


def test_records_have_sources_and_text(tmp_path, vocab, font_path):
    random.seed(2)
    result = _run(tmp_path, vocab, font_path, empty_field_range=(0.0, 0.0))
    sources = {r["source"] for r in result["records"]}
    assert "printed" in sources and "synthetic" in sources
    for r in result["records"]:
        if r["source"] == "printed":
            assert r["text"] is None
            assert r["bbox"] == [40, 20, 400, 50]  # labeled box passes through
        else:
            assert isinstance(r["text"], str) and len(r["text"]) > 0
    assert 0.0 <= result["empty_field_prob"] <= 0.40


def test_skip_f_fields_emits_handwritten_record(tmp_path, vocab, font_path):
    random.seed(3)
    result = _run(tmp_path, vocab, font_path,
                  skip_f_fields=True, empty_field_range=(0.0, 0.0))
    hw = [r for r in result["records"] if r["source"] == "handwritten"]
    assert len(hw) == 1
    assert hw[0]["label"] == "f"
    assert hw[0]["text"] is None
    assert hw[0]["bbox"] == [40, 320, 600, 360]
    # and the f field was NOT synthetically filled
    assert not any(r["label"] == "f" and r["source"] == "synthetic"
                   for r in result["records"])


def test_full_emptiness_leaves_only_printed(tmp_path, vocab, font_path):
    random.seed(4)
    result = _run(tmp_path, vocab, font_path, empty_field_range=(1.0, 1.0))
    assert all(r["source"] == "printed" for r in result["records"])
    assert result["empty_field_prob"] == 1.0


def test_tall_box_can_produce_multiple_lines(tmp_path, vocab, font_path):
    # over several seeds the 200px-tall box must at least once yield >= 2
    # separate synthetic records (multi-line filling)
    found = False
    for seed in range(12):
        random.seed(seed)
        result = _run(tmp_path, vocab, font_path, empty_field_range=(0.0, 0.0))
        tall = [r for r in result["records"]
                if r["source"] == "synthetic" and r["label"] == "t"
                and r["bbox"][1] >= 400]
        if len(tall) >= 2:
            # lines must not overlap vertically
            tall.sort(key=lambda r: r["bbox"][1])
            assert tall[0]["bbox"][3] <= tall[1]["bbox"][1] + 8  # small tolerance
            found = True
            break
    assert found, "tall box never produced multi-line fill in 12 seeds"


def test_plan_line_slots():
    random.seed(0)
    assert plan_line_slots(bbox_h=45, font_px=30) == []      # fits one line only
    slots_seen = set()
    for _ in range(40):
        slots = plan_line_slots(bbox_h=200, font_px=30)
        assert 1 <= len(slots) <= 3
        assert all(b - a >= 30 for a, b in zip(slots, slots[1:]))  # >= font_px apart
        assert all(0 <= s <= 200 - 30 for s in slots)
        slots_seen.add(len(slots))
    assert max(slots_seen) >= 2  # multi-line actually happens
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_fill_single_form.py -v`
Expected: FAIL — `ImportError: cannot import name 'plan_line_slots'`

- [ ] **Step 3: Dodaj `plan_line_slots` i przebuduj `fill_single_form` w `src/fill_form.py`**

Dodaj import na górze: `from field_content import generate_field_content`.

Dodaj po stałych (`INK_BLUE` itd.):

```python
# Digit grids: number field much wider than tall -> check for printed cells
GRID_ASPECT_RATIO = 6.0

# Multi-line filling of tall description fields
MULTILINE_MAX_LINES = 3
LINE_PITCH_RANGE = (1.2, 1.6)  # line spacing as multiple of handwriting size


def plan_line_slots(bbox_h: int, font_px: int) -> list[int]:
    """Plan y-offsets (from the field top) for writing 1-3 lines in a tall box.

    A field qualifies for multi-line filling when at least two line pitches
    fit into its height. Returns [] for non-qualifying (single-line) fields —
    the caller then uses the regular single-line placement.
    """
    pitch = int(font_px * random.uniform(*LINE_PITCH_RANGE))
    if pitch <= 0:
        return []
    usable = bbox_h - int(font_px * 0.4)  # bottom margin for descenders
    max_lines = usable // pitch
    if max_lines < 2:
        return []
    n = random.randint(1, min(MULTILINE_MAX_LINES, max_lines))
    top = int(bbox_h * 0.06)  # people start writing near the top
    return [top + i * pitch for i in range(n)]
```

Zamień CAŁĄ funkcję `fill_single_form` na:

```python
def fill_single_form(
    form_path: Path,
    fields: list[dict],
    vocab: Vocabulary,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    apply_scan: bool,
    skip_f_fields: bool = False,
    empty_field_range: tuple[float, float] = (0.0, 0.40),
) -> dict:
    """Generate one filled-form variant and return image + ground-truth records.

    Args:
        form_path: Path to the base image (the _blank or _partial variant).
        fields: Field dicts from TemplatePage.fields
            ({"label": "p|t|n|f|mix", "x_min", "y_min", "x_max", "y_max"}).
        vocab: Loaded Vocabulary (source of all written content).
        font_path: Font for this variant (one handwriting per form).
        config: Augmentation config for text rendering (None = no augment).
        pipeline: Kept for API stability (unused inside renders).
        apply_scan: Whether to apply scan-profile simulation at the end.
        skip_f_fields: True when form_path is the _partial base — f fields
            already contain real handwriting; they are not filled, but their
            labeled bbox is recorded as a "handwritten" text line.
        empty_field_range: Per-FORM diligence: one empty-probability is drawn
            from this range per variant and applied to every fill-in field
            (real forms are correlated — one is fully filled, another half-empty).

    Returns:
        Dict with keys:
            image (PIL.Image) — final RGB image
            records (list[dict]) — {"label", "source", "text", "bbox"} where
                source is printed|synthetic|handwritten; text is None unless
                synthetic; bbox is [x_min, y_min, x_max, y_max]
            font (str), text_style (dict|None), ink_color (list[int]),
            scan_augmentation (dict|None), empty_field_prob (float),
            multiline_fields (int) — how many fields got >= 2 lines
    """
    font_name = Path(font_path).name

    if config is not None and config.char.enabled:
        form_style = WordStyle.random(config.char)
        form_style.do_thicken = font_name in FONTS_NEEDING_THICKENING
        form_style.thicken_kernel = 3
    else:
        form_style = None

    # One pen per form: black or blue, with slight per-form shade variation
    base_ink = random.choice([INK_BLACK, INK_BLUE])
    ink_color = tuple(
        max(0, min(255, c + random.randint(-8, 8))) for c in base_ink
    )

    # One handwriting size per form; small fields shrink it, tall never enlarge
    form_font_px = random.randint(*FORM_FONT_SIZE_RANGE)

    # Per-form diligence: how likely each fill-in field stays empty
    form_empty_prob = random.uniform(*empty_field_range)

    form = Image.open(form_path).convert("RGB")

    records: list[dict] = []
    multiline_fields = 0

    def _measure_fn(font_size: int):
        """Width estimator matching what the renderer will actually draw."""
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        stretch = form_style.x_stretch if form_style is not None else 1.0
        tracking = (
            form_style.tracking_ratio * font_size if form_style is not None else 0.0
        )
        return lambda t: font.getlength(t) * stretch + tracking * len(t)

    def _render_and_paste(text: str, bbox_w: int, bbox_h: int, x_min: int,
                          y_min: int, font_size: int,
                          y_offset: Optional[int] = None):
        """Render one line, optionally pen-fade it, paste, return tight bbox."""
        text_img = render_field_to_bbox(
            text, bbox_w, bbox_h, font_path, config, pipeline, form_style,
            font_size=font_size,
        )
        if random.random() < PEN_FADE_PROB:
            text_img = apply_pen_fade(text_img)
        v_jitter = int(bbox_h * V_OVERFLOW_FRAC) if y_offset is None else 3
        return paste_text_on_form(
            form, text_img, bbox_w, bbox_h, x_min, y_min,
            ink_color=ink_color, v_jitter_px=v_jitter, y_offset=y_offset,
        )

    for field in fields:
        label = field["label"]
        x_min, y_min = field["x_min"], field["y_min"]
        x_max, y_max = field["x_max"], field["y_max"]
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w <= 0 or bbox_h <= 0:
            continue

        # Printed text: pass the labeled bbox straight through to GT
        if label == "p":
            records.append({
                "label": label, "source": "printed",
                "text": None, "bbox": [x_min, y_min, x_max, y_max],
            })
            continue

        # Real handwriting already on the _partial base: record, don't fill
        if label == "f" and skip_f_fields:
            records.append({
                "label": label, "source": "handwritten",
                "text": None, "bbox": [x_min, y_min, x_max, y_max],
            })
            continue

        # Per-form diligence: some fields stay empty
        if random.random() < form_empty_prob:
            continue

        field_font_size = max(MIN_FONT_SIZE, min(form_font_px, int(bbox_h * 0.7)))
        measure = _measure_fn(field_font_size)
        content_kind = "t" if label == "f" else label  # f on blank behaves like t

        # Digit grids (kratki): detected from the printed separators
        if label == "n" and (bbox_w / bbox_h) >= GRID_ASPECT_RATIO:
            grid_cells = detect_grid_cells(form, x_min, y_min, x_max, y_max)
            if grid_cells:
                n_cells = len(grid_cells)
                if n_cells == 11:
                    text, _ = vocab.get_random_text("pesel")
                else:
                    text = "".join(
                        random.choice("0123456789") for _ in range(n_cells)
                    )
                tight = fill_digit_cells(
                    form, text, grid_cells, y_min, bbox_h,
                    font_path, config, pipeline, form_style, ink_color,
                )
                if tight is not None:
                    records.append({
                        "label": label, "source": "synthetic",
                        "text": text, "bbox": list(tight),
                    })
                continue

        # Multi-line filling for tall text-ish boxes
        slots: list[int] = []
        if content_kind in ("t", "mix"):
            slots = plan_line_slots(bbox_h, field_font_size)

        if len(slots) >= 2:
            multiline_fields += 1
            line_h = int(field_font_size * 1.4)
            for slot in slots:
                line_text = generate_field_content(
                    content_kind, vocab, measure, bbox_w
                )
                if not line_text:
                    continue
                tight = _render_and_paste(
                    line_text, bbox_w, line_h, x_min, y_min,
                    field_font_size, y_offset=slot,
                )
                if tight is not None:
                    records.append({
                        "label": label, "source": "synthetic",
                        "text": line_text, "bbox": list(tight),
                    })
            continue

        # Single-line fill
        text = generate_field_content(content_kind, vocab, measure, bbox_w)
        if not text:
            continue
        tight = _render_and_paste(
            text, bbox_w, bbox_h, x_min, y_min, field_font_size
        )
        if tight is not None:
            records.append({
                "label": label, "source": "synthetic",
                "text": text, "bbox": list(tight),
            })

    # Scan simulation is photometric only — bboxes stay valid as-is.
    scan_meta = None
    if apply_scan:
        form, scan_meta = apply_scan_augmentation(form)

    text_style_meta = None
    if form_style is not None:
        text_style_meta = {
            "base_rotation_deg": round(form_style.base_rotation, 2),
            "base_scale": round(form_style.base_scale, 3),
            "handwriting_size_px": form_font_px,
            "x_stretch": round(form_style.x_stretch, 3),
            "tracking_ratio": round(form_style.tracking_ratio, 3),
        }

    return {
        "image": form,
        "records": records,
        "font": font_name,
        "text_style": text_style_meta,
        "ink_color": list(ink_color),
        "scan_augmentation": scan_meta,
        "empty_field_prob": round(form_empty_prob, 3),
        "multiline_fields": multiline_fields,
    }
```

Rozszerz `paste_text_on_form` o parametr `y_offset` — w sygnaturze po `v_jitter_px: int = 0` dodaj `y_offset: Optional[int] = None`, opis w docstringu: "Explicit paste offset from the field top (used for multi-line slots); overrides the centering/top-anchoring logic." Zamień blok wyliczania `paste_y`:

```python
    paste_x = x_min
    if y_offset is not None:
        paste_y = y_min + y_offset
    elif bbox_h > text_img.height * 2.5:
        # Tall (multi-line-style) box: people start writing near the top,
        # they don't vertically center a single line in a big rectangle
        paste_y = y_min + int(bbox_h * 0.12)
    else:
        paste_y = y_min + (bbox_h - text_img.height) // 2
```

(reszta funkcji bez zmian — jitter, clamp, paste, tight bbox).

- [ ] **Step 4: Usuń martwy kod wymieniony w Interfaces**

Usuń z `fill_form.py` wszystkie symbole z listy "Do usunięcia" (wraz z docstringiem modułu opisującym CLI — nowy docstring modułu: `"""Form-filling core: renders synthetic handwriting into labeled template fields. Used by generate_yolo_dataset.py; see docs/superpowers/specs/2026-07-06-medical-dataset-v2-design.md."""`). Usuń nieużywane importy (`argparse`, `csv`, `json`, `sys` — sprawdź faktyczne użycie po cięciu). UWAGA: `generate_yolo_dataset.py` importuje `EMPTY_FIELD_PROB` i `load_annotations` — w tym kroku zamień tam linię importu na `from fill_form import EXCLUDED_FONTS, fill_single_form` i tymczasowo zostaw resztę generatora bez zmian (naprawa w Task 8; generator w tym commicie może być niefunkcjonalny, ale importowalny — testy tego nie dotykają).

- [ ] **Step 5: Uruchom pełny zestaw testów**

Run: `python -m pytest tests/ -v`
Expected: wszystkie przechodzą (w tym 5 nowych; test multi-line może chwilę mielić — renderuje realne obrazy)

- [ ] **Step 6: Commit**

```bash
git add src/fill_form.py src/generate_yolo_dataset.py tests/test_fill_single_form.py
git commit -m "Rework fill_single_form: p/t/n/f/mix labels, medical content, multi-line tall fields, per-form emptiness"
```

---

### Task 8: Przebudowa `generate_yolo_dataset.py` — nowe wejście, transkrypcja w GT

**Files:**
- Modify: `src/generate_yolo_dataset.py` (przepisanie main + argumentów)
- Modify: `src/visualize_bboxes.py` (kolor dla źródła `handwritten`)
- Test: `tests/test_generate_dataset_e2e.py`

**Interfaces:**
- Consumes: `load_templates` (Task 2), `fill_single_form` z nową sygnaturą (Task 7)
- Produces:
  - CLI: `python src/generate_yolo_dataset.py --templates-dir templates --output-dir output/<nazwa> --variants-per-form N [--seed S] [--no-metadata] [--no-scan]` (bez `--forms-dir`, `--annotations`, `--rotate`)
  - `PARTIAL_BASE_PROB = 0.5`
  - COCO: każda adnotacja ma dodatkowo `"text"` (str|None) i `"source"` ("printed"|"synthetic"|"handwritten")
  - `ground_truth.csv`: nagłówek `filename,label,x_min,y_min,x_max,y_max,source,text`
  - metadata JSON: dodatkowo `base` ("blank"|"partial"), `empty_field_prob`, `multiline_fields`; `fields` = `records`

- [ ] **Step 1: Napisz failujący test end-to-end (mini-templates w tmp_path)**

`tests/test_generate_dataset_e2e.py`:

```python
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
         "--seed", "123"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr

    coco = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 6
    sources = {a["source"] for a in coco["annotations"]}
    assert "printed" in sources and "synthetic" in sources
    for a in coco["annotations"]:
        assert "text" in a and "source" in a
        if a["source"] == "synthetic":
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
```

Uwaga: jeśli przy seedzie 123 wszystkie 6 wariantów wypadnie na jednym podkładzie, zwiększ `--variants-per-form` do 10 w teście — chodzi o pokrycie obu gałęzi, nie o dokładny podział.

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_generate_dataset_e2e.py -v`
Expected: FAIL (generator jeszcze czyta stare argumenty / stary format wyniku)

- [ ] **Step 3: Przepisz `src/generate_yolo_dataset.py`**

Docstring modułu zaktualizuj (wejście = templates dir; wyjście z text/source). Kluczowe zmiany:

```python
"""Generate a COCO line-detection + OCR training dataset from form templates.

Input:  a templates/ directory built by build_templates.py (COCO
        annotations.json with p/n/t/f/mix labels + per-page _blank/_partial
        base images).
Output: output-dir/
        ├── images/{page}_{v:04d}.jpg
        ├── annotations.json    COCO; each annotation carries "text" (what was
        │                       written; None for printed/handwritten) and
        │                       "source" (printed|synthetic|handwritten)
        ├── ground_truth.csv    filename,label,x_min,y_min,x_max,y_max,source,text
        └── metadata/*.json     per-image generation parameters

Usage:
    python src/generate_yolo_dataset.py --templates-dir templates \
        --output-dir output/run1 --variants-per-form 25 --seed 42
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from vocabulary import Vocabulary
from renderer import find_fonts
from fill_form import EXCLUDED_FONTS, fill_single_form
from template_loader import load_templates
from transforms import AugmentConfig, TransformPipeline

CLASS_NAME = "text_line"
COCO_CATEGORY_ID = 1

# When a page has a _partial base, half the variants use it (real handwriting
# in f-fields), half use _blank (f-fields filled synthetically)
PARTIAL_BASE_PROB = 0.5
```

`parse_args()`: zamień `--forms-dir`/`--annotations` na `--templates-dir` (type=str, default="templates"), usuń `--rotate`; reszta argumentów bez zmian. `build_form_filling_config()` i `clamp_bbox()` zostają jak są.

Nowy `main()` (pętla generacji):

```python
def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        print(f"ERROR: templates-dir not found: {templates_dir}", file=sys.stderr)
        sys.exit(1)

    pages = load_templates(templates_dir)
    if not pages:
        print("ERROR: no usable template pages", file=sys.stderr)
        sys.exit(1)
    print(f"Templates to fill: {len(pages)}")
    for p in pages:
        partial = " (+partial)" if p.partial_path else ""
        print(f"  {p.name} ({len(p.fields)} fields){partial}")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_metadata:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary...")
    vocab = Vocabulary(args.resource_dir)
    all_fonts = find_fonts(args.font_dir)
    fonts = [f for f in all_fonts if Path(f).name not in EXCLUDED_FONTS]
    if not fonts:
        print(f"ERROR: no usable fonts in {args.font_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(fonts)} fonts available")

    config = build_form_filling_config()
    pipeline = TransformPipeline(config)
    apply_scan = not args.no_scan

    coco: dict = {
        "info": {
            "description": "Synthetic Polish medical forms — text lines with transcriptions",
            "version": "2.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": COCO_CATEGORY_ID, "name": CLASS_NAME, "supercategory": "text"}
        ],
    }
    next_image_id = 1
    next_ann_id = 1

    csv_out = open(output_dir / "ground_truth.csv", "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(
        ["filename", "label", "x_min", "y_min", "x_max", "y_max", "source", "text"])

    n_variants = max(1, args.variants_per_form)
    print(f"Generating {n_variants} variants per template "
          f"({n_variants * len(pages)} total)")
    print(f"Scan augmentation: {'ON' if apply_scan else 'OFF'}")

    total_count = 0
    skipped_blank = 0

    for page in pages:
        for v in range(1, n_variants + 1):
            font_path = random.choice(fonts)
            stem = f"{page.name}_{v:04d}"

            use_partial = (
                page.partial_path is not None and random.random() < PARTIAL_BASE_PROB
            )
            base_path = page.partial_path if use_partial else page.blank_path

            result = fill_single_form(
                form_path=base_path,
                fields=page.fields,
                vocab=vocab,
                font_path=font_path,
                config=config,
                pipeline=pipeline,
                apply_scan=apply_scan,
                skip_f_fields=use_partial,
            )

            img: Image.Image = result["image"]
            iw, ih = img.size
            img_path = images_dir / f"{stem}.jpg"

            image_id = next_image_id
            next_image_id += 1
            coco["images"].append({
                "id": image_id, "file_name": img_path.name,
                "width": iw, "height": ih,
            })

            n_boxes = 0
            for rec in result["records"]:
                clamped = clamp_bbox(tuple(rec["bbox"]), iw, ih)
                if clamped is None:
                    skipped_blank += 1
                    continue
                x1, y1, x2, y2 = clamped
                w, h = x2 - x1, y2 - y1
                coco["annotations"].append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY_ID,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "source": rec["source"],
                    "text": rec["text"],
                })
                next_ann_id += 1
                n_boxes += 1
                csv_writer.writerow(
                    [img_path.name, CLASS_NAME, x1, y1, x2, y2,
                     rec["source"], rec["text"] or ""])

            img.save(img_path, quality=92)

            if not args.no_metadata:
                metadata = {
                    "template": page.name,
                    "base": "partial" if use_partial else "blank",
                    "output_image": img_path.name,
                    "image_size": [iw, ih],
                    "font": result["font"],
                    "ink_color": result["ink_color"],
                    "empty_field_prob": result["empty_field_prob"],
                    "multiline_fields": result["multiline_fields"],
                    "fields": result["records"],
                    "num_annotations": n_boxes,
                }
                if result["text_style"] is not None:
                    metadata["text_style"] = result["text_style"]
                if result["scan_augmentation"] is not None:
                    metadata["scan_augmentation"] = result["scan_augmentation"]
                with open(metadata_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            total_count += 1
            if total_count % 50 == 0 or total_count == 1:
                print(f"  [{total_count}] {stem}.jpg  ({n_boxes} bboxes)")

    csv_out.close()
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\nDone. {total_count} images generated in {output_dir}/")
    print(f"  annotations.json  --> COCO ({len(coco['annotations'])} annotations "
          f"with text+source)")
    if skipped_blank:
        print(f"  Skipped {skipped_blank} blank/degenerate bbox(es)")
```

- [ ] **Step 4: Zaktualizuj `src/visualize_bboxes.py`**

Czyta `ground_truth.csv` po nazwach kolumn — nowa kolumna `text` jest addytywna, ale `source` ma teraz trzy wartości. Dodaj kolor: `printed`→czerwony (jak było), `rendered`/`synthetic`→zielony, `handwritten`→niebieski. Znajdź mapowanie koloru po source i rozszerz o `"handwritten": (40, 90, 255)` oraz przemianuj oczekiwaną wartość `rendered` na `synthetic` (zostaw obie w mapie dla starych CSV).

- [ ] **Step 5: Uruchom pełny zestaw testów**

Run: `python -m pytest tests/ -v`
Expected: wszystkie przechodzą (e2e odpala generator subprocessem — potrwa kilkanaście sekund)

- [ ] **Step 6: Commit**

```bash
git add src/generate_yolo_dataset.py src/visualize_bboxes.py tests/test_generate_dataset_e2e.py
git commit -m "Generator v2: templates-dir input, blank/partial bases, transcriptions in COCO ground truth"
```

---

### Task 9: Pełny przebieg na prawdziwych szablonach + weryfikacja kryteriów specu

**Files:**
- Create (wygenerowane, poza gitem): `output/test_run/`
- Brak zmian w kodzie (chyba że weryfikacja coś wykryje)

**Interfaces:**
- Consumes: `templates/` z Task 3, cały pipeline z Tasków 2–8
- Produces: dataset testowy do przeglądu przez użytkownika + raport zgodności ze spec'iem

- [ ] **Step 1: Wygeneruj testowy dataset (3 warianty × ~40 stron)**

```bash
cd /c/Users/tomek/Desktop/text-gen
python src/generate_yolo_dataset.py --templates-dir templates \
    --output-dir output/test_run --variants-per-form 3 --seed 123
```

Expected: ~120 obrazów bez błędów, log pokazuje strony z `(+partial)`.

- [ ] **Step 2: Zweryfikuj kryteria sukcesu ze specu**

```bash
python -c "
import json, collections
coco = json.load(open('output/test_run/annotations.json', encoding='utf-8'))
print('images:', len(coco['images']), 'annotations:', len(coco['annotations']))
print('sources:', collections.Counter(a['source'] for a in coco['annotations']))
syn = [a for a in coco['annotations'] if a['source'] == 'synthetic']
print('sample texts:', [a['text'] for a in syn[:10]])
import glob
profiles = collections.Counter(
    json.load(open(m, encoding='utf-8'))['scan_augmentation']['profile']
    for m in glob.glob('output/test_run/metadata/*.json'))
print('profiles:', profiles)
bases = collections.Counter(
    json.load(open(m, encoding='utf-8'))['base']
    for m in glob.glob('output/test_run/metadata/*.json'))
print('bases:', bases)
ml = sum(json.load(open(m, encoding='utf-8'))['multiline_fields']
         for m in glob.glob('output/test_run/metadata/*.json'))
print('multiline fields total:', ml)
"
```

Expected: sources zawiera printed+synthetic+handwritten; sample texts to polskie wyrażenia medyczne/osobowe (nie sylaby); profile ~45/35/20 (±10 p.p. przy 120 obrazach); bases zawiera blank i partial; multiline > 0.

- [ ] **Step 3: Wygeneruj wizualizacje do przeglądu**

```bash
python src/visualize_yolo.py output/test_run/images/KARTA_PACJENTA_1_p19_0001.jpg
python src/visualize_yolo.py output/test_run/images/KARTA_PACJENTA_1_p03_0002.jpg
```

Obejrzyj wyniki: bboxy tight na każdej linii (w tym każdej linii pól wielolinijkowych i polach f na podkładach _partial), tekst czytelny, profile skanu wizualnie różne (kolor / szarość / ksero).

- [ ] **Step 4: Raport dla użytkownika**

Przedstaw użytkownikowi: liczby z kroku 2, ścieżki do 3–4 wizualizacji (różne profile, strona z _partial, strona z multi-line) — użytkownik robi swój przegląd wizualny (jego pętla review jest częścią procesu).

- [ ] **Step 5: Commit końcowy (jeśli coś było poprawiane) + push**

```bash
git status   # working tree powinno być czyste poza output/ (ignorowane)
git push origin augmentations
```

---

## Self-Review (wykonane przy pisaniu planu)

- Spec coverage: sekcja 1→Task 2+3, sekcja 2→Task 7+8, sekcja 3→Task 4+7, sekcja 4→Task 7, sekcja 5→Task 7, sekcja 6→Task 5+6, sekcja 7→Task 8, sekcja 8→Task 1, kryteria sukcesu→Task 9. ✓
- Typy spójne: `TemplatePage.fields` (Task 2) = `fields` param `fill_single_form` (Task 7); `records` (Task 7) = pętla GT (Task 8); `measure` callable (Task 4) = `_measure_fn` (Task 7); `apply_scan_augmentation(form) -> (img, meta)` (Task 6) = wywołanie w Task 7. ✓
- `paste_text_on_form` zyskuje `y_offset` w Task 7 — jedyny konsument (fill_single_form) aktualizowany w tym samym tasku. ✓
