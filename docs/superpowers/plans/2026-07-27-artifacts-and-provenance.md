# Artefakty na dokumencie + historia zbiorów — plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dodać do generatora trzy artefakty papierowe (kreski długopisem, pieczątki, marker), zabezpieczyć treść przed brakującymi glifami czcionek i wprowadzić kartę + rejestr wygenerowanych zbiorów danych.

**Architecture:** Nowy moduł `src/text_sanitize.py` czyści treść przed pomiarem i renderem (normalizacja typograficzna + odsiew znaków spoza `cmap` fontu). Nowy moduł `src/artifacts.py` rysuje trzy efekty na gotowym, wypełnionym obrazie — przed symulacją skanu, bez zmiany geometrii. `fill_single_form` wywołuje je i dokłada rekordy pieczątki do ground truth. Nowy `src/dataset_card.py` zrzuca ustawienia i statystyki generacji do karty w folderze zbioru oraz do rejestru w repo.

**Tech Stack:** Python 3.13, Pillow, numpy, fontTools (nowa zależność), pytest. Windows/PowerShell; komendy podane dla Git Basha.

**Spec:** `docs/superpowers/specs/2026-07-27-artifacts-and-provenance-design.md`

## Global Constraints

- Kolejność w pipeline jest wiążąca: wypełnienie tekstem → pieczątka → marker → kreski → symulacja skanu.
- Żaden artefakt nie zmienia rozmiaru ani geometrii obrazu; istniejące bboxy muszą zostać nietknięte co do piksela.
- Transkrypcja w ground truth musi być **znak w znak** tym, co poszło do renderu — sanityzacja zawsze przed pomiarem szerokości i przed zapisem rekordu.
- Prawdopodobieństwa: `STAMP_PROB = 0.30`, `STROKE_PROB = 0.20`, `HIGHLIGHT_PROB = 0.12`.
- Pieczątki: kąt `±20°`, do ground truth **tylko** prostokątne o `|kąt| ≤ 5°`; okrągłe nigdy. Udział okrągłych: `STAMP_ROUND_PROB = 0.30`.
- Marker: luminancja koloru (`0.299R + 0.587G + 0.114B`) musi wynosić **≥ 150**; kolor o niższej jest rozjaśniany do bieli aż spełni warunek.
- Kreski i marker nie produkują żadnych bboxów.
- Rotacja strony pozostaje trwale wyłączona.
- Docstringi i komentarze po angielsku (jak reszta kodu). Commity **bez** stopki `Co-Authored-By`.
- Testy uruchamiane z roota repo: `python -m pytest tests/ -v` (conftest dodaje `src/` do `sys.path`). Stan wyjściowy: 29 testów przechodzi.
- Znane, akceptowane ostrzeżenie w wyjściu testów: openpyxl „Workbook contains no default style". Żadne inne ostrzeżenia nie są akceptowane.
- `output/` jest w `.gitignore` — wygenerowane zbiory nigdy nie trafiają do commita.

## Struktura plików

| Plik | Odpowiedzialność |
|---|---|
| `src/text_sanitize.py` (nowy) | normalizacja typograficzna + odsiew znaków spoza fontu |
| `src/artifacts.py` (nowy) | trzy efekty rysowane na wypełnionym formularzu |
| `src/dataset_card.py` (nowy) | zrzut ustawień, karta zbioru, rejestr w `docs/datasets.md` |
| `src/field_content.py` | wywołanie sanityzacji przed pomiarem |
| `src/fill_form.py` | wpięcie artefaktów w pipeline, rekordy pieczątki |
| `src/generate_yolo_dataset.py` | metadane artefaktów, liczniki, CLI karty |
| `resources/fonts_print/` (nowy) | DejaVu Sans — czcionka drukowana pieczątek |
| `docs/datasets.md` (nowy) | rejestr zbiorów, wersjonowany w gicie |

---

### Task 1: Sanityzacja treści

**Files:**
- Create: `src/text_sanitize.py`
- Modify: `src/field_content.py` (sygnatura `generate_field_content`, wywołanie samplera)
- Modify: `src/fill_form.py` (3 wywołania `generate_field_content`)
- Modify: `requirements.txt`
- Test: `tests/test_text_sanitize.py`

**Interfaces:**
- Consumes: `Vocabulary.get_random_text` (bez zmian), `field_content._UNIT_SAMPLERS`
- Produces:
  - `normalize_for_handwriting(text: str) -> str`
  - `font_charset(font_path: str) -> frozenset[int]` (cache'owane)
  - `strip_unsupported(text: str, font_path: str) -> str`
  - `sanitize(text: str, font_path: str) -> str`
  - `generate_field_content(kind, vocab, measure, bbox_w, font_path: Optional[str] = None) -> str` — nowy, opcjonalny piąty parametr

- [ ] **Step 1: Dodaj fontTools do zależności**

W `requirements.txt` dopisz na końcu linię:

```
fonttools>=4.40
```

Potem: `pip install "fonttools>=4.40"`

- [ ] **Step 2: Napisz failujący test**

`tests/test_text_sanitize.py`:

```python
import random
from pathlib import Path

import pytest

from text_sanitize import (
    normalize_for_handwriting,
    font_charset,
    strip_unsupported,
    sanitize,
)

REPO = Path(__file__).resolve().parent.parent
FONT_DIR = REPO / "resources" / "fonts"


def test_dagger_removed_and_brackets_become_parens():
    assert normalize_for_handwriting("Zapalenie † [ostre]") == "Zapalenie (ostre)"


def test_micro_sign_becomes_u():
    assert normalize_for_handwriting("100µg") == "100ug"
    assert normalize_for_handwriting("100μg") == "100ug"


def test_all_dashes_unified():
    assert normalize_for_handwriting("A–B‑C—D") == "A-B-C-D"


def test_quotes_and_nbsp():
    assert normalize_for_handwriting("„ostry” stan") == '"ostry" stan'


def test_plain_text_untouched():
    assert normalize_for_handwriting("nadciśnienie tętnicze 42 mg") == \
        "nadciśnienie tętnicze 42 mg"


def test_font_charset_reports_polish_coverage():
    path = str(FONT_DIR / "Marek_1" / "Marek_1-Regular.ttf")
    codes = font_charset(path)
    assert ord("ą") in codes and ord("ż") in codes and ord("5") in codes
    assert ord("†") not in codes  # the dagger this font lacks


def test_strip_unsupported_drops_missing_glyph():
    path = str(FONT_DIR / "Marek_1" / "Marek_1-Regular.ttf")
    # dagger survives normalization only if we skip it; feed it directly
    assert strip_unsupported("A†B", path) == "AB"
    assert strip_unsupported("Zażółć gęślą jaźń", path) == "Zażółć gęślą jaźń"


def test_strip_unsupported_keeps_text_when_font_unreadable(tmp_path):
    broken = tmp_path / "broken.ttf"
    broken.write_bytes(b"not a font")
    assert strip_unsupported("cokolwiek", str(broken)) == "cokolwiek"


def test_sanitize_runs_both_stages():
    path = str(FONT_DIR / "Marek_1" / "Marek_1-Regular.ttf")
    assert sanitize("Zapalenie † [ostre] 100µg", path) == \
        "Zapalenie (ostre) 100ug"


def test_every_shipped_font_renders_all_sanitized_vocabulary():
    """Success criterion 1: no .notdef boxes anywhere in generated data."""
    from vocabulary import Vocabulary
    from field_content import generate_field_content

    vocab = Vocabulary(str(REPO / "resources"))
    fonts = sorted(FONT_DIR.rglob("*.ttf"))
    assert len(fonts) >= 15, "expected the full font pool"

    random.seed(11)
    samples: list[str] = []
    for kind in ("t", "n", "mix"):
        for _ in range(400):
            samples.append(
                generate_field_content(kind, vocab, lambda s: len(s) * 8.0, 900)
            )

    for font in fonts:
        codes = font_charset(str(font))
        if not codes:
            continue
        for raw in samples:
            clean = sanitize(raw, str(font))
            missing = {ch for ch in clean if ch != " " and ord(ch) not in codes}
            assert not missing, f"{font.name}: {missing!r} in {clean!r}"
```

- [ ] **Step 3: Uruchom test — ma failować**

Run: `python -m pytest tests/test_text_sanitize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'text_sanitize'`

- [ ] **Step 4: Napisz `src/text_sanitize.py`**

```python
"""Keep generated content renderable by every handwriting font we ship.

Custom fonts built from real handwriting cover letters, digits and Polish
diacritics but not the typographic oddities our medical vocabulary carries
(ICD-10 daggers, en dashes, micro signs). Rendering those produces an empty
.notdef box while the ground-truth transcription still claims a character is
there — poison for the OCR model that trains on this data.

Two stages, in this order:
    normalize_for_handwriting — replace typography with what a hand writes
    strip_unsupported         — drop whatever the chosen font still lacks
"""

import re
from functools import lru_cache

from fontTools.ttLib import TTFont

# Typographic character -> what a person filling the form would actually draw
CHAR_REPLACEMENTS: dict[str, str] = {
    "‑": "-",   # non-breaking hyphen
    "–": "-",   # en dash
    "—": "-",   # em dash
    "µ": "u",   # micro sign: 100µg -> 100ug
    "μ": "u",   # greek small mu
    "†": "",    # dagger — ICD-10 aetiology marker, never handwritten
    "‡": "",    # double dagger
    "[": "(",
    "]": ")",
    "„": '"',
    "”": '"',
    "“": '"',
    "’": "'",
    " ": " ",   # non-breaking space
}

_MULTISPACE = re.compile(r"\s{2,}")


def normalize_for_handwriting(text: str) -> str:
    """Replace typographic characters with their handwritten equivalents.

    Dropping a character can leave a double space behind, so whitespace is
    collapsed afterwards.
    """
    out = "".join(CHAR_REPLACEMENTS.get(ch, ch) for ch in text)
    return _MULTISPACE.sub(" ", out).strip()


@lru_cache(maxsize=64)
def font_charset(font_path: str) -> frozenset[int]:
    """Return the unicode code points a font can render.

    Empty frozenset means the file could not be parsed; callers treat that
    as "unknown" and leave the text alone rather than deleting everything.
    """
    try:
        font = TTFont(font_path, fontNumber=0, lazy=True)
    except Exception:
        return frozenset()
    try:
        codes: set[int] = set()
        for table in font["cmap"].tables:
            codes |= set(table.cmap.keys())
        return frozenset(codes)
    except Exception:
        return frozenset()
    finally:
        font.close()


def strip_unsupported(text: str, font_path: str) -> str:
    """Drop characters the font has no glyph for. Spaces are always kept."""
    codes = font_charset(font_path)
    if not codes:
        return text
    out = "".join(ch for ch in text if ch == " " or ord(ch) in codes)
    return _MULTISPACE.sub(" ", out).strip()


def sanitize(text: str, font_path: str) -> str:
    """Normalize typography, then drop anything the font still cannot draw."""
    return strip_unsupported(normalize_for_handwriting(text), font_path)
```

- [ ] **Step 5: Wepnij sanityzację w `src/field_content.py`**

Dodaj import na górze pliku:

```python
from text_sanitize import normalize_for_handwriting, sanitize
```

Zmień sygnaturę i ciało `generate_field_content` — nowy parametr `font_path` oraz sanityzacja każdej próbki **przed** pomiarem. Zamień nagłówek funkcji i jej ciało (od `sampler = ...` do `return text`) na:

```python
def generate_field_content(
    kind: str,
    vocab: Vocabulary,
    measure: Callable[[str], float],
    bbox_w: int,
    font_path: Optional[str] = None,
) -> str:
    """Build field content spanning ~30-100% of the field width.

    Args:
        kind: Field label — "t"/"f" (text), "n" (digits), "mix".
        vocab: Loaded Vocabulary.
        measure: Callable returning the rendered pixel width of a string
            (must already account for glyph stretch and letter tracking).
        bbox_w: Field width in pixels.
        font_path: Font the text will be rendered with. When given, content
            is sanitized against that font's glyph coverage BEFORE it is
            measured, so the returned string is exactly what gets drawn and
            recorded as ground truth. When None, only typographic
            normalization is applied.

    Returns:
        Content string whose measured width is <= 0.97 * bbox_w
        (empty string if even one character doesn't fit).
    """
    sampler = _UNIT_SAMPLERS.get(kind, _sample_text_unit)
    target = bbox_w * random.uniform(FILL_FRAC_MIN, FILL_FRAC_MAX)
    hard_cap = bbox_w * HARD_CAP_FRAC

    def sample_unit() -> str:
        raw = sampler(vocab)
        if font_path is None:
            return normalize_for_handwriting(raw)
        return sanitize(raw, font_path)

    text = _shrink_to_fit(sample_unit(), measure, hard_cap)
    if not text:
        return ""

    while True:
        candidate = text + " " + sample_unit()
        w = measure(candidate)
        if w > hard_cap:
            break
        text = candidate
        if w >= target:
            break
    return text
```

Upewnij się, że `Optional` jest zaimportowane (`from typing import Callable, Optional`).

- [ ] **Step 6: Przekaż font we wszystkich wywołaniach w `src/fill_form.py`**

Są trzy wywołania `generate_field_content`. Dopisz w każdym `font_path=font_path`:

```python
                line_text = generate_field_content(
                    content_kind, vocab, measure, bbox_w, font_path=font_path
                )
```

oraz

```python
        text = generate_field_content(
            content_kind, vocab, measure, bbox_w, font_path=font_path
        )
```

(dwa pierwsze to gałąź wielolinijkowa i jednolinijkowa; jeśli w pliku jest tylko dwa wywołania, popraw oba — liczba wywołań ma się zgadzać z tym, co faktycznie jest w kodzie).

- [ ] **Step 7: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_text_sanitize.py -v`
Expected: 10 passed (ostatni test jest wolniejszy — renderuje 1200 próbek przez 15 fontów)

Potem cały zestaw: `python -m pytest tests/ -q`
Expected: 39 passed

- [ ] **Step 8: Commit**

```bash
git add src/text_sanitize.py src/field_content.py src/fill_form.py requirements.txt tests/test_text_sanitize.py
git commit -m "Sanitize generated content against font glyph coverage"
```

---

### Task 2: Kreski długopisem

**Files:**
- Create: `src/artifacts.py`
- Test: `tests/test_artifacts_strokes.py`

**Interfaces:**
- Consumes: nic z wcześniejszych zadań (moduł samodzielny)
- Produces:
  - stałe `STROKE_PROB = 0.20`, `STROKE_COUNT_RANGE = (1, 3)`
  - `draw_stray_strokes(form: Image.Image, ink_color: tuple[int, int, int]) -> dict` — rysuje **w miejscu**, zwraca `{"count": int, "kinds": list[str]}`

- [ ] **Step 1: Napisz failujący test**

`tests/test_artifacts_strokes.py`:

```python
import random

import numpy as np
from PIL import Image

from artifacts import draw_stray_strokes, STROKE_PROB, STROKE_COUNT_RANGE


def _blank(w=800, h=1000):
    return Image.new("RGB", (w, h), (252, 250, 247))


def test_constants():
    assert STROKE_PROB == 0.20
    assert STROKE_COUNT_RANGE == (1, 3)


def test_draws_ink_in_place_and_reports_meta():
    random.seed(3)
    form = _blank()
    before = np.array(form).copy()
    meta = draw_stray_strokes(form, (28, 42, 120))
    after = np.array(form)

    assert not np.array_equal(before, after), "nothing was drawn"
    assert 1 <= meta["count"] <= 3
    assert len(meta["kinds"]) == meta["count"]
    assert set(meta["kinds"]) <= {"sweep", "tick", "scribble"}


def test_image_geometry_untouched():
    random.seed(4)
    form = _blank(640, 480)
    draw_stray_strokes(form, (20, 20, 28))
    assert form.size == (640, 480)
    assert form.mode == "RGB"


def test_uses_the_given_pen_colour():
    random.seed(5)
    form = _blank()
    ink = (200, 30, 30)
    draw_stray_strokes(form, ink)
    arr = np.array(form).reshape(-1, 3)
    changed = arr[(arr != np.array([252, 250, 247])).any(axis=1)]
    assert len(changed) > 0
    # every painted pixel is the pen colour (lines are drawn opaque)
    assert (changed == np.array(ink)).all(axis=1).mean() > 0.9


def test_ink_stays_inside_the_page():
    for seed in range(15):
        random.seed(seed)
        form = _blank(500, 700)
        draw_stray_strokes(form, (20, 20, 28))
        assert form.size == (500, 700)
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_artifacts_strokes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'artifacts'`

- [ ] **Step 3: Utwórz `src/artifacts.py` z nagłówkiem modułu i kreskami**

```python
"""Physical artifacts that end up on a filled paper form.

Applied after the text is written and before the scan simulation, because a
scanner captures them like anything else lying on the page:

    stray pen strokes — the pen accidentally touching the paper
    stamps            — clinic/doctor stamps, usually slapped on at an angle
    highlighter       — someone marking a line they care about

None of them change the image geometry, so ground-truth boxes produced by the
filling step stay valid to the pixel. Only rectangular stamps pressed straight
enough (|angle| <= STAMP_GT_MAX_ANGLE) contribute new boxes; everything else
here is deliberately unlabelled — the detector should learn that not every
dark mark on paper is a line of text.
"""

import math
import random

from PIL import Image, ImageDraw

# --- stray pen strokes ---
STROKE_PROB = 0.20
STROKE_COUNT_RANGE = (1, 3)
_STROKE_KINDS = ("sweep", "tick", "scribble")


def _stroke_points(
    x: float, y: float, length: float, angle_deg: float, curvature: float
) -> list[tuple[float, float]]:
    """Walk a pen tip across the page, drifting slightly each step."""
    points: list[tuple[float, float]] = []
    angle = math.radians(angle_deg)
    step = 4.0
    for _ in range(max(2, int(length / step))):
        points.append((x, y))
        angle += curvature + random.uniform(-0.03, 0.03)
        x += step * math.cos(angle)
        y += step * math.sin(angle)
    return points


def _scribble_points(x: float, y: float, width_px: float) -> list[tuple[float, float]]:
    """A tight back-and-forth zigzag, like someone testing whether a pen works."""
    n = random.randint(4, 9)
    dx = width_px / n
    amplitude = random.uniform(8.0, 26.0)
    return [
        (x + i * dx, y + (amplitude if i % 2 else -amplitude) + random.uniform(-3, 3))
        for i in range(n + 1)
    ]


def draw_stray_strokes(form: Image.Image, ink_color: tuple[int, int, int]) -> dict:
    """Draw 1-3 accidental pen marks on the form, in place.

    Uses the same pen as the handwriting — it is the same person's hand
    brushing the paper. No ground truth is produced: these are not text, and
    the detector benefits from seeing dark marks it must NOT report.

    Args:
        form: Filled form image (RGB), modified in place.
        ink_color: The pen colour used for this form's handwriting.

    Returns:
        Metadata dict with the stroke count and the kind of each stroke.
    """
    draw = ImageDraw.Draw(form)
    count = random.randint(*STROKE_COUNT_RANGE)
    kinds: list[str] = []

    for _ in range(count):
        kind = random.choice(_STROKE_KINDS)
        kinds.append(kind)
        x = random.uniform(0.03, 0.95) * form.width
        y = random.uniform(0.05, 0.95) * form.height

        if kind == "sweep":
            points = _stroke_points(
                x, y, random.uniform(150, 500),
                random.uniform(-180, 180), random.uniform(-0.05, 0.05),
            )
        elif kind == "tick":
            points = _stroke_points(
                x, y, random.uniform(30, 120),
                random.uniform(-180, 180), random.uniform(-0.12, 0.12),
            )
        else:
            points = _scribble_points(x, y, random.uniform(40, 150))

        draw.line(points, fill=ink_color, width=random.randint(1, 3), joint="curve")

    return {"count": count, "kinds": kinds}
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_artifacts_strokes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/artifacts.py tests/test_artifacts_strokes.py
git commit -m "Add stray pen stroke artifact (unlabelled negative examples)"
```

---

### Task 3: Marker (zakreślacz)

**Files:**
- Modify: `src/artifacts.py` (dopisz sekcję markera na końcu pliku)
- Test: `tests/test_artifacts_highlight.py`

**Interfaces:**
- Consumes: `artifacts` z Task 2 (ten sam moduł)
- Produces:
  - stałe `HIGHLIGHT_PROB = 0.12`, `HIGHLIGHT_COUNT_RANGE = (1, 3)`, `HIGHLIGHT_MIN_LUMA = 150.0`, `HIGHLIGHT_COLORS`
  - `draw_highlights(form: Image.Image, records: list[dict]) -> dict` — rysuje **w miejscu**, zwraca `{"count": int, "colors": list[list[int]], "targets": list[int]}`; `targets` to indeksy w **oryginalnej** liście `records`

- [ ] **Step 1: Napisz failujący test**

`tests/test_artifacts_highlight.py`:

```python
import random

import numpy as np
from PIL import Image, ImageDraw

from artifacts import (
    draw_highlights,
    HIGHLIGHT_PROB,
    HIGHLIGHT_COUNT_RANGE,
    HIGHLIGHT_MIN_LUMA,
    HIGHLIGHT_COLORS,
    _ensure_min_luma,
)


def _luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _form_with_text():
    """White page with two dark 'text lines' plus matching records."""
    form = Image.new("RGB", (600, 400), (255, 255, 255))
    d = ImageDraw.Draw(form)
    d.rectangle([50, 100, 300, 125], fill=(20, 20, 28))
    d.rectangle([50, 200, 250, 225], fill=(20, 20, 28))
    records = [
        {"label": "t", "source": "synthetic", "text": "abc", "bbox": [50, 100, 300, 125]},
        {"label": "p", "source": "printed", "text": None, "bbox": [50, 200, 250, 225]},
    ]
    return form, records


def test_constants():
    assert HIGHLIGHT_PROB == 0.12
    assert HIGHLIGHT_COUNT_RANGE == (1, 3)
    assert HIGHLIGHT_MIN_LUMA == 150.0
    assert all(_luma(c) >= HIGHLIGHT_MIN_LUMA for c in HIGHLIGHT_COLORS)


def test_ensure_min_luma_lightens_dark_colours():
    dark = (10, 10, 10)
    fixed = _ensure_min_luma(dark)
    assert _luma(fixed) >= HIGHLIGHT_MIN_LUMA
    bright = (255, 242, 60)
    assert _ensure_min_luma(bright) == bright  # already fine, left alone


def test_paper_gets_tinted_and_geometry_survives():
    random.seed(2)
    form, records = _form_with_text()
    meta = draw_highlights(form, records)

    assert form.size == (600, 400)
    assert 1 <= meta["count"] <= 2  # only two records available
    assert len(meta["colors"]) == meta["count"]
    assert all(0 <= i < len(records) for i in meta["targets"])

    arr = np.array(form)
    tinted = ((arr != 255).any(axis=2)).sum()
    assert tinted > 0


def test_text_stays_darker_than_the_band():
    """Success criterion 5: ink under the marker must remain readable."""
    for seed in range(10):
        random.seed(seed)
        form, records = _form_with_text()
        meta = draw_highlights(form, records)
        if not meta["targets"]:
            continue
        arr = np.array(form).astype(float)
        gray = arr @ np.array([0.299, 0.587, 0.114])
        for idx in meta["targets"]:
            x1, y1, x2, y2 = records[idx]["bbox"]
            ink = gray[y1 + 3:y2 - 3, x1 + 3:x2 - 3]
            band_above = gray[max(0, y1 - 2):y1, x1 + 3:x2 - 3]
            if ink.size and band_above.size:
                assert ink.mean() + 40 < band_above.mean()


def test_no_records_is_a_no_op():
    random.seed(1)
    form = Image.new("RGB", (200, 200), (255, 255, 255))
    before = np.array(form).copy()
    meta = draw_highlights(form, [])
    assert meta == {"count": 0, "colors": [], "targets": []}
    assert np.array_equal(before, np.array(form))


def test_handwritten_records_are_not_targeted():
    random.seed(7)
    form = Image.new("RGB", (400, 300), (255, 255, 255))
    records = [
        {"label": "f", "source": "handwritten", "text": None, "bbox": [10, 10, 200, 40]},
    ]
    meta = draw_highlights(form, records)
    assert meta["count"] == 0
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_artifacts_highlight.py -v`
Expected: FAIL — `ImportError: cannot import name 'draw_highlights'`

- [ ] **Step 3: Dopisz sekcję markera na końcu `src/artifacts.py`**

Rozszerz importy na górze pliku do:

```python
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
```

Dopisz na końcu pliku:

```python
# --- highlighter ---
HIGHLIGHT_PROB = 0.12
HIGHLIGHT_COUNT_RANGE = (1, 3)
# Multiplying a colour into white paper yields the colour itself, so the
# marker's own luminance decides how dark the band gets. Below this floor the
# photocopy profile (which crushes everything under ~80-115 to black) would
# swallow the writing underneath and leave a labelled but unreadable line.
HIGHLIGHT_MIN_LUMA = 150.0
HIGHLIGHT_COLORS: list[tuple[int, int, int]] = [
    (255, 242, 60),    # yellow
    (170, 255, 90),    # green
    (255, 140, 200),   # pink
    (255, 190, 70),    # orange
]
_HIGHLIGHT_SOURCES = ("synthetic", "printed")


def _luminance(color: tuple[int, int, int]) -> float:
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def _ensure_min_luma(
    color: tuple[int, int, int], min_luma: float = HIGHLIGHT_MIN_LUMA
) -> tuple[int, int, int]:
    """Lighten a colour toward white until it clears the readability floor."""
    r, g, b = color
    while _luminance((r, g, b)) < min_luma:
        r = min(255, int(r + (255 - r) * 0.15) + 1)
        g = min(255, int(g + (255 - g) * 0.15) + 1)
        b = min(255, int(b + (255 - b) * 0.15) + 1)
    return (r, g, b)


def _band_mask(size: tuple[int, int], bbox: list[int], tilt_deg: float) -> Image.Image:
    """Build the coverage mask of one marker stroke over a text bbox.

    Real highlighting overshoots the word, runs a little crooked and has a
    wobbly edge where the felt tip wandered.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = bbox

    pad_top = random.randint(2, 6)
    pad_bottom = random.randint(2, 6)
    x_start = x1 - random.randint(0, 8)
    x_end = x2 + random.randint(0, 12)
    phase = random.uniform(0.0, 2 * math.pi)
    wobble = random.uniform(1.0, 3.0)
    wavelength = random.uniform(30.0, 90.0)
    slope = math.tan(math.radians(tilt_deg))

    for x in range(max(0, x_start), min(size[0], x_end)):
        drift = (x - x_start) * slope
        top = y1 - pad_top + drift + wobble * math.sin(x / wavelength + phase)
        bottom = y2 + pad_bottom + drift + wobble * math.sin(x / wavelength + phase + 1.7)
        draw.line([(x, top), (x, bottom)], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(1.0))


def draw_highlights(form: Image.Image, records: list[dict]) -> dict:
    """Run a highlighter over 1-3 existing text lines, in place.

    Multiply blending keeps dark ink dark while the paper takes the colour,
    which is exactly how a translucent marker behaves. Ground truth is
    untouched — the text is still there, just on a coloured background.

    Args:
        form: Filled form image (RGB), modified in place.
        records: Ground-truth records; only "synthetic" and "printed" lines
            are eligible targets.

    Returns:
        Metadata with the count, the colours used and the indices (into
        `records`) of the highlighted lines.
    """
    eligible = [
        i for i, r in enumerate(records) if r["source"] in _HIGHLIGHT_SOURCES
    ]
    if not eligible:
        return {"count": 0, "colors": [], "targets": []}

    count = min(random.randint(*HIGHLIGHT_COUNT_RANGE), len(eligible))
    targets = random.sample(eligible, count)

    arr = np.array(form, dtype=np.float32)
    colors: list[list[int]] = []

    for index in targets:
        color = _ensure_min_luma(random.choice(HIGHLIGHT_COLORS))
        colors.append(list(color))
        mask = _band_mask(form.size, records[index]["bbox"], random.uniform(-2.0, 2.0))
        coverage = (np.array(mask, dtype=np.float32) / 255.0)[:, :, None]
        tinted = arr * (np.array(color, dtype=np.float32) / 255.0)
        arr = arr * (1.0 - coverage) + tinted * coverage

    form.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    return {"count": count, "colors": colors, "targets": targets}
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_artifacts_highlight.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/artifacts.py tests/test_artifacts_highlight.py
git commit -m "Add highlighter artifact with readability floor"
```

---

### Task 4: Pieczątki

**Files:**
- Create: `resources/fonts_print/DejaVuSans.ttf`, `resources/fonts_print/DejaVuSans-Bold.ttf`
- Modify: `src/artifacts.py` (dopisz sekcję pieczątek)
- Test: `tests/test_artifacts_stamp.py`

**Interfaces:**
- Consumes: `Vocabulary.get_random_text("department"|"doctor_name")`, `transforms.rotate_bbox(bbox, angle_deg, src_size, dst_size)`, `text_sanitize.sanitize`
- Produces:
  - stałe `STAMP_PROB = 0.30`, `STAMP_ROUND_PROB = 0.30`, `STAMP_ANGLE_MAX = 20.0`, `STAMP_GT_MAX_ANGLE = 5.0`, `STAMP_PLACEMENT_TRIES = 8`, `STAMP_MAX_COVER_FRAC = 0.15`, `PRINT_FONT_DIR = "resources/fonts_print"`
  - `draw_stamp(form, vocab, records, print_font_dir=PRINT_FONT_DIR) -> tuple[Optional[dict], list[dict]]` — rysuje w miejscu; zwraca `(meta, new_records)`, gdzie `meta is None` oznacza brak miejsca na pieczątkę; rekordy mają kształt `{"label": "stamp", "source": "stamp", "text": str, "bbox": [x1, y1, x2, y2]}`

- [ ] **Step 1: Skopiuj czcionkę drukowaną do repo**

```bash
cd /c/Users/tomek/Desktop/text-gen
mkdir -p resources/fonts_print
python -c "
import matplotlib, shutil
from pathlib import Path
src = Path(matplotlib.__file__).parent / 'mpl-data' / 'fonts' / 'ttf'
for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
    shutil.copy2(src / name, Path('resources/fonts_print') / name)
    print('copied', name)
"
```

Expected: dwa pliki skopiowane. Katalog musi być **poza** `resources/fonts`, bo `find_fonts` zbiera stamtąd czcionki odręczne i drukowana nie może trafić do puli pisma.

- [ ] **Step 2: Napisz failujący test**

`tests/test_artifacts_stamp.py`:

```python
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from artifacts import (
    draw_stamp,
    STAMP_PROB,
    STAMP_ROUND_PROB,
    STAMP_ANGLE_MAX,
    STAMP_GT_MAX_ANGLE,
    STAMP_MAX_COVER_FRAC,
    PRINT_FONT_DIR,
    _overlaps_text,
)
from vocabulary import Vocabulary

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(REPO / "resources"))


def _blank(w=1654, h=2338):
    return Image.new("RGB", (w, h), (252, 250, 247))


def test_constants():
    assert STAMP_PROB == 0.30
    assert STAMP_ROUND_PROB == 0.30
    assert STAMP_ANGLE_MAX == 20.0
    assert STAMP_GT_MAX_ANGLE == 5.0


def test_print_font_is_shipped():
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        assert (REPO / PRINT_FONT_DIR / name).exists(), f"{name} missing from repo"


def test_stamp_is_drawn_and_geometry_survives(vocab):
    random.seed(1)
    form = _blank()
    before = np.array(form).copy()
    meta, records = draw_stamp(form, vocab, [])
    assert meta is not None
    assert form.size == (1654, 2338)
    assert not np.array_equal(before, np.array(form))
    assert meta["shape"] in ("rect", "round")
    assert abs(meta["angle_deg"]) <= STAMP_ANGLE_MAX


def test_ground_truth_rule_holds_over_many_draws(vocab):
    """Only straight rectangular stamps may produce boxes."""
    seen_gt = seen_no_gt = 0
    for seed in range(60):
        random.seed(seed)
        form = _blank()
        meta, records = draw_stamp(form, vocab, [])
        if meta is None:
            continue
        should_label = meta["shape"] == "rect" and abs(meta["angle_deg"]) <= STAMP_GT_MAX_ANGLE
        assert meta["in_gt"] == should_label
        if should_label:
            seen_gt += 1
            assert records, "labelled stamp produced no records"
            for rec in records:
                assert rec["source"] == "stamp"
                assert rec["label"] == "stamp"
                assert isinstance(rec["text"], str) and rec["text"]
                x1, y1, x2, y2 = rec["bbox"]
                assert x2 > x1 and y2 > y1
                assert 0 <= x1 and 0 <= y1
                assert x2 <= form.width and y2 <= form.height
        else:
            seen_no_gt += 1
            assert records == []
    assert seen_gt > 0, "no labelled stamp in 60 draws"
    assert seen_no_gt > 0, "no unlabelled stamp in 60 draws"


def test_round_stamps_never_labelled(vocab):
    for seed in range(60):
        random.seed(seed)
        meta, records = draw_stamp(_blank(), vocab, [])
        if meta and meta["shape"] == "round":
            assert meta["in_gt"] is False
            assert records == []


def test_stamp_bboxes_land_on_actual_ink(vocab):
    """A labelled box must contain dark pixels — proof the maths lines up."""
    for seed in range(40):
        random.seed(seed)
        form = _blank()
        meta, records = draw_stamp(form, vocab, [])
        if not records:
            continue
        gray = np.array(form.convert("L"))
        for rec in records:
            x1, y1, x2, y2 = rec["bbox"]
            patch = gray[y1:y2, x1:x2]
            assert patch.size > 0
            assert (patch < 160).sum() > 0, "stamp bbox contains no ink"


def test_overlap_rejection_helper():
    records = [{"bbox": [100, 100, 200, 140], "source": "printed"}]
    # fully covering the record -> rejected
    assert _overlaps_text((90, 90, 260, 200), records) is True
    # far away -> accepted
    assert _overlaps_text((500, 500, 700, 600), records) is False


def test_stamp_skipped_when_page_is_full(vocab):
    """Bottom third packed with text: no room, no stamp, no crash."""
    random.seed(9)
    form = _blank(800, 1000)
    records = [
        {"label": "p", "source": "printed", "text": None,
         "bbox": [0, y, 800, y + 30]}
        for y in range(560, 1000, 32)
    ]
    meta, new_records = draw_stamp(form, vocab, records)
    assert meta is None
    assert new_records == []
```

- [ ] **Step 3: Uruchom test — ma failować**

Run: `python -m pytest tests/test_artifacts_stamp.py -v`
Expected: FAIL — `ImportError: cannot import name 'draw_stamp'`

- [ ] **Step 4: Dopisz sekcję pieczątek na końcu `src/artifacts.py`**

Rozszerz importy na górze pliku do:

```python
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from text_sanitize import sanitize
from transforms import rotate_bbox
from vocabulary import Vocabulary
```

Dopisz na końcu pliku:

```python
# --- stamps ---
STAMP_PROB = 0.30
STAMP_ROUND_PROB = 0.30
STAMP_ANGLE_MAX = 20.0
# Above this angle an axis-aligned box around slanted text gets too loose to
# be useful ground truth, so those stamps stay unlabelled noise.
STAMP_GT_MAX_ANGLE = 5.0
STAMP_PLACEMENT_TRIES = 8
STAMP_MAX_COVER_FRAC = 0.15
PRINT_FONT_DIR = "resources/fonts_print"
STAMP_COLORS: list[tuple[int, int, int]] = [
    (30, 60, 140),   # blue pad
    (35, 35, 40),    # black pad
    (150, 40, 45),   # red pad
]


@lru_cache(maxsize=16)
def _print_font(font_dir: str, size: int, bold: bool) -> ImageFont.FreeTypeFont:
    """Load the printed font stamps are set in (cached: same file every call)."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(Path(font_dir) / name), size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _stamp_lines(vocab: Vocabulary, font_dir: str) -> list[str]:
    """Compose the 2-4 lines a clinic stamp carries."""
    department, _ = vocab.get_random_text("department")
    doctor, _ = vocab.get_random_text("doctor_name")
    lines = [department, doctor]
    if random.random() < 0.7:
        lines.append(f"nr prawa wyk. zawodu {random.randint(1000000, 9999999)}")
    if random.random() < 0.3:
        lines.append(
            f"NIP {random.randint(100, 999)}-{random.randint(100, 999)}"
            f"-{random.randint(10, 99)}-{random.randint(10, 99)}"
        )
    font_path = str(Path(font_dir) / "DejaVuSans.ttf")
    return [sanitize(line, font_path) for line in lines[:4]]


def _render_rect_stamp(
    lines: list[str], font: ImageFont.FreeTypeFont, color: tuple[int, int, int]
) -> tuple[Image.Image, list[list[int]]]:
    """Draw a bordered rectangular stamp; also return each line's ink bbox."""
    pad_x, pad_y = 14, 10
    line_h = int(font.size * 1.5)
    width = int(max(font.getlength(t) for t in lines) + 2 * pad_x)
    height = int(len(lines) * line_h + 2 * pad_y)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    border = random.choice((1, 2))
    draw.rectangle([0, 0, width - 1, height - 1], outline=color + (255,), width=border)
    if random.random() < 0.35:
        gap = border + 3
        draw.rectangle(
            [gap, gap, width - 1 - gap, height - 1 - gap],
            outline=color + (255,), width=1,
        )

    boxes: list[list[int]] = []
    for i, text in enumerate(lines):
        y = pad_y + i * line_h
        draw.text((pad_x, y), text, font=font, fill=color + (255,))
        left, top, right, bottom = draw.textbbox((pad_x, y), text, font=font)
        boxes.append([int(left), int(top), int(right), int(bottom)])
    return layer, boxes


def _render_round_stamp(
    lines: list[str], font: ImageFont.FreeTypeFont, color: tuple[int, int, int]
) -> Image.Image:
    """Draw a circular stamp: arc text on top, straight text in the middle."""
    diameter = random.randint(180, 300)
    layer = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        [2, 2, diameter - 3, diameter - 3],
        outline=color + (255,), width=random.choice((2, 3)),
    )
    inset = random.randint(10, 18)
    draw.ellipse(
        [inset, inset, diameter - 1 - inset, diameter - 1 - inset],
        outline=color + (255,), width=1,
    )

    arc_text = lines[0][:32]
    centre = diameter / 2
    radius = centre - inset - font.size * 0.7
    span = math.radians(min(200.0, 12.0 * len(arc_text)))
    start = -math.pi / 2 - span / 2
    for i, char in enumerate(arc_text):
        angle = start + span * i / max(1, len(arc_text) - 1)
        glyph = Image.new("RGBA", (font.size * 2, font.size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text(
            (font.size // 2, font.size // 2), char, font=font, fill=color + (255,)
        )
        glyph = glyph.rotate(-math.degrees(angle) - 90, resample=Image.BICUBIC)
        layer.alpha_composite(
            glyph,
            (int(centre + radius * math.cos(angle) - glyph.width / 2),
             int(centre + radius * math.sin(angle) - glyph.height / 2)),
        )

    if len(lines) > 1:
        text = lines[1][:22]
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (centre - (right - left) / 2, centre - (bottom - top) / 2),
            text, font=font, fill=color + (255,),
        )
    return layer


def _apply_ink_unevenness(layer: Image.Image) -> Image.Image:
    """Fade parts of the stamp: a rubber pad never transfers ink evenly."""
    width, height = layer.size
    small = np.random.random((max(2, height // 12), max(2, width // 12)))
    noise = np.array(
        Image.fromarray((small * 255).astype(np.uint8)).resize(
            (width, height), Image.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    factor = np.clip(0.45 + noise * 0.85, 0.0, 1.0)
    alpha = np.array(layer.getchannel("A"), dtype=np.float32) * factor
    layer.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    return layer


def _overlaps_text(
    rect: tuple[int, int, int, int],
    records: list[dict],
    max_cover: float = STAMP_MAX_COVER_FRAC,
) -> bool:
    """True if the rect would bury more than `max_cover` of any text line."""
    rx1, ry1, rx2, ry2 = rect
    for record in records:
        x1, y1, x2, y2 = record["bbox"]
        area = max(1, (x2 - x1) * (y2 - y1))
        overlap_w = max(0, min(rx2, x2) - max(rx1, x1))
        overlap_h = max(0, min(ry2, y2) - max(ry1, y1))
        if overlap_w * overlap_h / area > max_cover:
            return True
    return False


def _find_spot(
    form_size: tuple[int, int], stamp_size: tuple[int, int], records: list[dict]
) -> Optional[tuple[int, int]]:
    """Look for a free patch in the bottom third, where stamps really land."""
    form_w, form_h = form_size
    stamp_w, stamp_h = stamp_size
    if stamp_w >= form_w or stamp_h >= form_h:
        return None

    y_low = int(form_h * 0.60)
    y_high = max(y_low, form_h - stamp_h - 10)
    x_high = max(10, form_w - stamp_w - 10)
    for _ in range(STAMP_PLACEMENT_TRIES):
        y = random.randint(y_low, y_high)
        x = random.randint(10, x_high)
        if not _overlaps_text((x, y, x + stamp_w, y + stamp_h), records):
            return x, y
    return None


def draw_stamp(
    form: Image.Image,
    vocab: Vocabulary,
    records: list[dict],
    print_font_dir: str = PRINT_FONT_DIR,
) -> tuple[Optional[dict], list[dict]]:
    """Press one clinic stamp onto the form, in place.

    Rectangular stamps pressed straight (|angle| <= STAMP_GT_MAX_ANGLE) hand
    back one ground-truth record per text line, because an axis-aligned box
    still hugs the text at that angle. Round stamps and heavily tilted ones
    are unlabelled noise.

    Args:
        form: Filled form image (RGB), modified in place.
        vocab: Vocabulary supplying department and doctor names.
        records: Existing ground-truth records, used to avoid burying text.
        print_font_dir: Directory holding the printed stamp font.

    Returns:
        Tuple of (metadata, new records). Metadata is None when no free spot
        was found, in which case nothing was drawn and no records are returned.
    """
    lines = _stamp_lines(vocab, print_font_dir)
    color = random.choice(STAMP_COLORS)
    is_round = random.random() < STAMP_ROUND_PROB
    font = _print_font(print_font_dir, random.randint(14, 22), random.random() < 0.4)

    if is_round:
        layer = _render_round_stamp(lines, font, color)
        line_boxes: list[list[int]] = []
    else:
        layer, line_boxes = _render_rect_stamp(lines, font, color)

    layer = _apply_ink_unevenness(layer)
    if random.random() < 0.6:
        layer = layer.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.6)))

    angle = random.uniform(-STAMP_ANGLE_MAX, STAMP_ANGLE_MAX)
    src_size = layer.size
    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=True)

    spot = _find_spot(form.size, rotated.size, records)
    if spot is None:
        return None, []
    x, y = spot
    form.paste(rotated, (x, y), rotated)

    in_gt = (not is_round) and abs(angle) <= STAMP_GT_MAX_ANGLE
    new_records: list[dict] = []
    if in_gt:
        for text, box in zip(lines, line_boxes):
            bx1, by1, bx2, by2 = rotate_bbox(
                tuple(box), angle, src_size, rotated.size
            )
            new_records.append({
                "label": "stamp",
                "source": "stamp",
                "text": text,
                "bbox": [bx1 + x, by1 + y, bx2 + x, by2 + y],
            })

    meta = {
        "shape": "round" if is_round else "rect",
        "angle_deg": round(angle, 2),
        "color": list(color),
        "in_gt": in_gt,
        "lines": len(lines),
    }
    return meta, new_records
```

- [ ] **Step 5: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_artifacts_stamp.py -v`
Expected: 8 passed

Jeśli `test_stamp_bboxes_land_on_actual_ink` failuje, sprawdź kolejność argumentów `rotate_bbox(bbox, angle_deg, src_size, dst_size)` — `src_size` to rozmiar warstwy **przed** obrotem, `dst_size` po obrocie z `expand=True`.

- [ ] **Step 6: Commit**

```bash
git add resources/fonts_print src/artifacts.py tests/test_artifacts_stamp.py
git commit -m "Add stamp artifact with ground truth for straight rectangular stamps"
```

---

### Task 5: Wpięcie artefaktów w pipeline

**Files:**
- Modify: `src/fill_form.py` (import, sygnatura i koniec `fill_single_form`)
- Modify: `src/generate_yolo_dataset.py` (metadane)
- Modify: `tests/test_fill_single_form.py` (izolacja istniejących testów)
- Test: `tests/test_fill_form_artifacts.py`

**Interfaces:**
- Consumes: `artifacts.draw_stamp/draw_highlights/draw_stray_strokes` oraz stałe `STAMP_PROB/HIGHLIGHT_PROB/STROKE_PROB` (Task 2–4)
- Produces:
  - `fill_single_form(..., enable_artifacts: bool = True)` — nowy parametr na końcu listy
  - nowy klucz w zwracanym słowniku: `"artifacts": {"stamp": dict|None, "highlights": dict|None, "strokes": dict|None}`
  - rekordy pieczątki dołączone do `records` (`source: "stamp"`)

- [ ] **Step 1: Napisz failujący test**

`tests/test_fill_form_artifacts.py`:

```python
import random
from pathlib import Path

import pytest
from PIL import Image

from fill_form import fill_single_form, EXCLUDED_FONTS
from renderer import find_fonts
from transforms import AugmentConfig
from vocabulary import Vocabulary

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(REPO / "resources"))


@pytest.fixture(scope="module")
def font_path():
    fonts = [f for f in find_fonts(str(REPO / "resources/fonts"))
             if Path(f).name not in EXCLUDED_FONTS]
    assert fonts
    return fonts[0]


FIELDS = [
    {"label": "p", "x_min": 40, "y_min": 20, "x_max": 400, "y_max": 50},
    {"label": "t", "x_min": 40, "y_min": 80, "x_max": 700, "y_max": 125},
    {"label": "n", "x_min": 40, "y_min": 160, "x_max": 500, "y_max": 200},
]


def _form(tmp_path, w=800, h=1200):
    p = tmp_path / "blank.png"
    Image.new("RGB", (w, h), (252, 250, 247)).save(p)
    return p


def _config():
    config = AugmentConfig()
    config.paper.enabled = False
    config.scan.enabled = False
    return config


def _run(tmp_path, vocab, font_path, **kwargs):
    return fill_single_form(
        form_path=_form(tmp_path), fields=FIELDS, vocab=vocab,
        font_path=font_path, config=_config(), pipeline=None,
        apply_scan=False, empty_field_range=(0.0, 0.0), **kwargs,
    )


def test_artifacts_disabled_produces_no_artifact_records(tmp_path, vocab, font_path):
    random.seed(1)
    result = _run(tmp_path, vocab, font_path, enable_artifacts=False)
    assert result["artifacts"] == {"stamp": None, "highlights": None, "strokes": None}
    assert all(r["source"] != "stamp" for r in result["records"])


def test_artifact_meta_key_always_present(tmp_path, vocab, font_path):
    random.seed(2)
    result = _run(tmp_path, vocab, font_path)
    assert set(result["artifacts"]) == {"stamp", "highlights", "strokes"}


def test_stamp_records_appear_and_are_well_formed(tmp_path, vocab, font_path):
    """Over many seeds at least one stamp must land in the ground truth."""
    found = False
    for seed in range(40):
        random.seed(seed)
        result = _run(tmp_path, vocab, font_path)
        stamps = [r for r in result["records"] if r["source"] == "stamp"]
        if not stamps:
            continue
        found = True
        assert result["artifacts"]["stamp"]["in_gt"] is True
        for rec in stamps:
            x1, y1, x2, y2 = rec["bbox"]
            assert x2 > x1 and y2 > y1
            assert isinstance(rec["text"], str) and rec["text"]
        break
    assert found, "no stamp reached the ground truth in 40 seeds"


def test_artifacts_never_change_text_bboxes(tmp_path, vocab, font_path):
    """Same seed, artifacts on vs off: the text records must be identical."""
    random.seed(5)
    with_art = _run(tmp_path, vocab, font_path)
    random.seed(5)
    without = _run(tmp_path, vocab, font_path, enable_artifacts=False)

    text_only = [r for r in with_art["records"] if r["source"] != "stamp"]
    assert len(text_only) == len(without["records"])
    for a, b in zip(text_only, without["records"]):
        assert a["bbox"] == b["bbox"]
        assert a["text"] == b["text"]
        assert a["source"] == b["source"]


def test_image_size_unchanged_by_artifacts(tmp_path, vocab, font_path):
    random.seed(6)
    result = _run(tmp_path, vocab, font_path)
    assert result["image"].size == (800, 1200)
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_fill_form_artifacts.py -v`
Expected: FAIL — `TypeError: fill_single_form() got an unexpected keyword argument 'enable_artifacts'`

- [ ] **Step 3: Wepnij artefakty w `src/fill_form.py`**

Dodaj import obok pozostałych:

```python
from artifacts import (
    draw_stamp,
    draw_highlights,
    draw_stray_strokes,
    STAMP_PROB,
    HIGHLIGHT_PROB,
    STROKE_PROB,
)
```

Dopisz parametr na końcu sygnatury `fill_single_form`:

```python
    empty_field_range: tuple[float, float] = (0.0, 0.40),
    enable_artifacts: bool = True,
) -> dict:
```

W docstringu, w sekcji `Args:`, dopisz po opisie `empty_field_range`:

```
        enable_artifacts: Whether to add paper artifacts (stamp, highlighter,
            stray pen strokes) after the text is written. Off in tests that
            need an exactly predictable record list.
```

a w sekcji `Returns:` dopisz do wyliczanki kluczy:

```
            artifacts (dict) — {"stamp": ..., "highlights": ..., "strokes": ...},
                each None when that artifact did not fire
```

Wstaw blok artefaktów **między** pętlą po polach a symulacją skanu — czyli tuż przed linią `# Scan simulation is photometric only`:

```python
    # Paper artifacts: stamped, highlighted, then accidentally scribbled on —
    # the order a real document collects them. All are photometric-only, so
    # the ground-truth boxes recorded above stay valid to the pixel.
    artifact_meta: dict = {"stamp": None, "highlights": None, "strokes": None}
    if enable_artifacts:
        if random.random() < STAMP_PROB:
            stamp_meta, stamp_records = draw_stamp(form, vocab, records)
            if stamp_meta is not None:
                artifact_meta["stamp"] = stamp_meta
                records.extend(stamp_records)
        if random.random() < HIGHLIGHT_PROB:
            artifact_meta["highlights"] = draw_highlights(form, records)
        if random.random() < STROKE_PROB:
            artifact_meta["strokes"] = draw_stray_strokes(form, ink_color)
```

Dopisz klucz do zwracanego słownika, po `"multiline_fields": multiline_fields,`:

```python
        "artifacts": artifact_meta,
```

- [ ] **Step 4: Odizoluj istniejące testy w `tests/test_fill_single_form.py`**

Trzy testy zakładają, że rekordy zawierają wyłącznie wypełnienia i tekst drukowany; pieczątka by je psuła. W funkcji pomocniczej `_run` w tym pliku dopisz `enable_artifacts=False` do wywołania `fill_single_form`:

```python
    return fill_single_form(
        form_path=_blank_form(tmp_path),
        fields=FIELDS,
        vocab=vocab,
        font_path=font_path,
        config=config,
        pipeline=None,
        apply_scan=False,
        enable_artifacts=False,
        **kwargs,
    )
```

- [ ] **Step 5: Przekaż metadane artefaktów w `src/generate_yolo_dataset.py`**

W bloku budującym `metadata` dopisz po `"multiline_fields": result["multiline_fields"],`:

```python
                    "artifacts": result["artifacts"],
```

- [ ] **Step 6: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_fill_form_artifacts.py -v`
Expected: 5 passed

Potem cały zestaw: `python -m pytest tests/ -q`
Expected: 63 passed

- [ ] **Step 7: Commit**

```bash
git add src/fill_form.py src/generate_yolo_dataset.py tests/test_fill_single_form.py tests/test_fill_form_artifacts.py
git commit -m "Wire paper artifacts into the form filling pipeline"
```

---

### Task 6: Próbka do przeglądu (BRAMKA — czeka na Tomka)

**Files:**
- Brak zmian w kodzie (chyba że przegląd coś wykryje)
- Wygenerowane (poza gitem): `output/sample_artifacts/`

**Interfaces:**
- Consumes: cały pipeline z Task 1–5
- Produces: obrazy + wizualizacje do obejrzenia przez Tomka; **zgoda Tomka jest warunkiem przejścia do Task 7**

- [ ] **Step 1: Wygeneruj próbkę na nowych czcionkach**

```bash
cd /c/Users/tomek/Desktop/text-gen
python src/generate_yolo_dataset.py --templates-dir templates \
    --output-dir output/sample_artifacts --variants-per-form 1 --seed 7
```

Expected: 40 obrazów, bez błędów.

- [ ] **Step 2: Zbierz statystyki i znajdź reprezentatywne obrazy**

```bash
python -c "
import json, glob, collections
rows = []
for m in sorted(glob.glob('output/sample_artifacts/metadata/*.json')):
    d = json.load(open(m, encoding='utf-8'))
    a = d['artifacts']
    rows.append((d['output_image'], d['font'], d['scan_augmentation']['profile'],
                 (a['stamp'] or {}).get('shape'), (a['stamp'] or {}).get('in_gt'),
                 bool(a['highlights']), bool(a['strokes'])))
print('obrazy z pieczatka :', sum(1 for r in rows if r[3]))
print('  w tym w GT       :', sum(1 for r in rows if r[4]))
print('obrazy z markerem  :', sum(1 for r in rows if r[5]))
print('obrazy z kreskami  :', sum(1 for r in rows if r[6]))
print('uzyte czcionki     :', len({r[1] for r in rows}))
print()
for r in rows:
    if r[3] or r[5] or r[6]:
        print(f'{r[0]:44} {r[2]:11} stamp={r[3] or \"-\":5} gt={r[4]} hl={r[5]} str={r[6]}')
"
```

Expected: ~12 pieczątek (z czego kilka w GT), ~5 markerów, ~8 z kreskami, po sanityzacji żadnych kwadracików.

- [ ] **Step 3: Wygeneruj wizualizacje dla całej próbki**

```bash
mkdir -p output/sample_artifacts/viz
for img in output/sample_artifacts/images/*.jpg; do
  python src/visualize_yolo.py "$img" \
      --output "output/sample_artifacts/viz/$(basename ${img%.jpg})_viz.jpg" > /dev/null
done
ls output/sample_artifacts/viz | wc -l
```

Expected: 40 plików.

- [ ] **Step 4: Przedstaw próbkę Tomkowi i CZEKAJ na akceptację**

Podaj: statystyki z kroku 2 oraz ścieżki do 4–6 wizualizacji pokrywających: pieczątkę prostokątną **w GT**, pieczątkę okrągłą (bez GT), marker na profilu ksero, kreski, oraz obraz na jednej z nowych czcionek (Marek/Pokladowski).

**Nie przechodź do Task 7, dopóki Tomek nie zaakceptuje próbki.** Jeśli zgłosi poprawki — nanieś je, zregeneruj próbkę i pokaż ponownie.

---

### Task 7: Karta zbioru i rejestr

**Files:**
- Create: `src/dataset_card.py`
- Create: `docs/datasets.md`
- Modify: `src/generate_yolo_dataset.py` (CLI, liczniki, zapis karty)
- Test: `tests/test_dataset_card.py`

**Interfaces:**
- Consumes: stałe z `fill_form` (`PEN_FADE_PROB`, `V_OVERFLOW_FRAC`, `FORM_FONT_SIZE_RANGE`, `SCAN_PROFILES`, `MULTILINE_MAX_LINES`, `LINE_PITCH_RANGE`) i z `artifacts` (`STAMP_PROB`, `STROKE_PROB`, `HIGHLIGHT_PROB`, `STAMP_ANGLE_MAX`, `STAMP_GT_MAX_ANGLE`, `STAMP_ROUND_PROB`, `HIGHLIGHT_MIN_LUMA`), oraz `field_content.MEDICAL_TEXT_PROB`, `generate_yolo_dataset.PARTIAL_BASE_PROB`
- Produces:
  - `augmentation_settings() -> dict`
  - `git_info(repo_root: Path) -> dict` → `{"commit": str|None, "dirty": bool}`
  - `build_card(...) -> dict`
  - `write_card(output_dir: Path, card: dict) -> Path`
  - `update_registry(registry_path: Path, card: dict) -> None`
  - CLI: `--dataset-name`, `--note`

- [ ] **Step 1: Napisz failujący test**

`tests/test_dataset_card.py`:

```python
import json
from pathlib import Path

from dataset_card import (
    augmentation_settings,
    git_info,
    build_card,
    write_card,
    update_registry,
)

REPO = Path(__file__).resolve().parent.parent


def test_settings_are_read_from_the_code():
    settings = augmentation_settings()
    assert settings["stamp_prob"] == 0.30
    assert settings["stroke_prob"] == 0.20
    assert settings["highlight_prob"] == 0.12
    assert settings["pen_fade_prob"] == 0.15
    assert settings["form_font_size_range"] == [26, 40]
    assert settings["stamp_gt_max_angle"] == 5.0
    assert settings["highlight_min_luma"] == 150.0
    assert ["clean_color", 0.45] in settings["scan_profiles"]


def test_settings_are_json_serialisable():
    json.dumps(augmentation_settings())


def test_git_info_reads_the_repo():
    info = git_info(REPO)
    assert info["commit"] is None or len(info["commit"]) == 7
    assert isinstance(info["dirty"], bool)


def test_git_info_survives_a_non_repo(tmp_path):
    info = git_info(tmp_path)
    assert info == {"commit": None, "dirty": False}


def _card(name="demo", images=10, annotations=100):
    return build_card(
        name=name,
        command="python src/generate_yolo_dataset.py --output-dir output/demo",
        seed=42,
        repo_root=REPO,
        counts={"images": images, "annotations": annotations, "templates": 4},
        sources={"printed": 60, "synthetic": 30, "stamp": 10},
        fonts=["Marek_1-Regular.ttf"],
        observed={"scan_profiles": {"grayscale": 10}, "bases": {"blank": 10}},
        note="proba",
    )


def test_card_has_every_required_section():
    card = _card()
    for key in ("name", "created", "git_commit", "git_dirty", "command", "seed",
                "counts", "sources", "fonts", "augmentations", "observed", "note"):
        assert key in card, key
    assert card["augmentations"]["stamp_prob"] == 0.30


def test_write_card_lands_next_to_the_data(tmp_path):
    path = write_card(tmp_path, _card())
    assert path == tmp_path / "dataset_card.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["name"] == "demo"
    assert loaded["counts"]["images"] == 10


def test_registry_is_created_then_updated_in_place(tmp_path):
    registry = tmp_path / "datasets.md"

    update_registry(registry, _card(name="alpha", images=10))
    update_registry(registry, _card(name="beta", images=20))
    text = registry.read_text(encoding="utf-8")
    assert text.count("| alpha |") == 1
    assert text.count("| beta |") == 1

    # regenerating the same dataset replaces its row instead of duplicating
    update_registry(registry, _card(name="alpha", images=999))
    text = registry.read_text(encoding="utf-8")
    assert text.count("| alpha |") == 1
    assert "999" in text
    assert text.count("| beta |") == 1
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `python -m pytest tests/test_dataset_card.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_card'`

- [ ] **Step 3: Napisz `src/dataset_card.py`**

```python
"""Provenance for generated datasets.

Every run drops a dataset_card.json next to the images and adds a row to a
registry table in the repo, so months later it is still possible to answer
"what exactly was in that dataset and how was it made?".

The augmentation section is dumped straight from the code's own constants —
nothing is retyped by hand, so the card cannot drift away from what the
generator actually did.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import artifacts
import fill_form
import field_content
import generate_yolo_dataset

REGISTRY_HEADER = "# Wygenerowane zbiory danych\n"
REGISTRY_INTRO = (
    "\nJeden wiersz na zbiór. Karta z pełnymi ustawieniami leży w folderze\n"
    "zbioru jako `dataset_card.json` (foldery z danymi nie są w repo).\n\n"
)
_TABLE_HEAD = (
    "| Data | Nazwa | Commit | Seed | Obrazy | Anotacje | Notatka |\n"
    "|---|---|---|---|---|---|---|\n"
)


def augmentation_settings() -> dict:
    """Dump every augmentation knob the generator currently uses."""
    return {
        "stamp_prob": artifacts.STAMP_PROB,
        "stamp_round_prob": artifacts.STAMP_ROUND_PROB,
        "stamp_angle_max": artifacts.STAMP_ANGLE_MAX,
        "stamp_gt_max_angle": artifacts.STAMP_GT_MAX_ANGLE,
        "stroke_prob": artifacts.STROKE_PROB,
        "highlight_prob": artifacts.HIGHLIGHT_PROB,
        "highlight_min_luma": artifacts.HIGHLIGHT_MIN_LUMA,
        "pen_fade_prob": fill_form.PEN_FADE_PROB,
        "v_overflow_frac": fill_form.V_OVERFLOW_FRAC,
        "form_font_size_range": list(fill_form.FORM_FONT_SIZE_RANGE),
        "min_font_size": fill_form.MIN_FONT_SIZE,
        "max_font_size": fill_form.MAX_FONT_SIZE,
        "multiline_max_lines": fill_form.MULTILINE_MAX_LINES,
        "line_pitch_range": list(fill_form.LINE_PITCH_RANGE),
        "scan_profiles": [[name, weight] for name, weight in fill_form.SCAN_PROFILES],
        "medical_text_prob": field_content.MEDICAL_TEXT_PROB,
        "fill_frac_range": [field_content.FILL_FRAC_MIN, field_content.FILL_FRAC_MAX],
        "partial_base_prob": generate_yolo_dataset.PARTIAL_BASE_PROB,
    }


def git_info(repo_root: Path) -> dict:
    """Short commit hash plus whether the tree had uncommitted changes.

    Without the dirty flag the commit hash can quietly lie about what code
    produced the data.
    """
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo_root),
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "--short", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": False}
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def build_card(
    name: str,
    command: str,
    seed: Optional[int],
    repo_root: Path,
    counts: dict,
    sources: dict,
    fonts: list[str],
    observed: dict,
    note: str = "",
) -> dict:
    """Assemble the full provenance record for one generation run."""
    info = git_info(repo_root)
    return {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "command": command,
        "seed": seed,
        "counts": counts,
        "sources": sources,
        "fonts": sorted(fonts),
        "augmentations": augmentation_settings(),
        "observed": observed,
        "note": note,
    }


def write_card(output_dir: Path, card: dict) -> Path:
    """Write dataset_card.json into the dataset directory."""
    path = Path(output_dir) / "dataset_card.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2)
    return path


def _row(card: dict) -> str:
    return (
        f"| {card['created'][:10]} | {card['name']} | "
        f"{card['git_commit'] or '-'}{'*' if card['git_dirty'] else ''} | "
        f"{card['seed'] if card['seed'] is not None else '-'} | "
        f"{card['counts'].get('images', '-')} | "
        f"{card['counts'].get('annotations', '-')} | {card.get('note', '')} |\n"
    )


def update_registry(registry_path: Path, card: dict) -> None:
    """Add or replace this dataset's row in the markdown registry."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if stripped.startswith("| Data |") or set(stripped) <= set("|- "):
                continue
            rows.append(line)

    key = f"| {card['name']} |"
    rows = [r for r in rows if key not in r]
    rows.append(_row(card))
    rows.sort()

    path.write_text(
        REGISTRY_HEADER + REGISTRY_INTRO + _TABLE_HEAD + "".join(rows),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Uruchom testy — mają przejść**

Run: `python -m pytest tests/test_dataset_card.py -v`
Expected: 7 passed

- [ ] **Step 5: Podłącz kartę do generatora**

W `src/generate_yolo_dataset.py` dodaj import:

```python
from dataset_card import build_card, write_card, update_registry
```

oraz `import sys` jeśli go nie ma (potrzebny do odtworzenia komendy).

W `parse_args()` dopisz przed `return parser.parse_args()`:

```python
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name recorded in the dataset card and registry "
             "(default: output directory name).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-text note stored in the dataset card and registry.",
    )
```

W `main()`, obok istniejących liczników (`total_count`, `skipped_blank`), dodaj:

```python
    observed_profiles: dict[str, int] = {}
    observed_bases: dict[str, int] = {}
    observed_multiline = 0
    observed_stamps = 0
    observed_stamps_in_gt = 0
    observed_strokes = 0
    observed_highlights = 0
    fonts_used: set[str] = set()
```

Wewnątrz pętli po wariantach, zaraz po otrzymaniu `result`, dopisz:

```python
            fonts_used.add(result["font"])
            observed_bases["partial" if use_partial else "blank"] = (
                observed_bases.get("partial" if use_partial else "blank", 0) + 1
            )
            observed_multiline += result["multiline_fields"]
            if result["scan_augmentation"] is not None:
                profile = result["scan_augmentation"]["profile"]
                observed_profiles[profile] = observed_profiles.get(profile, 0) + 1
            stamp_meta = result["artifacts"]["stamp"]
            if stamp_meta is not None:
                observed_stamps += 1
                observed_stamps_in_gt += int(stamp_meta["in_gt"])
            if result["artifacts"]["strokes"] is not None:
                observed_strokes += 1
            if result["artifacts"]["highlights"] is not None:
                observed_highlights += 1
```

Po zapisie `annotations.json` (na końcu `main()`, przed końcowymi `print`ami) dopisz:

```python
    source_counts: dict[str, int] = {}
    for ann in coco["annotations"]:
        source_counts[ann["source"]] = source_counts.get(ann["source"], 0) + 1

    repo_root = Path(__file__).resolve().parent.parent
    card = build_card(
        name=args.dataset_name or output_dir.name,
        command="python " + " ".join([str(Path(sys.argv[0]).as_posix())] + sys.argv[1:]),
        seed=args.seed,
        repo_root=repo_root,
        counts={
            "images": total_count,
            "annotations": len(coco["annotations"]),
            "templates": len(pages),
        },
        sources=source_counts,
        fonts=sorted(fonts_used),
        observed={
            "scan_profiles": observed_profiles,
            "bases": observed_bases,
            "multiline_fields": observed_multiline,
            "stamps": observed_stamps,
            "stamps_in_ground_truth": observed_stamps_in_gt,
            "strokes": observed_strokes,
            "highlights": observed_highlights,
        },
        note=args.note,
    )
    card_path = write_card(output_dir, card)
    update_registry(repo_root / "docs" / "datasets.md", card)
    print(f"  dataset_card.json --> {card_path}")
    print(f"  docs/datasets.md  --> wpis '{card['name']}' zaktualizowany")
```

- [ ] **Step 6: Sprawdź na małym przebiegu**

```bash
python src/generate_yolo_dataset.py --templates-dir templates \
    --output-dir output/card_check --variants-per-form 1 --seed 3 \
    --note "sprawdzenie karty"
python -c "
import json
c = json.load(open('output/card_check/dataset_card.json', encoding='utf-8'))
print(c['name'], c['git_commit'], 'dirty=', c['git_dirty'], 'seed=', c['seed'])
print('counts:', c['counts'])
print('sources:', c['sources'])
print('observed:', c['observed'])
print('fonts:', len(c['fonts']))
"
cat docs/datasets.md
```

Expected: karta z wypełnionymi wszystkimi sekcjami, tabelka z jednym wierszem `card_check`.

- [ ] **Step 7: Uruchom cały zestaw testów**

Run: `python -m pytest tests/ -q`
Expected: 70 passed

- [ ] **Step 8: Commit**

```bash
git add src/dataset_card.py src/generate_yolo_dataset.py tests/test_dataset_card.py docs/datasets.md
git commit -m "Add dataset provenance: card per dataset plus registry in repo"
```

---

### Task 8: Karta dla istniejącego zbioru 12k i weryfikacja końcowa

**Files:**
- Create (poza gitem): `output/12k_records/dataset_card.json`
- Modify: `docs/datasets.md` (wiersz zbioru 12k)

**Interfaces:**
- Consumes: `dataset_card.build_card/write_card/update_registry` (Task 7)
- Produces: kompletny rejestr obejmujący zbiór wygenerowany przed tą zmianą

- [ ] **Step 1: Zbuduj kartę retrospektywnie ze statystyk zbioru**

Zbiór `output/12k_records` powstał z seedem 2026 na commicie `2b208f2`, przed dodaniem artefaktów i własnych czcionek. Karta odtwarza znane fakty, a statystyki liczy z danych:

```bash
cd /c/Users/tomek/Desktop/text-gen
python -c "
import json, glob, sys
from pathlib import Path
sys.path.insert(0, 'src')
from dataset_card import build_card, write_card, update_registry

root = Path('output/12k_records')
coco = json.load(open(root / 'annotations.json', encoding='utf-8'))
sources, profiles, bases, fonts = {}, {}, {}, set()
multiline = 0
for a in coco['annotations']:
    sources[a['source']] = sources.get(a['source'], 0) + 1
for m in glob.glob(str(root / 'metadata' / '*.json')):
    d = json.load(open(m, encoding='utf-8'))
    p = d['scan_augmentation']['profile']
    profiles[p] = profiles.get(p, 0) + 1
    bases[d['base']] = bases.get(d['base'], 0) + 1
    multiline += d['multiline_fields']
    fonts.add(d['font'])

card = build_card(
    name='12k_records',
    command=('python src/generate_yolo_dataset.py --templates-dir templates '
             '--output-dir output/dataset_v2 --variants-per-form 300 --seed 2026'),
    seed=2026,
    repo_root=Path('.'),
    counts={'images': len(coco['images']), 'annotations': len(coco['annotations']),
            'templates': 40},
    sources=sources,
    fonts=sorted(fonts),
    observed={'scan_profiles': profiles, 'bases': bases,
              'multiline_fields': multiline, 'stamps': 0,
              'stamps_in_ground_truth': 0, 'strokes': 0, 'highlights': 0},
    note='przed artefaktami i wlasnymi czcionkami; folder zmieniony z dataset_v2',
)
card['git_commit'] = '2b208f2'
card['git_dirty'] = False
card['augmentations'] = {k: v for k, v in card['augmentations'].items()
                         if not k.startswith(('stamp_', 'stroke_', 'highlight_'))}
print(write_card(root, card))
update_registry(Path('docs/datasets.md'), card)
"
cat docs/datasets.md
```

Expected: karta w `output/12k_records/dataset_card.json` oraz dwa wiersze w rejestrze (`12k_records` i `card_check`).

Uwaga: sekcja `augmentations` tej karty świadomie nie zawiera kluczy artefaktów — tamten zbiór powstał, zanim istniały, więc wpisanie ich sugerowałoby nieprawdę.

- [ ] **Step 2: Posprzątaj kartę testową z rejestru**

`card_check` był tylko sprawdzeniem, nie jest zbiorem do zapamiętania:

```bash
rm -rf output/card_check
python -c "
from pathlib import Path
p = Path('docs/datasets.md')
lines = [l for l in p.read_text(encoding='utf-8').splitlines(keepends=True)
         if '| card_check |' not in l]
p.write_text(''.join(lines), encoding='utf-8')
"
cat docs/datasets.md
```

Expected: rejestr z jednym wierszem — `12k_records`.

- [ ] **Step 3: Zweryfikuj kryteria sukcesu ze specu**

```bash
python -m pytest tests/ -q
```

Expected: 70 passed; w podsumowaniu ostrzeżeń wyłącznie znane openpyxl.

```bash
python -c "
import json, glob, collections
metas = [json.load(open(m, encoding='utf-8'))
         for m in glob.glob('output/sample_artifacts/metadata/*.json')]
art = collections.Counter()
for d in metas:
    a = d['artifacts']
    art['stamp'] += a['stamp'] is not None
    art['stamp_in_gt'] += bool(a['stamp'] and a['stamp']['in_gt'])
    art['highlights'] += a['highlights'] is not None
    art['strokes'] += a['strokes'] is not None
print(len(metas), 'obrazow;', dict(art))
coco = json.load(open('output/sample_artifacts/annotations.json', encoding='utf-8'))
print('sources:', collections.Counter(a['source'] for a in coco['annotations']))
"
```

Expected: `sources` zawiera `stamp`; liczby artefaktów zgodne z prawdopodobieństwami.

- [ ] **Step 4: Commit**

```bash
git add docs/datasets.md
git commit -m "Register the 12k dataset generated before the artifact work"
```

---

## Self-Review

**Pokrycie specu:**

| Sekcja specu | Zadanie |
|---|---|
| 1. Sanityzacja treści | Task 1 |
| 2. Kolejność w pipeline | Task 5 (blok artefaktów przed skanem) |
| 3. Kreski długopisem | Task 2 |
| 4. Pieczątki | Task 4 (+ czcionka drukowana w Step 1) |
| 5. Marker | Task 3 |
| 6. Historia zbiorów | Task 7 (karta + rejestr + CLI), Task 8 (zbiór istniejący) |
| 7. Próbka do przeglądu | Task 6 (bramka) |
| Kryterium 1 (brak .notdef) | Task 1, `test_every_shipped_font_renders_all_sanitized_vocabulary` |
| Kryterium 2 (transkrypcja = render) | Task 1, sanityzacja przed pomiarem |
| Kryterium 3 (reguła GT pieczątek) | Task 4, `test_ground_truth_rule_holds_over_many_draws` |
| Kryterium 4 (bboxy nietknięte) | Task 5, `test_artifacts_never_change_text_bboxes` |
| Kryterium 5 (czytelność pod markerem) | Task 3, `test_text_stays_darker_than_the_band` |
| Kryterium 6 (karta i rejestr) | Task 7, `tests/test_dataset_card.py` |
| Kryterium 7 (testy bez ostrzeżeń) | Task 8, Step 3 |

**Spójność typów:** `draw_stamp` zwraca `(Optional[dict], list[dict])` — tak konsumuje Task 5. `draw_highlights`/`draw_stray_strokes` zwracają `dict` i mutują obraz w miejscu — tak samo w Task 5. `records` mają wszędzie klucze `label/source/text/bbox`. `generate_field_content` dostaje piąty parametr `font_path` w Task 1 i tylko tam jest wołane. `augmentation_settings()` czyta stałe zdefiniowane w Task 2–4 pod tymi samymi nazwami.

**Liczby testów:** start 29 → Task 1: +10 (39) → Task 2: +5 (44) → Task 3: +6 (50) → Task 4: +8 (58) → Task 5: +5 (63) → Task 7: +7 (70). Jeśli faktyczna suma różni się o kilka, liczy się to, że **nic nie failuje** — nie dopasowuj testów do liczby.
