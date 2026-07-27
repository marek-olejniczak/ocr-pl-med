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
