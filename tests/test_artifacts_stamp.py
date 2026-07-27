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


def test_stamp_records_keep_line_order(vocab):
    """Record i's box must sit above record i+1's.

    The stamp's lines are drawn top to bottom and paired with their boxes by
    position, so a mismatch between a transcription and its box (the defect
    that would silently poison OCR labels) shows up as records whose boxes no
    longer descend the page in order.
    """
    checked = 0
    for seed in range(60):
        random.seed(seed)
        meta, records = draw_stamp(_blank(), vocab, [])
        if len(records) < 2:
            continue
        checked += 1
        tops = [rec["bbox"][1] for rec in records]
        assert tops == sorted(tops), f"stamp lines out of order: {tops}"
    assert checked > 0, "no multi-line labelled stamp in 60 draws"


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
