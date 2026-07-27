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
