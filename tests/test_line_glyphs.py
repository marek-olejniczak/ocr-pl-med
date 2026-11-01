"""Generator v3 glyph families: joins, letter variants, moved diacritics."""

import random
from pathlib import Path

import numpy as np
import pytest

import generate_ocr_lines as gol
import line_glyphs as LG
import ocr_content
from fill_form import EXCLUDED_FONTS
from renderer import find_fonts
from vocabulary import Vocabulary

REPO = Path(__file__).resolve().parent.parent
RES = str(REPO / "resources")


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(RES)


@pytest.fixture(scope="module")
def pools():
    return ocr_content.LinePools(RES)


@pytest.fixture(scope="module")
def fonts():
    return [f for f in find_fonts(str(REPO / "resources" / "fonts"))
            if Path(f).name not in EXCLUDED_FONTS]


def font(fonts, name):
    return next(f for f in fonts if name in Path(f).name)


def test_glyph_families_are_switchable():
    assert set(LG.GLYPH_FAMILIES) <= set(gol.ALL_FAMILIES)
    assert not set(LG.GLYPH_FAMILIES) & set(gol.V2_FAMILIES)
    assert gol.resolve_families(",".join(LG.GLYPH_FAMILIES)) == set(gol.V2_FAMILIES)


def lines(vocab, pools, fonts, enabled, n=40, seed=11):
    """Render like generate_ocr_lines --line-seeds."""
    config = gol.build_line_config()
    out = []
    attempt = 0
    while len(out) < n:
        s = gol.seed_line(seed, len(out), attempt)
        img, text, meta = gol.render_line(vocab, fonts, config, True, pools=pools,
                                          enabled=enabled, line_seed=s)
        if img is None:
            attempt += 1
            continue
        attempt = 0
        out.append((np.asarray(img), text, meta))
    return out


def test_glyph_families_leave_the_global_stream_alone(vocab, pools, fonts):
    """Turning a glyph family on changes the ink, never the text or the other effects."""
    off = lines(vocab, pools, fonts, set(gol.V2_FAMILIES))
    on = lines(vocab, pools, fonts, set(gol.ALL_FAMILIES))
    # Values that do not depend on the size of the ink.
    keys = ("font", "font_size", "kind", "content", "ink", "target_width", "style",
            "pen_fade", "rule", "neighbours", "morphology", "elastic", "scan_profile",
            "morphology_params")
    for (_, t0, m0), (_, t1, m1) in zip(off, on):
        assert t0 == t1
        assert {k: m0[k] for k in keys} == {k: m1[k] for k in keys}
    assert any(m["joins_params"] or m["variant_params"] or m["diacritic_params"]
               for _, _, m in on)
    assert all(m["joins_params"] is None and m["variant_params"] is None
               and m["diacritic_params"] is None for _, _, m in off)


def test_same_seed_same_pixels_with_glyph_families(vocab, pools, fonts):
    a = lines(vocab, pools, fonts, set(gol.ALL_FAMILIES), n=15)
    b = lines(vocab, pools, fonts, set(gol.ALL_FAMILIES), n=15)
    for (i0, t0, _), (i1, t1, _) in zip(a, b):
        assert t0 == t1 and np.array_equal(i0, i1)


def test_joins_pull_letters_together():
    rng = random.Random(0)
    LGJ = LG.JOIN_PROB
    try:
        LG.JOIN_PROB = 1.0
        record = {}
        tracking, stretch, kern = LG.plan_joins("kości przyczep", 40, rng, record)
    finally:
        LG.JOIN_PROB = LGJ
    assert LG.JOIN_TRACKING_RANGE[0] <= tracking <= LG.JOIN_TRACKING_RANGE[1]
    assert LG.JOIN_STRETCH_RANGE[0] <= stretch <= LG.JOIN_STRETCH_RANGE[1]
    assert all(kern(i) <= 0 for i in range(14))
    assert set(record["pulled_pairs"]) <= LG.JOIN_PAIRS


def test_variant_donors_pick_lookalikes(fonts):
    donors = LG.variant_donors(tuple(sorted(fonts)))
    assert set(donors) == set(LG.VARIANT_PAIRS)
    for letter, ranked in donors.items():
        assert 0 < len(ranked) <= LG.VARIANT_TOP_FONTS
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)
    # The looped cursive k is the one read as h.
    assert any("Script" in Path(f).name or "Playwrite" in Path(f).name
               for f, _ in donors["k"][:4])


def test_pen_width_separates_bold_from_thin(fonts):
    assert LG.stroke_width(font(fonts, "CaveatBrush")) > 1.3 * LG.stroke_width(font(fonts, "Marek_3"))


@pytest.mark.parametrize("char", ["ś", "ż", "ó", "ą", "ę", "i"])
def test_split_mark_finds_the_mark(fonts, char):
    path = font(fonts, "Kalam")
    body, mark, box = LG._split_mark(char, path, 48)
    assert mark.any() and body.any()
    assert (mark > 32).sum() < (body > 32).sum()


def test_dropped_mark_leaves_the_base_letter(fonts):
    path = font(fonts, "Kalam")
    body, mark, _ = LG._split_mark("ś", path, 48)
    dropped = LG._move_mark(body, mark, {"op": "drop"})
    base = LG._raster("s", LG._font(path, 48), 48)
    assert abs(int((dropped > 96).sum()) - int((base > 96).sum())) < 0.15 * (base > 96).sum()


def test_hook_keeps_letter_position_when_mark_moves(fonts):
    path = font(fonts, "Kalam")
    hook = LG.make_glyph_hook(path, 48, {}, {0: {"op": "shift", "dx": 0.5, "dy": -0.3,
                                                 "rot": 0.0, "scale": 1.0}})
    img, advance = hook(0, "ś", LG._font(path, 48))
    plain = LG._to_rgba(LG._raster("ś", LG._font(path, 48), 48))
    assert advance == int(LG._font(path, 48).getlength("ś"))
    assert img.width >= plain.width and img.height >= plain.height
    assert hook(1, "s", LG._font(path, 48)) is None
