"""Phase-1 OCR line renderer: content pools, image effects, family switches."""

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import generate_ocr_lines as gol
import line_effects
import ocr_content
from fill_form import EXCLUDED_FONTS
from renderer import find_fonts
from text_sanitize import font_charset
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


def test_resolve_families():
    assert gol.resolve_families("") == set(gol.ALL_FAMILIES)
    assert gol.resolve_families("all") == set()
    assert gol.resolve_families("elastic, caps") == set(gol.ALL_FAMILIES) - {"elastic", "caps"}
    with pytest.raises(SystemExit):
        gol.resolve_families("nonsense")


def test_pools_load_anatomy_and_words(pools):
    assert len(pools.phrases["anatomy"]) > 300
    assert all(3 <= len(w) <= 12 for w in pools.words)
    assert "udowa" in pools.words or "udowy" in pools.words


def test_disabled_kinds_never_drawn():
    random.seed(0)
    enabled = set(gol.ALL_FAMILIES) - {"short_words", "arrows_bullets", "caps"}
    kinds = {ocr_content.pick_line_kind(enabled) for _ in range(500)}
    assert not kinds & {"word", "heading", "bullet", "arrow"}
    assert "phrase" in kinds


def test_content_matches_kind(pools, vocab, fonts):
    random.seed(1)
    # Arrows need a ">" glyph, which the custom hands lack.
    font = next(f for f in fonts if ord(">") in font_charset(f))
    measure = gol.make_measure(font, 30, gol.WordStyle.random(gol.build_line_config().char))
    seen = set()
    for _ in range(300):
        text, kind = ocr_content.generate_line_content(
            pools, vocab, measure, 600, font, set(gol.ALL_FAMILIES))
        if not text:
            continue
        seen.add(kind)
        if kind == "heading":
            assert text == text.upper()
        if kind == "bullet":
            assert text[0] in "-•*123abc"
        if kind == "arrow":
            assert any(a in text for a in ocr_content.ARROWS)
        if kind == "word":
            assert len(text.split()) <= 2
    assert {"word", "phrase", "heading", "bullet", "arrow"} <= seen


def test_anatomy_off_falls_back_to_form_content(pools, vocab, fonts):
    random.seed(2)
    font = fonts[0]
    measure = gol.make_measure(font, 30, gol.WordStyle.random(gol.build_line_config().char))
    kinds = {ocr_content.generate_line_content(
        pools, vocab, measure, 600, font, set(gol.ALL_FAMILIES) - {"anatomy_vocab"})[1]
        for _ in range(60)}
    assert kinds <= {"t", "n", "mix"}


def test_digit_share_drops_with_new_content(pools, vocab, fonts):
    random.seed(3)
    font = fonts[0]
    measure = gol.make_measure(font, 30, gol.WordStyle.random(gol.build_line_config().char))

    def share(enabled):
        chars = "".join(ocr_content.generate_line_content(
            pools, vocab, measure, 600, font, enabled)[0] for _ in range(400))
        return sum(c.isdigit() for c in chars) / max(1, len(chars))

    assert share(set(gol.ALL_FAMILIES)) < 0.08
    assert share(set()) > 0.15


def test_neighbour_glyphs_add_ink_at_edges(fonts):
    random.seed(4)
    canvas = Image.new("RGB", (400, 60), (250, 250, 250))
    cfg = gol.build_line_config()
    style = gol.WordStyle.random(cfg.char)
    line_effects.NEIGHBOUR_BOTH_PROB = 1.0
    try:
        sides = line_effects.intrude_neighbour_glyphs(
            canvas, (20, 20, 28), fonts[0], 32, cfg, style, lambda: "kość")
    finally:
        line_effects.NEIGHBOUR_BOTH_PROB = 0.25
    assert set(sides) == {"top", "bottom"}
    arr = np.array(canvas.convert("L"))
    assert (arr[:3] < 200).any(), "no ink at the top edge"
    assert (arr[-3:] < 200).any(), "no ink at the bottom edge"
    assert not (arr[25:35] < 200).any(), "neighbour ink reached the middle of the line"


def test_effects_keep_size_and_text_alignment():
    random.seed(5)
    img = Image.new("RGB", (300, 50), (250, 250, 250))
    img.paste((0, 0, 0), (100, 20, 200, 30))
    for fn in (line_effects.apply_morphology, line_effects.apply_elastic):
        out, name = fn(img.copy())
        assert out.size == img.size, name
        assert (np.array(out.convert("L")) < 128).sum() > 200, name
    out, meta = line_effects.apply_phone_photo(img.copy())
    assert out.size == img.size and meta["profile"] == "phone_photo"


def test_grid_paper_draws_lines():
    random.seed(6)
    img = Image.new("RGB", (200, 60), (252, 250, 247))
    kind = line_effects.draw_grid_paper(img, text_height=30, baseline_y=40)
    assert kind in ("grid", "ruled")
    arr = np.array(img)
    assert (arr != np.array((252, 250, 247))).any(axis=2).sum() > 200


def test_render_line_every_family_setting(vocab, pools, fonts):
    """Baseline, all-on and each single family render without error and
    report the family in metadata when it fired."""
    random.seed(7)
    np.random.seed(7)
    cfg = gol.build_line_config()
    for enabled in [set(), set(gol.ALL_FAMILIES)] + [{f} for f in gol.ALL_FAMILIES]:
        got = 0
        for _ in range(12):
            img, text, meta = gol.render_line(vocab, fonts, cfg, True,
                                              pools=pools, enabled=enabled)
            if img is None:
                continue
            got += 1
            assert text and img.mode == "RGB"
            assert set(meta) >= {"kind", "neighbours", "morphology", "elastic", "scan_profile"}
            if "phone_photo" not in enabled:
                assert meta["scan_profile"] != "phone_photo"
            if "morphology" not in enabled:
                assert meta["morphology"] is None
            if "elastic" not in enabled:
                assert meta["elastic"] is None
        assert got >= 6, enabled


def test_arrow_kind_degrades_to_phrase_without_gt_glyph(pools, vocab, fonts):
    font = next(f for f in fonts if ord(">") not in font_charset(f))
    measure = gol.make_measure(font, 30, gol.WordStyle.random(gol.build_line_config().char))
    ocr_content.LINE_KINDS, saved = {"arrow": 1.0}, ocr_content.LINE_KINDS
    try:
        kinds = {ocr_content.generate_line_content(
            pools, vocab, measure, 600, font, set(gol.ALL_FAMILIES))[1] for _ in range(20)}
    finally:
        ocr_content.LINE_KINDS = saved
    assert kinds == {"phrase"}
