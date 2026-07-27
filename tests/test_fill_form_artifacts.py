import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fill_form import fill_single_form, EXCLUDED_FONTS
from renderer import find_fonts
from transforms import AugmentConfig
from vocabulary import Vocabulary

REPO = Path(__file__).resolve().parent.parent


def _seed(value: int) -> None:
    """Seed both RNGs the pipeline draws from.

    `random` decides what happens; numpy decides how it looks (pen-fade
    speckle, stamp ink unevenness). Seeding only `random` leaves the pixels —
    and with them the tight ink bboxes derived from the alpha mask —
    dependent on whatever consumed numpy earlier in the session, which would
    make the artifacts-on vs artifacts-off comparisons below order-dependent.
    """
    random.seed(value)
    np.random.seed(value)


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
    _seed(1)
    result = _run(tmp_path, vocab, font_path, enable_artifacts=False)
    assert result["artifacts"] == {"stamp": None, "highlights": None, "strokes": None}
    assert all(r["source"] != "stamp" for r in result["records"])


def test_artifact_meta_key_always_present(tmp_path, vocab, font_path):
    _seed(2)
    result = _run(tmp_path, vocab, font_path)
    assert set(result["artifacts"]) == {"stamp", "highlights", "strokes"}


def test_stamp_records_appear_and_are_well_formed(tmp_path, vocab, font_path):
    """Over many seeds at least one stamp must land in the ground truth."""
    found = False
    for seed in range(40):
        _seed(seed)
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
    _seed(5)
    with_art = _run(tmp_path, vocab, font_path)
    _seed(5)
    without = _run(tmp_path, vocab, font_path, enable_artifacts=False)

    text_only = [r for r in with_art["records"] if r["source"] != "stamp"]
    assert len(text_only) == len(without["records"])
    for a, b in zip(text_only, without["records"]):
        assert a["bbox"] == b["bbox"]
        assert a["text"] == b["text"]
        assert a["source"] == b["source"]


def test_image_size_unchanged_by_artifacts(tmp_path, vocab, font_path):
    _seed(6)
    result = _run(tmp_path, vocab, font_path)
    assert result["image"].size == (800, 1200)


# --- non-vacuous variants of the above ---------------------------------------
# STAMP/HIGHLIGHT/STROKE_PROB are 0.30/0.12/0.20, so most seeds fire nothing at
# all: a test that hard-codes a seed number almost certainly exercises the
# no-artifact path and would still pass against an integration that never calls
# the artifact functions. The seeds below are therefore discovered at runtime.

@pytest.fixture(scope="module")
def drawing_seeds(tmp_path_factory, vocab, font_path):
    """Seeds where a stamp or a stray stroke actually puts ink on the page.

    Highlights are excluded on purpose: with no eligible records they report
    count 0 and leave the pixels untouched, which would make the
    "artifacts changed the image" assertions below ambiguous.
    """
    tmp_path = tmp_path_factory.mktemp("drawing")
    seeds = []
    for seed in range(60):
        _seed(seed)
        art = _run(tmp_path, vocab, font_path)["artifacts"]
        if art["stamp"] is not None or art["strokes"] is not None:
            seeds.append(seed)
        if len(seeds) == 3:
            break
    assert seeds, "no artifact fired in 60 seeds — are the probabilities wired up?"
    return seeds


def test_disabling_artifacts_actually_suppresses_drawing(
    tmp_path, vocab, font_path, drawing_seeds
):
    """The flag must gate the drawing, not just the reported metadata."""
    seed = drawing_seeds[0]

    _seed(seed)
    with_art = _run(tmp_path, vocab, font_path)
    _seed(seed)
    without = _run(tmp_path, vocab, font_path, enable_artifacts=False)

    # same seed, same text — the only possible difference is the artifact ink
    assert with_art["image"].tobytes() != without["image"].tobytes()
    assert without["artifacts"] == {"stamp": None, "highlights": None, "strokes": None}


def test_text_records_survive_seeds_where_artifacts_really_fire(
    tmp_path, vocab, font_path, drawing_seeds
):
    """The photometric-only guarantee, checked on seeds that draw something."""
    for seed in drawing_seeds:
        _seed(seed)
        with_art = _run(tmp_path, vocab, font_path)
        _seed(seed)
        without = _run(tmp_path, vocab, font_path, enable_artifacts=False)

        art = with_art["artifacts"]
        assert any(art[k] is not None for k in art), f"seed {seed} drew nothing"
        assert with_art["image"].size == without["image"].size

        text_only = [r for r in with_art["records"] if r["source"] != "stamp"]
        assert len(text_only) == len(without["records"])
        for a, b in zip(text_only, without["records"]):
            assert a["bbox"] == b["bbox"]
            assert a["text"] == b["text"]
            assert a["source"] == b["source"]


def test_stamp_metadata_is_reported_when_a_stamp_is_pressed(
    tmp_path, vocab, font_path
):
    """A drawn stamp must show up in the metadata, not only on the pixels."""
    for seed in range(40):
        _seed(seed)
        result = _run(tmp_path, vocab, font_path)
        meta = result["artifacts"]["stamp"]
        if meta is None:
            continue
        assert meta["shape"] in ("rect", "round")
        assert abs(meta["angle_deg"]) <= 20.0
        assert isinstance(meta["in_gt"], bool)
        assert meta["lines"] >= 2
        # in_gt is the contract for whether records were contributed
        has_records = any(r["source"] == "stamp" for r in result["records"])
        assert has_records is meta["in_gt"]
        return
    pytest.fail("no stamp was pressed in 40 seeds")


def _first_meta(tmp_path, vocab, font_path, key, limit=60):
    """Metadata of the first seed in [0, limit) where `key` fires."""
    for seed in range(limit):
        _seed(seed)
        meta = _run(tmp_path, vocab, font_path)["artifacts"][key]
        if meta is not None:
            return meta
    pytest.fail(f"{key!r} never fired in {limit} seeds — is it wired into the pipeline?")


def test_highlighter_is_wired_in(tmp_path, vocab, font_path):
    meta = _first_meta(tmp_path, vocab, font_path, "highlights")
    # a printed "p" field is always eligible, so a real pass must mark something
    assert meta["count"] >= 1
    assert len(meta["colors"]) == meta["count"]
    assert len(meta["targets"]) == meta["count"]


def test_stray_strokes_are_wired_in(tmp_path, vocab, font_path):
    meta = _first_meta(tmp_path, vocab, font_path, "strokes")
    assert 1 <= meta["count"] <= 3
    assert len(meta["kinds"]) == meta["count"]
    assert all(k in ("sweep", "tick", "scribble") for k in meta["kinds"])
