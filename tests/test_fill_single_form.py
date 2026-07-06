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
