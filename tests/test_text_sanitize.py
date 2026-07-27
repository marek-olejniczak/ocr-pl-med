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


@pytest.fixture(scope="module")
def vocab():
    """Loading the vocabulary workbook is slow — do it once per module."""
    from vocabulary import Vocabulary
    return Vocabulary(str(REPO / "resources"))


def test_dagger_removed_and_brackets_become_parens():
    assert normalize_for_handwriting("Zapalenie † [ostre]") == "Zapalenie (ostre)"


def test_micro_sign_becomes_u():
    assert normalize_for_handwriting("100µg") == "100ug"
    assert normalize_for_handwriting("100μg") == "100ug"


def test_all_dashes_unified():
    assert normalize_for_handwriting("A–B‑C—D") == "A-B-C-D"


def test_quotes_and_nbsp():
    assert normalize_for_handwriting("„ostry” stan") == '"ostry" stan'


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


def test_raw_vocabulary_still_contains_characters_worth_sanitizing(vocab):
    """Guard against vacuity: if the vocabulary ever stops producing these,
    the acceptance test below would pass for the wrong reason."""
    import field_content as fc

    random.seed(11)
    raw_chars: set[str] = set()
    for sampler in (fc._sample_text_unit, fc._sample_number_unit,
                    fc._sample_mix_unit):
        for _ in range(500):
            raw_chars |= set(sampler(vocab))

    offenders = raw_chars & set("[]†µ")
    assert offenders, (
        "raw vocabulary no longer emits any of [ ] † µ — the acceptance test "
        "would now pass vacuously; add new offenders or drop the sanitizer"
    )


def test_every_shipped_font_renders_all_sanitized_vocabulary(vocab):
    """Success criterion 1: no .notdef boxes anywhere in generated data.

    Goes through the real generation path with font_path set, so the
    sanitization wiring inside generate_field_content is under test.

    400 fields per kind per font is not arbitrary: offenders appear in ~0.4%
    of raw units and 9 of the 29 fonts lack them, so a smaller sweep can miss
    them entirely and pass even with sanitization removed.
    """
    from field_content import generate_field_content

    fonts = sorted(FONT_DIR.rglob("*.ttf"))
    assert len(fonts) >= 15, "expected the full font pool"

    random.seed(11)
    for font in fonts:
        codes = font_charset(str(font))
        # An unreadable cmap is exactly the case that yields .notdef boxes in
        # production — fail loudly instead of skipping.
        assert codes, f"{font.name}: unreadable"
        for kind in ("t", "n", "mix"):
            for _ in range(400):
                text = generate_field_content(
                    kind, vocab, lambda s: len(s) * 8.0, 900,
                    font_path=str(font),
                )
                missing = {ch for ch in text
                           if ch != " " and ord(ch) not in codes}
                assert not missing, f"{font.name}: {missing!r} in {text!r}"


def test_fill_single_form_passes_its_font_to_every_content_call(vocab):
    """The fill_form call sites must forward font_path.

    Output inspection cannot catch a missing font_path today (stage 1 already
    removes every offender the current font pool lacks), so assert on the
    call itself — otherwise the wiring could be reverted with a green suite.
    """
    import fill_form
    from PIL import Image
    from transforms import AugmentConfig

    font = str(FONT_DIR / "Marek_1" / "Marek_1-Regular.ttf")
    seen: list[object] = []
    real = fill_form.generate_field_content

    def spy(*args, **kwargs):
        seen.append(kwargs.get("font_path", "MISSING"))
        return real(*args, **kwargs)

    form_path = REPO / "tests" / "_tmp_wiring_form.png"
    Image.new("RGB", (800, 700), (252, 250, 247)).save(form_path)
    config = AugmentConfig()
    config.paper.enabled = False
    config.scan.enabled = False
    fields = [
        {"label": "t", "x_min": 40, "y_min": 80, "x_max": 700, "y_max": 125},
        {"label": "mix", "x_min": 40, "y_min": 240, "x_max": 400, "y_max": 280},
        {"label": "t", "x_min": 40, "y_min": 420, "x_max": 760, "y_max": 620},
    ]

    fill_form.generate_field_content = spy
    try:
        random.seed(2)
        fill_form.fill_single_form(
            form_path=form_path, fields=fields, vocab=vocab,
            font_path=font, config=config, pipeline=None, apply_scan=False,
            empty_field_range=(0.0, 0.0),
        )
    finally:
        fill_form.generate_field_content = real
        form_path.unlink(missing_ok=True)

    assert seen, "fill_single_form never generated any field content"
    assert all(fp == font for fp in seen), f"call sites lost font_path: {seen}"
