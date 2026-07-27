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
