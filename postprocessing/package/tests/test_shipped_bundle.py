"""Każda dołączona paczka musi się wczytać (bez słowników: sam model i progi)."""
import gzip
import json

import pytest

from ocr_postprocess import bundle
from ocr_postprocess.decide import DecisionParams
from ocr_postprocess.error_model import ErrorModel

BUNDLES = sorted(p.parent for p in bundle.MODELS_DIR.glob("*/bundle.json"))


def test_default_bundle_is_shipped():
    assert bundle.resolve().name == bundle.DEFAULT_BUNDLE
    assert bundle.resolve(bundle.DEFAULT_BUNDLE) == bundle.resolve()
    with pytest.raises(FileNotFoundError):
        bundle.resolve("nie-ma-takiej")


@pytest.mark.parametrize("path", BUNDLES, ids=lambda p: p.name)
def test_bundle_loads(path):
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    assert meta["format"] == bundle.FORMAT
    DecisionParams(**meta["params"])
    assert DecisionParams(**meta["params_rescored"]).w > 0
    m = ErrorModel.from_dict(json.loads(gzip.decompress((path / "error_model.json.gz").read_bytes())))
    assert m.logp("kosc", "kość") > m.logp("kosc", "kosm")


def test_rescored_requires_one_image_per_line():
    from ocr_postprocess.correct import Corrector
    c = Corrector.__new__(Corrector)
    c.p_rescored = DecisionParams(w=1.0)
    with pytest.raises(ValueError, match="obraz"):
        c.correct_lines_rescored(["a", "b"], [None], scorer=None)
