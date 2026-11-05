"""P0: metrics.py odtwarza CER/WER baseline z eval_cer.py (metrics.json, dataset-v1)."""

from __future__ import annotations

import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest

from evaluation.metrics import BASELINE_DIR, load_predictions, normalize_ws, score

REPO = Path(__file__).resolve().parents[4]
EVAL_CER = REPO / "training" / "ocr" / "surya" / "hp_search" / "eval_cer.py"


@pytest.fixture(scope="module")
def rows():
    return load_predictions()


def test_baseline_matches_metrics_json(rows):
    ref = json.loads((BASELINE_DIR / "metrics.json").read_text())["metrics"]
    s = score([r["ground_truth"] for r in rows], [r["prediction"] for r in rows])
    assert s.n == 2160
    assert s.cer == pytest.approx(ref["cer"], abs=1e-6)
    assert s.wer == pytest.approx(ref["wer"], abs=1e-6)


def test_nfc_is_noop_on_baseline(rows):
    for r in rows:
        for k in ("ground_truth", "prediction"):
            assert unicodedata.normalize("NFC", r[k]) == r[k], r["file_name"]


@pytest.mark.skipif(not EVAL_CER.exists(), reason="eval_cer.py not in checkout")
def test_identical_to_eval_cer_functions(rows):
    spec = importlib.util.spec_from_file_location("eval_cer", EVAL_CER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # importuje tylko stdlib na poziomie modułu
    gts = [r["ground_truth"] for r in rows]
    prs = [r["prediction"] for r in rows]
    s = score(gts, prs)
    assert s.cer == mod.character_error_rate(gts, prs)
    assert s.wer == mod.word_error_rate(gts, prs)


def test_normalize_ws():
    assert normalize_ws("  a \t b\n") == "a b"
