"""Paczka korektora: model pomyłek + progi + skróty z treningu, bez tekstu linii.

Paczka to katalog z `bundle.json` i `error_model.json.gz`. Dołączona do pakietu leży w `models/<nazwa>/`.
Nową paczkę buduje `experiments/` (`python -m evaluation.build_bundle` albo cały przebieg `rebuild.py`).

    from ocr_postprocess import load
    corr = load()                       # domyślna paczka; ok. 1.5 s po `ocr-postprocess setup`
    corr = load("v4_200k_25k")          # po nazwie
    corr = load("/sciezka/do/paczki")   # albo po ścieżce
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ocr_postprocess.correct import Corrector
from ocr_postprocess.decide import DecisionParams
from ocr_postprocess.error_model import ErrorModel
from ocr_postprocess.lexicon import Lexicon, domain_terms
from ocr_postprocess.lm import UnigramLM

FORMAT = 2
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_BUNDLE = "v4_200k_25k"  # adapter Surya LoRA, dla którego zbudowano paczkę


def resolve(bundle: str | Path | None = None) -> Path:
    """Katalog paczki: ścieżka do katalogu albo nazwa paczki z `models/`."""
    b = Path(bundle or DEFAULT_BUNDLE)
    if (b / "bundle.json").exists():
        return b
    if (MODELS_DIR / b / "bundle.json").exists():
        return MODELS_DIR / b
    have = sorted(p.parent.name for p in MODELS_DIR.glob("*/bundle.json"))
    raise FileNotFoundError(f"brak paczki {str(bundle)!r}; dołączone: {have}")


def save(out: Path, model: ErrorModel, meta: dict) -> dict:
    """Zapisuje paczkę; plik modelu jest deterministyczny (stały mtime w gzipie)."""
    meta = {"format": FORMAT, **meta}
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(model.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    (out / "error_model.json.gz").write_bytes(gzip.compress(data, mtime=0))
    (out / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def load(bundle: str | Path | None = None, lexicon: Lexicon | None = None, lm: UnigramLM | None = None,
         compiled: bool = True) -> Corrector:
    """`compiled=True`: słownik i częstości ze skompilowanego indeksu (`compiled.py`, ok. 1.5 s; jeśli go brakuje,
    zbuduje go raz, ok. 40 s); `False`: budowa od zera z plików zasobów (ok. 23 s), wynik identyczny."""
    path = resolve(bundle)
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: format {meta.get('format')}, oczekiwano {FORMAT}")
    model = ErrorModel.from_dict(json.loads(gzip.decompress((path / "error_model.json.gz").read_bytes())))
    if lexicon is None and compiled:
        from ocr_postprocess import compiled as C
        lexicon, lm = C.load(meta["domain_langs"])
    lex = lexicon or Lexicon(extra_known=domain_terms(meta["domain_langs"]))
    lm = lm or UnigramLM(lex.targets())
    corr = Corrector(lex, lm, model, DecisionParams(**meta["params"]), lex.abbrev | set(meta["train_abbrev"]))
    if meta.get("params_rescored"):
        corr.p_rescored = DecisionParams(**meta["params_rescored"])
    return corr
