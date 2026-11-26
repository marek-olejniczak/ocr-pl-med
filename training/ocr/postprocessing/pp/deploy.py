"""Paczka korektora do wdrożenia: model kanału + progi + skróty z treningu, bez tekstu linii.

Budowa (raz, po nowym modelu OCR; predykcje z etykietami w formacie data/baseline/predictions.jsonl):

    PYTHONPATH=. python -m pp.deploy build --predictions PLIK.jsonl --groups data/v12/groups.csv \\
        --params results/v12/deploy_params.json --domain en,la,pl --out deploy/v4_200k_25k

Użycie w aplikacji (słownik i częstości z $PP_CACHE, pobierane przez fetch_resources.py):

    from pp.deploy import load
    corr = load("deploy/v4_200k_25k")               # ok. 1.5 s (skompilowany indeks, pp.compiled), potem partią
    corr.correct_lines(["linia z OCR", ...])        # -> poprawione linie
    corr.correct_lines_detailed([...])              # -> [{"text": ..., "changes": [{"token", "from", "to"}]}]

    PYTHONPATH=. python -m pp.deploy correct --bundle deploy/v4_200k_25k < wejscie.txt > wyjscie.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

from pp.correct import Corrector
from pp.decide import DecisionParams
from pp.error_model import ErrorModel, Params, word_pairs
from pp.lexicon import Lexicon, domain_terms
from pp.lm import UnigramLM
from pp.protect import train_abbreviations

FORMAT = 1


def build(predictions: Path, groups_csv: Path | None, params_json: Path, domain: list[str], out: Path, note: str = "",
          rescored_json: Path | None = None) -> dict:
    from evaluation.folds import line_groups, read_group_map
    from evaluation.metrics import load_predictions, normalize_ws

    rows = load_predictions(predictions)
    gts = [normalize_ws(r["ground_truth"]) for r in rows]
    prs = [normalize_ws(r["prediction"]) for r in rows]
    groups = line_groups(rows, "fine", read_group_map(groups_csv) if groups_csv else None)
    params = json.loads(params_json.read_text(encoding="utf-8"))["params"]
    rescored = json.loads(rescored_json.read_text(encoding="utf-8"))["params"] if rescored_json else None
    lex = Lexicon(extra_known=domain_terms(domain))
    model = ErrorModel.fit(word_pairs(gts, prs, groups), Params(multi=True))
    abbrev = sorted(train_abbreviations(gts, lex.known))
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "error_model.pkl.gz", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"format": FORMAT, "built": date.today().isoformat(), "params": params, "params_rescored": rescored,
            "domain_langs": domain,
            "train_abbrev": abbrev, "lines": len(rows), "groups": sorted(set(groups)),
            "predictions": predictions.name, "note": note}
    (out / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def load(bundle: str | Path, lexicon: Lexicon | None = None, lm: UnigramLM | None = None,
         compiled: bool = True) -> Corrector:
    """`compiled=True`: słownik i częstości ze skompilowanego indeksu (`pp.compiled`, ok. 1–2 s; pierwszy raz buduje go
    ok. 30 s); `False`: budowa od zera z plików zasobów (ok. 23 s), wynik identyczny."""
    bundle = Path(bundle)
    meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{bundle}: format {meta.get('format')}, oczekiwano {FORMAT}")
    with gzip.open(bundle / "error_model.pkl.gz", "rb") as f:
        model = pickle.load(f)
    if lexicon is None and compiled:
        from pp import compiled as C
        lexicon, lm = C.load(meta["domain_langs"])
    lex = lexicon or Lexicon(extra_known=domain_terms(meta["domain_langs"]))
    lm = lm or UnigramLM(lex.targets())
    corr = Corrector(lex, lm, model, DecisionParams(**meta["params"]), lex.abbrev | set(meta["train_abbrev"]))
    if meta.get("params_rescored"):
        corr.p_rescored = DecisionParams(**meta["params_rescored"])
    return corr


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--predictions", type=Path, required=True)
    b.add_argument("--groups", type=Path)
    b.add_argument("--params", type=Path, required=True)
    b.add_argument("--params-rescored", type=Path, help="progi z oceną Suryi (eval_surya --deploy), opcjonalnie")
    b.add_argument("--domain", default="en,la,pl")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--note", default="")
    c = sub.add_parser("correct")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--detailed", action="store_true", help="jsonl z listą zmian zamiast samego tekstu")
    a = ap.parse_args()
    if a.cmd == "build":
        meta = build(a.predictions, a.groups, a.params, [x for x in a.domain.split(",") if x], a.out, a.note,
                     a.params_rescored)
        print(json.dumps({k: v for k, v in meta.items() if k != "train_abbrev"}, ensure_ascii=False))
        return
    t0 = time.time()
    corr = load(a.bundle)
    print(f"korektor gotowy po {time.time() - t0:.0f}s", file=sys.stderr)
    lines = [l.rstrip("\n") for l in sys.stdin]
    if a.detailed:
        for r in corr.correct_lines_detailed(lines):
            print(json.dumps(r, ensure_ascii=False))
    else:
        for l in corr.correct_lines(lines):
            print(l)


if __name__ == "__main__":
    main()
