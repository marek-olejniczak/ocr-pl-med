"""Paczka do wdrożenia z predykcji z etykietami: model pomyłek uczony na wszystkich liniach + wystrojone progi.

    python -m evaluation.build_bundle --predictions P.jsonl --groups data/v12/groups.csv \\
        --params ../results/v12/deploy_params_top5.json --params-rescored ../results/v12/deploy_params_surya.json \\
        --out ../package/src/ocr_postprocess/models/NAZWA
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from evaluation.folds import line_groups, read_group_map
from evaluation.metrics import load_predictions, normalize_ws
from ocr_postprocess import bundle
from ocr_postprocess.error_model import ErrorModel, Params, word_pairs
from ocr_postprocess.lexicon import Lexicon, domain_terms
from ocr_postprocess.protect import train_abbreviations


def build(predictions: Path, groups_csv: Path | None, params_json: Path, domain: list[str], out: Path, note: str = "",
          rescored_json: Path | None = None) -> dict:
    rows = load_predictions(predictions)
    gts = [normalize_ws(r["ground_truth"]) for r in rows]
    prs = [normalize_ws(r["prediction"]) for r in rows]
    groups = line_groups(rows, "fine", read_group_map(groups_csv) if groups_csv else None)
    params = json.loads(params_json.read_text(encoding="utf-8"))["params"]
    rescored = json.loads(rescored_json.read_text(encoding="utf-8"))["params"] if rescored_json else None
    lex = Lexicon(extra_known=domain_terms(domain))
    model = ErrorModel.fit(word_pairs(gts, prs, groups), Params(multi=True))
    return bundle.save(out, model, {
        "built": date.today().isoformat(), "params": params, "params_rescored": rescored, "domain_langs": domain,
        "train_abbrev": sorted(train_abbreviations(gts, lex.known)), "lines": len(rows),
        "groups": sorted(set(groups)), "predictions": predictions.name, "note": note})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--groups", type=Path)
    ap.add_argument("--params", type=Path, required=True, help="progi statyczne (top-5)")
    ap.add_argument("--params-rescored", type=Path, help="progi z oceną Suryi (run_rescore --deploy)")
    ap.add_argument("--domain", default="en,la,pl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    meta = build(a.predictions, a.groups, a.params, [x for x in a.domain.split(",") if x], a.out, a.note,
                 a.params_rescored)
    print(json.dumps({k: v for k, v in meta.items() if k != "train_abbrev"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
