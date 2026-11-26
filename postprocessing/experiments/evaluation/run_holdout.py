"""Protokół dev → test (benchmark/splits): progi i model kanału tylko z dev, test oceniany raz.

Run_cv stroi progi także na grupach testowych, więc jego liczby na teście nie są ściśle zamrożone.
Tutaj linie testowe nie biorą udziału ani w strojeniu, ani w uczeniu modelu kanału.

    python -m evaluation.run_holdout --tag base            # tylko LOGO wewnątrz dev
    python -m evaluation.run_holdout --tag base --test     # + jednorazowa ocena testu

Bez `--test` kod nie czyta wyniku na liniach testowych, więc warianty porównuje się na dev.
Do `results/holdout_<tag>.json` trafiają tylko liczby, bez tekstu linii.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from evaluation import run_cv as R
from evaluation.compare import compare
from evaluation.folds import logo
from ocr_postprocess.decide import DecisionParams
from ocr_postprocess.error_model import Params
from ocr_postprocess.lexicon import Lexicon, domain_terms
from ocr_postprocess.lm import UnigramLM

# Tylko dla dataset-v1 (CSV bez kolumny split). benchmark/splits/make_split.py, TEST_SOURCES.
TEST_GROUPS = {"G1_noga", "G2_anatomia_zmysly", "G4_unaczynienie"}


def split_lines(groups: Sequence[str], splits: Sequence[str] | None = None) -> tuple[list[int], list[int]]:
    """Indeksy linii dev i test: z kolumny `split` (manifest zbioru), a gdy jej nie ma, z TEST_GROUPS."""
    is_test = [s == "test" for s in splits] if splits else [g in TEST_GROUPS for g in groups]
    dev = [i for i, t in enumerate(is_test) if not t]
    test = [i for i, t in enumerate(is_test) if t]
    if not dev or not test:
        raise ValueError(f"pusty podział: dev {len(dev)}, test {len(test)}")
    return dev, test


def sub_logo(groups: Sequence[str], lines: Sequence[int]):
    """Leave-one-group-out po podzbiorze linii (indeksy globalne)."""
    for g, train, held in logo([groups[i] for i in lines]):
        yield g, [lines[j] for j in train], [lines[j] for j in held]


def tune_on(d, lex, lm, em, lines, lp_unseen: float, harm_max: float, protect: frozenset[str]) -> tuple[DecisionParams, dict]:
    recs, ok = [], 0
    for _, itrain, itest in sub_logo(d.groups, lines):
        recs += R.records(d, R.Fold(d, lex, lm, itrain, em, protect=protect), itest)
        ok += R.n_correct(d, itest)
    return R.tune(d, recs, ok, lp_unseen, harm_max)


def summary(d, lines: Sequence[int], out: Sequence[str]) -> dict:
    return compare([d.gts[i] for i in lines], [d.prs[i] for i in lines], [out[i] for i in lines],
                   [d.groups[i] for i in lines])


def show(name: str, r: dict) -> None:
    print(f"{name}: CER {r['base_cer']:.5f} -> {r['cer']:.5f} (Δ {r['d_cer'] * 100:+.3f} p.p., "
          f"{r['d_cer_rel'] * 100:+.1f}%), CI linie [{r['ci_lines_lo'] * 100:+.2f}, {r['ci_lines_hi'] * 100:+.2f}], "
          f"WER {r['base_wer']:.4f} -> {r['wer']:.4f}; fixed {r['fixed']} broken {r['broken']} "
          f"changed_wrong {r['changed_wrong']}; precision {r['precision']:.3f} recall {r['recall']:.3f} "
          f"harm {r['harm'] * 100:.2f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--harm-max", type=float, default=0.003)
    ap.add_argument("--domain", default="", help="języki listy TA chroniącej terminy, np. la,pl albo en,la,pl; "
                                                 "domyślnie wyłączona; progi stroimy bez niej")
    ap.add_argument("--domain-in-tuning", action="store_true", help="ablacja: lista także przy strojeniu progów")
    ap.add_argument("--test", action="store_true", help="oceń linie testowe; używać raz, po wyborze wariantu na dev")
    ap.add_argument("--predictions", type=Path, help="inny plik predykcji (format data/baseline/predictions.jsonl)")
    ap.add_argument("--groups", type=Path, help="CSV file_name,group dla linii spoza GROUP_RULES")
    ap.add_argument("--out-dir", type=Path, required=True, help="katalog wyników (same liczby)")
    a = ap.parse_args()
    t0 = time.time()
    terms = frozenset(domain_terms(a.domain.split(","))) if a.domain else frozenset()
    tuned_terms = terms if a.domain_in_tuning else frozenset()
    lex = Lexicon()
    lm = UnigramLM(lex.targets())
    lp_unseen = math.log(lm.unseen) - lm.logz
    d = R.prepare(lex, "fine", predictions=a.predictions, groups_csv=a.groups)
    em = Params(multi=True)
    dev, test = split_lines(d.groups, d.splits)
    print(f"dev {len(dev)} linii, test {len(test)} linii, grupy dev {sorted({d.groups[i] for i in dev})}", flush=True)

    out = list(d.prs)
    for g, train, held in sub_logo(d.groups, dev):
        p, info = tune_on(d, lex, lm, em, train, lp_unseen, a.harm_max, tuned_terms)
        R.apply(d, R.Fold(d, lex, lm, train, em, protect=terms), held, p, lp_unseen, out)
        print(f"{g:20s} {p.as_dict()} wewn. Δedycji {info['d_edits']} harm {info['harm']:.4f}", flush=True)
    res = {"tag": a.tag, "harm_max": a.harm_max, "domain": a.domain, "domain_in_tuning": a.domain_in_tuning, "dev_logo": summary(d, dev, out)}
    show("dev (LOGO)", res["dev_logo"])

    if a.test:
        p, info = tune_on(d, lex, lm, em, dev, lp_unseen, a.harm_max, tuned_terms)
        final = list(d.prs)
        R.apply(d, R.Fold(d, lex, lm, dev, em, protect=terms), test, p, lp_unseen, final)
        res |= {"params": p.as_dict(), "tune_on_dev": info, "test": summary(d, test, final)}
        print(f"progi z dev: {p.as_dict()} {info}")
        show("TEST", res["test"])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / f"holdout_{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"razem {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
