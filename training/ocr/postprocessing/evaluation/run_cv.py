"""P4: pełny protokół (spec §4): zagnieżdżone LOGO, strojenie progów na wewnętrznym LOGO
(min CER przy harm ≤ HARM_MAX), test na wyłączonej grupie, predykcje out-of-fold → wiersz tabeli.

    PYTHONPATH=. python -m evaluation.run_cv [--split fine|coarse] [--tag option1] [--no-multi] ...

Strojenie używa przybliżenia: zmiana liczby edycji linii przy podmianie JEDNEGO tokenu
(izolowanie), sumowana po tokenach. Wynik testowy jest liczony dokładnie na pełnej korekcie.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from evaluation.compare import compare, per_group_cer
from evaluation.folds import line_groups, nested_logo
from evaluation.metrics import load_predictions, normalize_ws
from fetch_resources import cache_dir
from pp.candidates import Prefilter
from pp.decide import DecisionParams, Scored, best_two, decide, score_key
from pp.error_model import ErrorModel, Params, word_pairs
from pp.lexicon import Lexicon, domain_terms, fold
from pp.lm import UnigramLM
from pp.protect import protected_class, train_abbreviations, upper_share
from pp.tokenize import is_word, restore_case, split_token, token_pairs, tokenize

RESULTS = Path(__file__).resolve().parent.parent / "results"
HARM_MAX = 0.005
GRID = {
    "lam": [0.25, 0.5, 0.75, 1.0, 1.5],
    "d_max": [1, 2, 3],
    "f_min": [0, 1, 5, 20],
    "delta": [x / 2 for x in range(-8, 21)],
    "mu": [0.0, 0.5, 1.0, 2.0, 3.0],
    "L_min": [3, 4, 5, 6],
}


@dataclass
class Data:
    gts: list[str]
    prs: list[str]
    groups: list[str]
    toks: list
    ups: list[float]
    gt_of: list[dict[int, str]]
    base_edits: list[int]
    cands: dict[str, list[tuple[str, int]]]


def prepare(lex: Lexicon, split: str, lex_tag: str = "polimorf", diacritics_only: bool = False,
            predictions: Path | None = None) -> Data:
    rows = load_predictions(predictions) if predictions else load_predictions()
    gts = [normalize_ws(r["ground_truth"]) for r in rows]
    prs = [normalize_ws(r["prediction"]) for r in rows]
    groups = line_groups(rows, split)
    toks = [tokenize(p) for p in prs]
    ups = [upper_share(t) for t in toks]
    gt_of = []
    for g, p in zip(gts, prs):
        gs, ps = g.split(), p.split()
        gt_of.append({pi: gs[gi] for gi, pi in token_pairs(gs, ps)})
    keys = sorted({t.key for line in toks for t in line if is_word(t.core) and not lex.known(t.key) and len(t.key) >= 3})
    h = hashlib.sha256((lex_tag + "\n" + "\n".join(keys)).encode()).hexdigest()[:12]
    cp = cache_dir() / f"prefilter_d3_{h}.pkl"
    if cp.exists():
        cands = pickle.loads(cp.read_bytes())
    else:
        t0 = time.time()
        cands = Prefilter(lex, d_max=3).batch(keys, 3)
        cp.write_bytes(pickle.dumps(cands))
        print(f"prefiltr: {len(keys)} kluczy w {time.time() - t0:.0f}s -> {cp}")
    if diacritics_only:  # opcja 2 osobno: tylko warianty różniące się diakrytykami
        cands = {k: [(c, dd) for c, dd in v if fold(c) == fold(k)] for k, v in cands.items()}
    return Data(gts, prs, groups, toks, ups, gt_of, [Levenshtein.distance(g, p) for g, p in zip(gts, prs)], cands)


def eligible(d: Data, i: int, pi: int, abbrev, disabled=()) -> bool:
    t = d.toks[i][pi]
    return t.key in d.cands and protected_class(t, d.ups[i], abbrev, disabled) is None


class Fold:
    """Model kanału + skróty wyuczone na zbiorze linii `train`; ocenia klucze z linii `evals`."""

    def __init__(self, d: Data, lex: Lexicon, lm: UnigramLM, train: list[int], em: Params, forward: bool = False,
                 protect: frozenset[str] = frozenset()):
        self.forward = forward
        self.model = ErrorModel.fit(word_pairs([d.gts[i] for i in train], [d.prs[i] for i in train],
                                               [d.groups[i] for i in train]), em)
        self.abbrev = lex.abbrev | train_abbreviations((d.gts[i] for i in train), lex.known) | protect
        self.lm = lm
        self._sc: dict[str, Scored] = {}

    def scored(self, d: Data, key: str) -> Scored:
        s = self._sc.get(key)
        if s is None:
            s = self._sc[key] = score_key(key, d.cands[key], self.model, self.lm, self.forward)
        return s


def records(d: Data, fold: Fold, lines: list[int], disabled=()) -> list[tuple[int, int, Scored]]:
    out = []
    for i in lines:
        for pi in range(len(d.toks[i])):
            if eligible(d, i, pi, fold.abbrev, disabled):
                out.append((i, pi, fold.scored(d, d.toks[i][pi].key)))
    return out


def n_correct(d: Data, lines) -> int:
    return sum(1 for i in lines for pi, t in enumerate(d.toks[i]) if d.gt_of[i].get(pi) == t.raw)


_delta_cache: dict[tuple[int, int, str], int] = {}


def iso_delta(d: Data, i: int, pi: int, cand: str) -> tuple[int, bool, bool]:
    """(Δ edycji linii, broken, fixed) przy podmianie tylko tokenu pi na cand."""
    t = d.toks[i][pi]
    new = t.rebuild(restore_case(t.core, cand))
    k = (i, pi, cand)
    if k not in _delta_cache:
        parts = [x.raw for x in d.toks[i]]
        parts[pi] = new
        _delta_cache[k] = Levenshtein.distance(d.gts[i], " ".join(parts)) - d.base_edits[i]
    g = d.gt_of[i].get(pi)
    return _delta_cache[k], g == t.raw and new != g, new == g


def tune(d: Data, recs, n_ok: int, lp_unseen: float, harm_max: float = HARM_MAX) -> tuple[DecisionParams, dict]:
    best, best_info = None, None
    for lam, dm, fm in itertools.product(GRID["lam"], GRID["d_max"], GRID["f_min"]):
        g, m, L, dl, br = [], [], [], [], []
        for i, pi, sc in recs:
            b, sb, sb2 = best_two(sc, lam, dm, fm)
            if b < 0:
                continue
            dd, broken, _ = iso_delta(d, i, pi, sc.cands[b])
            g.append(sb - (sc.keep_ch + lam * lp_unseen))
            m.append(sb - sb2)
            L.append(len(sc.key))
            dl.append(dd)
            br.append(broken)
        if not g:
            continue
        g, m, L, dl, br = map(np.asarray, (g, m, L, dl, br))
        for lmin in GRID["L_min"]:
            mL = L >= lmin
            for mu in GRID["mu"]:
                mLm = mL & (m >= mu)
                for delta in GRID["delta"]:
                    acc = mLm & (g >= delta)
                    harm = br[acc].sum() / n_ok
                    if harm > harm_max:
                        continue
                    tot = int(dl[acc].sum())
                    if best is None or tot < best_info["d_edits"]:
                        best = DecisionParams(lam, delta, mu, dm, lmin, fm)
                        best_info = {"d_edits": tot, "harm": float(harm), "n_changed": int(acc.sum())}
    return best, best_info


def apply(d: Data, fold: Fold, lines: list[int], p: DecisionParams, lp_unseen: float, out: list[str], disabled=()):
    for i in lines:
        parts = [t.raw for t in d.toks[i]]
        for pi, t in enumerate(d.toks[i]):
            if len(t.key) >= p.L_min and eligible(d, i, pi, fold.abbrev, disabled):
                new = decide(fold.scored(d, t.key), p, lp_unseen)
                if new:
                    parts[pi] = t.rebuild(restore_case(t.core, new))
        out[i] = " ".join(parts)


def lost_paths(d: Data, fold: Fold, lex: Lexicon, lines, p: DecisionParams) -> dict:
    """Błędne tokeny OOV: czy GT jest wśród kandydatów prefiltra; czy GT spoza zasięgu wygrałby z top-1."""
    c = {"wrong_eligible": 0, "gt_in_cands": 0, "gt_target_outside": 0, "gt_outside_would_win": 0}
    for i in lines:
        for pi, t in enumerate(d.toks[i]):
            g = d.gt_of[i].get(pi)
            if g is None or g == t.raw or not eligible(d, i, pi, fold.abbrev):
                continue
            gk = split_token(g).core.lower()
            c["wrong_eligible"] += 1
            sc = fold.scored(d, t.key)
            if gk in sc.cands:
                c["gt_in_cands"] += 1
            elif lex.target(gk):
                c["gt_target_outside"] += 1
                b, sb, _ = best_two(sc, p.lam, 3, p.f_min)
                sg = fold.model.logp(t.key, gk, fold.forward) + p.lam * fold.lm.logp(gk)
                c["gt_outside_would_win"] += sg > sb
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fine", choices=["fine", "coarse"])
    ap.add_argument("--tag", default="option1")
    ap.add_argument("--no-multi", action="store_true", help="ablacja: bez jednostek wieloznakowych")
    ap.add_argument("--uniform", action="store_true", help="ablacja: jednostajny Levenshtein zamiast wyuczonych pomyłek")
    ap.add_argument("--no-group-cap", action="store_true", help="ablacja: bez limitu udziału grupy w licznikach błędów")
    ap.add_argument("--no-protect", action="store_true", help="ablacja: bez klas chronionych P2–P8 (P1 zostaje)")
    ap.add_argument("--sjp", action="store_true", help="ablacja: + sjp odmiany")
    ap.add_argument("--forward", action="store_true", help="ablacja: forward (suma po ścieżkach) zamiast Viterbi")
    ap.add_argument("--domain", default="", help="języki listy TA chroniącej terminy, np. la,pl albo en,la,pl; "
                                                 "domyślnie wyłączona; progi stroimy bez niej")
    ap.add_argument("--domain-in-tuning", action="store_true", help="ablacja: lista także przy strojeniu progów")
    ap.add_argument("--diacritics-only", action="store_true", help="ablacja: opcja 2 osobno (tylko warianty diakrytyczne)")
    ap.add_argument("--harm-max", type=float, default=HARM_MAX, help="limit harm na walidacji wewnętrznej")
    ap.add_argument("--deploy", action="store_true",
                    help="progi do wdrożenia: LOGO na wszystkich grupach -> results/deploy_params.json (to NIE jest test)")
    ap.add_argument("--dump", type=Path, help="zapisz predykcje out-of-fold (z tekstem linii, poza repo!)")
    ap.add_argument("--predictions", type=Path,
                    help="inny plik predykcji (format data/baseline/predictions.jsonl), np. Surya v3 albo test v1.1")
    ap.add_argument("--out-dir", type=Path, default=RESULTS, help="katalog wyników (eksperymenty poza results/)")
    a = ap.parse_args()
    t0 = time.time()
    terms = frozenset(domain_terms(a.domain.split(","))) if a.domain else frozenset()
    tuned_terms = terms if a.domain_in_tuning else frozenset()
    lex = Lexicon(use_sjp=a.sjp)
    lm = UnigramLM(lex.targets())
    lp_unseen = math.log(lm.unseen) - lm.logz
    d = prepare(lex, a.split, "polimorf+sjp" if a.sjp else "polimorf", a.diacritics_only, a.predictions)
    em = Params(multi=not a.no_multi, uniform=a.uniform,
                max_group_share=1.0 if a.no_group_cap else Params.max_group_share)
    disabled = tuple(c for c in ("P2_digits", "P3_num_unit", "P4_abbr_dot", "P5_abbr_list", "P6_short_upper",
                                 "P7_len_le_2", "P8_hyphen")) if a.no_protect else ()
    print(f"przygotowanie {time.time() - t0:.0f}s")
    if a.deploy:
        recs, ok = [], 0
        for g in sorted(set(d.groups)):
            itest = [i for i, x in enumerate(d.groups) if x == g]
            itrain = [i for i, x in enumerate(d.groups) if x != g]
            recs += records(d, Fold(d, lex, lm, itrain, em, a.forward), itest, disabled)
            ok += n_correct(d, itest)
        p, info = tune(d, recs, ok, lp_unseen, a.harm_max)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "deploy_params.json").write_text(json.dumps(
            {"harm_max": a.harm_max, "params": p.as_dict(), "logo_all_groups": info,
             "note": "progi strojone na LOGO wszystkich 8 grup; model kanału do wdrożenia uczony na całości"},
            indent=1), encoding="utf-8")
        print(f"deploy: {p.as_dict()} {info}")
        return
    out = list(d.prs)
    folds_info = []
    for g, train, test, inner in nested_logo(d.groups):
        t1 = time.time()
        recs, ok = [], 0
        for h, itrain, itest in inner:
            f_in = Fold(d, lex, lm, itrain, em, a.forward, tuned_terms)
            recs += records(d, f_in, itest, disabled)
            ok += n_correct(d, itest)
        p, info = tune(d, recs, ok, lp_unseen, a.harm_max)
        f_out = Fold(d, lex, lm, train, em, a.forward, terms)
        apply(d, f_out, test, p, lp_unseen, out, disabled)
        lost = lost_paths(d, f_out, lex, test, p)
        tt = compare([d.gts[i] for i in test], [d.prs[i] for i in test], [out[i] for i in test],
                     [d.groups[i] for i in test], B_groups=10, B_lines=10)
        row = {"test_group": g, **p.as_dict(), "inner_d_edits": info["d_edits"], "inner_harm": info["harm"],
               "inner_changed": info["n_changed"], "test_d_cer": tt["d_cer"], "test_harm": tt["harm"],
               "test_fixed": tt["fixed"], "test_broken": tt["broken"], **lost, "seconds": time.time() - t1}
        folds_info.append(row)
        print(f"{g:20s} params {p.as_dict()} | inner Δedits {info['d_edits']} harm {info['harm']:.4f} | "
              f"test ΔCER {tt['d_cer'] * 100:+.2f} p.p. harm {tt['harm']:.4f} fixed {tt['fixed']} broken {tt['broken']} "
              f"| GT w kandydatach {lost['gt_in_cands']}/{lost['wrong_eligible']} [{row['seconds']:.0f}s]", flush=True)

    rows = load_predictions(a.predictions) if a.predictions else load_predictions()
    raw_gts = [r["ground_truth"] for r in rows]
    res = {"variant": a.tag, "split": a.split, "harm_max": a.harm_max, **compare(raw_gts, d.prs, out, d.groups)}
    print(f"\n{a.tag} ({a.split}): CER {res['base_cer']:.5f} -> {res['cer']:.5f} (Δ {res['d_cer'] * 100:+.3f} p.p., "
          f"{res['d_cer_rel'] * 100:+.1f}%), CI grupy [{res['ci_groups_lo'] * 100:+.2f}, {res['ci_groups_hi'] * 100:+.2f}], "
          f"CI linie [{res['ci_lines_lo'] * 100:+.2f}, {res['ci_lines_hi'] * 100:+.2f}], p_sign {res['p_sign']:.3g}; "
          f"WER {res['base_wer']:.4f} -> {res['wer']:.4f}; fixed {res['fixed']} broken {res['broken']} "
          f"changed_wrong {res['changed_wrong']} net {res['net']}; precision {res['precision']:.3f} "
          f"recall {res['recall']:.3f} harm {res['harm']:.4f}")
    res_dir = a.out_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    stem = f"p4_{a.tag}_{a.split}"
    with open(res_dir / f"{stem}_folds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(folds_info[0]))
        w.writeheader()
        for r in folds_info:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = res_dir / "p4_summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res))
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in res.items()})
    with open(res_dir / f"{stem}_per_group.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": per_group_cer(raw_gts, d.prs, d.groups),
                   a.tag: per_group_cer(raw_gts, out, d.groups)}, f, indent=1)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for r, o in zip(rows, out):
                f.write(json.dumps({"file_name": r["file_name"], "ground_truth": r["ground_truth"],
                                    "prediction": r["prediction"], "corrected": o}, ensure_ascii=False) + "\n")
    print(f"razem {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
