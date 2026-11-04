"""Oracle: górne granice zysku (ticket 10) + konwencja diakrytyków w GT (ticket 09).

Oracle podmienia token predykcji na token GT (wyrównanie 1:1 na słowach), jeśli spełnia
warunek danego wariantu. Nie łączy ani nie dzieli tokenów, więc to pułap korekty
token-po-tokenie, a nie całego postprocessingu.

    PYTHONPATH=. python -m evaluation.oracle
"""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from rapidfuzz.distance import Levenshtein

from evaluation.compare import compare, per_group_cer
from evaluation.folds import line_groups
from evaluation.metrics import load_predictions, normalize_ws, score
from pp.lexicon import fold, polimorf_forms

RESULTS = Path(__file__).resolve().parent.parent / "results"
ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


def core(tok: str) -> str:
    """Token bez brzegowej interpunkcji/symboli."""
    i, j = 0, len(tok)
    while i < j and unicodedata.category(tok[i])[0] in "PS":
        i += 1
    while j > i and unicodedata.category(tok[j - 1])[0] in "PS":
        j -= 1
    return tok[i:j]


def is_word(c: str) -> bool:
    return bool(c) and all(ch.isalpha() or ch == "-" for ch in c) and any(ch.isalpha() for ch in c)


def oov(tok: str, lex: frozenset[str]) -> bool:
    c = core(tok)
    return is_word(c) and c.lower() not in lex


def pairs(gt_tokens: list[str], pred_tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for op in Levenshtein.opcodes(gt_tokens, pred_tokens):
        if op.tag in ("equal", "replace"):
            for k in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                out.append((op.src_start + k, op.dest_start + k))
    return out


def apply_oracle(gts: list[str], preds: list[str], cond: Callable[[str, str], bool]) -> tuple[list[str], int]:
    out, n = [], 0
    for gt, pr in zip(gts, preds):
        g, p = normalize_ws(gt).split(), normalize_ws(pr).split()
        q = list(p)
        for gi, pi in pairs(g, p):
            if g[gi] != p[pi] and cond(g[gi], p[pi]):
                q[pi] = g[gi]
                n += 1
        out.append(" ".join(q))
    return out, n


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = load_predictions()
    gts = [r["ground_truth"] for r in rows]
    prs = [r["prediction"] for r in rows]
    groups = line_groups(rows)
    lex = polimorf_forms()
    print(f"PoliMorf 2026: {len(lex)} form")

    def reach(d: int) -> Callable[[str, str], bool]:
        def f(g: str, p: str) -> bool:
            cg, cp = core(g).lower(), core(p).lower()
            return oov(p, lex) and cg in lex and Levenshtein.distance(cg, cp) <= d
        return f

    variants: dict[str, Callable[[str, str], bool] | None] = {
        "baseline": None,
        "O0_hygiene_nfc_zw_ws": None,
        "O1_any_wrong_token->GT": lambda g, p: True,
        "O1_oov->GT": lambda g, p: oov(p, lex),
        "O1_oov->GT_inlex_lev<=3": reach(3),
        "O1_oov->GT_inlex_lev<=2": reach(2),
        "O1_oov->GT_inlex_lev<=1": reach(1),
        "O2_diakrytyki_wszystkie->GT": lambda g, p: fold(g) == fold(p),
        "O2_diakrytyki_tylko_oov->GT": lambda g, p: fold(g) == fold(p) and oov(p, lex),
    }
    table, per_group = [], {}
    for name, cond in variants.items():
        if name == "baseline":
            out, n = prs, 0
        elif name.startswith("O0"):
            out = [unicodedata.normalize("NFC", p).translate(ZW) for p in prs]
            n = sum(a != b for a, b in zip(out, prs))
        else:
            out, n = apply_oracle(gts, prs, cond)
        row = {"variant": name, "tokens_replaced": n}
        row.update(compare(gts, prs, out, groups))
        table.append(row)
        per_group[name] = per_group_cer(gts, out, groups)
        print(f"{name:42s} CER {row['cer']:.5f}  dCER {row['d_cer']*100:+.2f} p.p.  "
              f"aux {row['cer_aux']:.5f}  WER {row['wer']:.5f}  repl {n}")

    # Diagnostyka metryki: diakrytyki zdjęte po obu stronach (nie jest korektą).
    s = score(gts, prs, lambda t: normalize_ws(fold(t)))
    print(f"{'metryka: fold() po obu stronach':42s} CER {s.cer:.5f}")
    diag = {"variant": "METRIC_fold_both_sides", "cer": s.cer, "wer": s.wer}

    keys = list(table[0].keys())
    with open(RESULTS / "oracle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table + [diag]:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    gnames = sorted(set(groups))
    with open(RESULTS / "oracle_per_group.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", *gnames])
        for name, d in per_group.items():
            w.writerow([name, *(f"{d[g]:.5f}" for g in gnames)])

    gt_diacritics(rows, gts, prs, groups, lex)


def gt_diacritics(rows, gts, prs, groups, lex) -> None:
    """Ticket 09: tokeny GT spoza słownika, które z diakrytykami są w słowniku."""
    gt_cores = [[core(t) for t in normalize_ws(g).split()] for g in gts]
    need = {fold(c.lower()) for line in gt_cores for c in line if is_word(c) and c.lower() not in lex}
    by_fold: dict[str, set[str]] = defaultdict(set)
    for form in lex:
        k = fold(form)
        if k in need:
            by_fold[k].add(form)

    stats = defaultdict(Counter)
    suspects = []
    for r, line, pr, gr in zip(rows, gt_cores, prs, groups):
        pcores = {core(t).lower() for t in normalize_ws(pr).split()}
        for c in line:
            if not is_word(c):
                continue
            lc = c.lower()
            stats[gr]["words"] += 1
            if lc in lex:
                continue
            stats[gr]["oov"] += 1
            variants = sorted(by_fold.get(fold(lc), set()) - {lc})
            if not variants:
                continue
            stats[gr]["oov_diac_variant"] += 1
            pred_has = sorted(v for v in variants if v in pcores)
            stats[gr]["pred_has_variant"] += bool(pred_has)
            suspects.append({
                "file_name": r["file_name"], "group": gr, "gt_token": c,
                "lexicon_variants": "|".join(variants[:5]),
                "pred_has_variant": "|".join(pred_has),
                "decision": "",
            })

    with open(RESULTS / "gt_diacritics_by_group.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "words", "oov", "oov_pct", "oov_diac_variant", "diac_pct_of_words", "pred_has_variant"])
        tot = Counter()
        for g in sorted(stats):
            s = stats[g]
            tot.update(s)
            w.writerow([g, s["words"], s["oov"], f"{100*s['oov']/s['words']:.2f}", s["oov_diac_variant"],
                        f"{100*s['oov_diac_variant']/s['words']:.2f}", s["pred_has_variant"]])
        w.writerow(["ALL", tot["words"], tot["oov"], f"{100*tot['oov']/tot['words']:.2f}", tot["oov_diac_variant"],
                    f"{100*tot['oov_diac_variant']/tot['words']:.2f}", tot["pred_has_variant"]])
    suspects.sort(key=lambda x: (not x["pred_has_variant"], x["group"], x["file_name"]))
    with open(RESULTS / "suspect_gt_diacritics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(suspects[0].keys()))
        w.writeheader()
        w.writerows(suspects)
    print(f"ticket 09: {len(suspects)} podejrzanych tokenów GT "
          f"({sum(bool(s['pred_has_variant']) for s in suspects)} z wariantem w predykcji)")


if __name__ == "__main__":
    main()
