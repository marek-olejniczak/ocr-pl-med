"""P2: pokrycie GT słownikiem (vs ticket 05), wiersz „opcja 0”, statystyki klas chronionych.

    python -m evaluation.p2_report --out-dir KATALOG [--predictions P.jsonl --groups G.csv]
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from evaluation.compare import compare
from evaluation.cli import cli
from evaluation.folds import line_groups, logo, read_group_map
from evaluation.metrics import load_predictions, normalize_ws
from ocr_postprocess.lexicon import Lexicon, polimorf_forms, sjp_forms
from ocr_postprocess.lm import UnigramLM, nkjp_counts
from ocr_postprocess.normalize import hygiene
from ocr_postprocess.protect import CLASSES, protected_class, train_abbreviations, upper_share
from ocr_postprocess.tokenize import is_word, token_pairs as pairs, tokenize

RESULTS = Path()  # z --out-dir
LETTERS = re.compile(r"[^\W\d_]+")


def main() -> None:
    global RESULTS
    a = cli(__doc__)
    RESULTS = a.out_dir
    rows = load_predictions(a.predictions)
    gts = [r["ground_truth"] for r in rows]
    prs = [r["prediction"] for r in rows]
    groups = line_groups(rows, "fine", read_group_map(a.groups) if a.groups else None)

    # 1. Pokrycie (definicja ticketu 05: ciągi liter z GT, małe litery)
    toks = [m.lower() for g in gts for m in LETTERS.findall(g)]
    pm, sj = polimorf_forms(), sjp_forms()
    cov = {
        "polimorf2026": sum(t in pm for t in toks) / len(toks),
        "polimorf2026+sjp": sum(t in pm or t in sj for t in toks) / len(toks),
    }
    counts = nkjp_counts()
    lex = Lexicon(counts=counts)
    lex_sjp = Lexicon(use_sjp=True, counts=counts)
    cov["known(default)"] = sum(lex.known(t) for t in toks) / len(toks)
    cov["target(default)"] = sum(lex.target(t) for t in toks) / len(toks)
    cov["target(+sjp)"] = sum(lex_sjp.target(t) for t in toks) / len(toks)
    print(f"tokenów GT (ciągi liter): {len(toks)}")
    for k, v in cov.items():
        print(f"  pokrycie {k:20s} {100 * v:.2f}%")
    lm = UnigramLM(lex.targets(), counts)
    print(f"NKJP po czyszczeniu: {len(counts)} form; SGT r*(1)={lm.rstar[1]:.3f} r*(2)={lm.rstar[2]:.3f}; "
          f"target: {len(lex.common)} pospolitych + {len(lex.proper_target)} własnych (c≥5)")

    # 2. Opcja 0
    out = [hygiene(p, loops=False) for p in prs]  # P2: higiena bez deloop (wynik historyczny)
    row0 = {"variant": "option0_hygiene", "lines_changed": sum(a != b for a, b in zip(out, prs))}
    row0.update(compare(gts, prs, out, groups))
    print(f"opcja 0: CER {row0['cer']:.5f} (Δ {row0['d_cer'] * 100:+.3f} p.p.), linii zmienionych {row0['lines_changed']}")
    with open(RESULTS / "p2_option0.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row0))
        w.writeheader()
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row0.items()})

    # 3. Klasy chronione na tokenach OOV predykcji (wyrównanie 1:1), skróty z GT tylko z treningu foldu
    stat = Counter()
    for test_group, train, test in logo(groups):
        abbr = lex.abbrev | train_abbreviations((gts[i] for i in train), lex.known)
        for i in test:
            g, p = normalize_ws(gts[i]).split(), normalize_ws(prs[i]).split()
            ptoks = tokenize(" ".join(p))
            ups = upper_share(ptoks)
            for gi, pi in pairs(g, p):
                t = ptoks[pi]
                if is_word(t.core) and lex.known(t.key):
                    continue
                ok = g[gi] == p[pi]
                cls = protected_class(t, ups, abbr)
                stat[("correct" if ok else "wrong", cls or "unprotected")] += 1
    with open(RESULTS / "p2_protect.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "oov_correct", "oov_wrong"])
        for c in (*CLASSES, "unprotected"):
            w.writerow([c, stat[("correct", c)], stat[("wrong", c)]])
    tc = sum(v for (s, _), v in stat.items() if s == "correct")
    tw = sum(v for (s, _), v in stat.items() if s == "wrong")
    print(f"OOV/nie-słowo w pred (1:1): poprawnych {tc}, błędnych {tw}; "
          f"po ochronie narażone poprawne {stat[('correct', 'unprotected')]} "
          f"({100 * stat[('correct', 'unprotected')] / tc:.1f}%), błędne do korekty {stat[('wrong', 'unprotected')]} "
          f"(utrata {100 * (1 - stat[('wrong', 'unprotected')] / tw):.1f}%)")
    with open(RESULTS / "p2_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lexicon", "gt_coverage"])
        for k, v in cov.items():
            w.writerow([k, f"{v:.5f}"])


if __name__ == "__main__":
    main()
