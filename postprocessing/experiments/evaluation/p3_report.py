"""P3: model błędu per fold LOGO; log-likelihood par (GT, pred) na wyłączonej grupie
vs jednostajny Levenshtein (kryterium ukończenia P3).

    python -m evaluation.p3_report --out-dir KATALOG [--predictions P.jsonl --groups G.csv]
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from evaluation.cli import cli
from evaluation.folds import line_groups, logo, read_group_map
from evaluation.metrics import load_predictions, normalize_ws
from ocr_postprocess.error_model import ErrorModel, Params, word_pairs

RESULTS = Path()  # z --out-dir


def mean_ll(model: ErrorModel, pairs, forward: bool = False) -> float:
    return sum(model.logp(p, g, forward) for g, p, _ in pairs) / len(pairs)


def main() -> None:
    global RESULTS
    a = cli(__doc__)
    RESULTS = a.out_dir
    rows = load_predictions(a.predictions)
    gts = [normalize_ws(r["ground_truth"]) for r in rows]
    prs = [normalize_ws(r["prediction"]) for r in rows]
    groups = line_groups(rows, "fine", read_group_map(a.groups) if a.groups else None)
    out = []
    for test_group, train, test in logo(groups):
        t0 = time.time()
        tr = word_pairs([gts[i] for i in train], [prs[i] for i in train], [groups[i] for i in train])
        te = word_pairs([gts[i] for i in test], [prs[i] for i in test], [groups[i] for i in test])
        te_err = [x for x in te if x[0] != x[1]]
        bm = ErrorModel.fit(tr)
        bm1 = ErrorModel.fit(tr, Params(multi=False))
        uni = ErrorModel.uniform(tr)
        row = {
            "test_group": test_group, "train_pairs": len(tr), "test_pairs": len(te), "test_err_pairs": len(te_err),
            "inventory": len(bm.lmulti),
            "ll_uniform_all": mean_ll(uni, te), "ll_bm_all": mean_ll(bm, te),
            "ll_uniform_err": mean_ll(uni, te_err), "ll_bm1_err": mean_ll(bm1, te_err),
            "ll_bm_err": mean_ll(bm, te_err), "ll_bm_err_forward": mean_ll(bm, te_err, forward=True),
            "seconds": time.time() - t0,
        }
        out.append(row)
        print(f"{test_group:20s} train {len(tr):5d} test {len(te):4d} (err {len(te_err):4d}) inv {row['inventory']:3d} | "
              f"LL/para wszystkie: uni {row['ll_uniform_all']:.3f} BM {row['ll_bm_all']:.3f} | "
              f"błędne: uni {row['ll_uniform_err']:.3f} BM-1:1 {row['ll_bm1_err']:.3f} BM {row['ll_bm_err']:.3f} "
              f"fwd {row['ll_bm_err_forward']:.3f}  [{row['seconds']:.0f}s]", flush=True)
    with open(RESULTS / "p3_loglik.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Inwentarz z całości (model do wdrożenia; NIE jest wynikiem testu)
    allp = word_pairs(gts, prs, groups)
    full = ErrorModel.fit(allp)
    with open(RESULTS / "p3_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "tgt", "p"])
        for s, t, p in full.inventory():
            w.writerow([s, t, f"{p:.4f}"])
    print("inwentarz (całość):", ", ".join(f"{s}→{t} {p:.3f}" for s, t, p in full.inventory()[:15]))


if __name__ == "__main__":
    main()
