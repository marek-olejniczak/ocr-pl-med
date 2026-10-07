"""Turn benchmark predictions on the evaluation sets into per-augmentation tables.

robustness set (paired)
    For every condition: CER, and the change against the clean render of the
    same lines with a 95% bootstrap interval over base lines. Because every
    base line appears under every condition, the change is the cost of the
    condition alone.

val set (as trained)
    CER broken down by font, content kind and each augmentation as it was
    applied in the training generator, from the per-line metadata.

Usage:
    python src/robustness_report.py robustness --labels output/robustness/labels.jsonl \\
        v1=.../surya_lora_ocr800k/robustness/raw_predictions.csv \\
        v2=.../surya_lora_v2_800k/robustness/raw_predictions.csv --output docs/odpornosc.md
    python src/robustness_report.py val --labels output/v2_val10k/labels.jsonl \\
        v2=.../surya_lora_v2_800k/v2_val10k/raw_predictions.csv --output docs/val_breakdown.md
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ablation_report import levenshtein, normalize


def load_labels(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["file_name"]: r for r in map(json.loads, f)}


def load_predictions(path: Path, labels: dict) -> dict[str, tuple[int, int]]:
    """file_name -> (edit distance, ground-truth length)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = Path(r["file_name"]).name
            if name not in labels:
                continue
            gt = normalize(labels[name]["text"])
            out[name] = (levenshtein(gt, normalize(r["prediction"])), len(gt))
    return out


def pooled(pairs) -> float:
    d = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return d / n if n else float("nan")


def fmt(v: float) -> str:
    return f"{100 * v:.1f}%"


# ---------------------------------------------------------------- robustness

def robustness(labels: dict, models: dict[str, dict], n_boot: int) -> None:
    conds = []
    for r in labels.values():
        if r["condition"] not in [c[0] for c in conds]:
            conds.append((r["condition"], r["family"], r["severity"]))
    by_base: dict[str, dict[int, dict[str, tuple]]] = {}
    for m, preds in models.items():
        t = defaultdict(dict)
        for name, (d, n) in preds.items():
            r = labels[name]
            t[r["base_id"]][r["condition"]] = (d, n)
        # only base lines read under every condition, so pairs stay complete
        full = {b: v for b, v in t.items() if len(v) == len(conds)}
        by_base[m] = full

    print("## Koszt kazdego warunku (zbior kontrolny, pary)\n")
    print("Zmiana CER wzgledem czystego renderu tych samych linii; 95% przedzial "
          "bootstrap po liniach bazowych.\n")
    names = list(models)
    head = "| warunek | rodzina | sila | " + " | ".join(f"{m} CER | {m} zmiana" for m in names) + " |"
    print(head)
    print("|" + "---|" * (3 + 2 * len(names)))

    rng = random.Random(0)
    results = {}
    for m in names:
        bases = list(by_base[m])
        draws = [[rng.choice(bases) for _ in bases] for _ in range(n_boot)]
        res = {}
        for cname, _, _ in conds:
            cer = pooled([by_base[m][b][cname] for b in bases])
            base_cer = pooled([by_base[m][b]["clean"] for b in bases])
            boot = sorted(
                pooled([by_base[m][b][cname] for b in d]) - pooled([by_base[m][b]["clean"] for b in d])
                for d in draws)
            res[cname] = (cer, cer - base_cer, boot[int(.025 * n_boot)], boot[int(.975 * n_boot)])
        results[m] = res

    for cname, fam, sev in conds:
        cells = []
        for m in names:
            cer, delta, lo, hi = results[m][cname]
            sig = "" if lo <= 0 <= hi else " *"
            cells.append(f"{fmt(cer)} | {100 * delta:+.1f} ({100 * lo:+.1f}..{100 * hi:+.1f}){sig}")
        print(f"| {cname} | {fam} | {sev} | " + " | ".join(cells) + " |")
    print("\n`*` przedzial nie obejmuje zera, czyli warunek zmienia CER istotnie.")
    print(f"Linii bazowych: " + ", ".join(f"{m} {len(by_base[m])}" for m in names) + "\n")

    print("## Ranking rodzin (najgorszy warunek w rodzinie)\n")
    main = names[-1]
    worst = {}
    for cname, fam, _ in conds:
        if fam == "none":
            continue
        d = results[main][cname][1]
        if fam not in worst or d > worst[fam][1]:
            worst[fam] = (cname, d)
    print(f"| rodzina | najgorszy warunek | zmiana CER ({main}) |")
    print("|---|---|---|")
    for fam, (cname, d) in sorted(worst.items(), key=lambda kv: -kv[1][1]):
        print(f"| {fam} | {cname} | {100 * d:+.1f} pkt |")
    print()


# ----------------------------------------------------------------------- val

def _bucket_tables(labels: dict, models: dict, title: str, key, min_n: int = 30) -> None:
    names = list(models)
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for m, preds in models.items():
        for name, pair in preds.items():
            k = key(labels[name])
            if k is not None:
                groups[k][m].append(pair)
    rows = [(k, v) for k, v in groups.items() if len(v[names[-1]]) >= min_n]
    rows.sort(key=lambda kv: -pooled(kv[1][names[-1]]))
    print(f"## {title}\n")
    print("| grupa | linii | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for k, v in rows:
        print(f"| {k} | {len(v[names[-1]])} | " + " | ".join(fmt(pooled(v[m])) for m in names) + " |")
    print()


def val(labels: dict, models: dict) -> None:
    _bucket_tables(labels, models, "CER wg kroju pisma", lambda r: r.get("font"))
    _bucket_tables(labels, models, "CER wg rodzaju tresci", lambda r: r.get("kind"))
    _bucket_tables(labels, models, "CER wg tla", lambda r: str(r.get("rule")))
    _bucket_tables(labels, models, "CER wg sasiednich linii",
                   lambda r: "sasiedzi" if r.get("neighbours") else "brak")
    _bucket_tables(labels, models, "CER wg morfologii",
                   lambda r: (r.get("morphology") or "brak").rstrip("0123456789."))
    _bucket_tables(labels, models, "CER wg znieksztalcenia", lambda r: r.get("elastic") or "brak")
    _bucket_tables(labels, models, "CER wg urzadzenia", lambda r: r.get("scan_profile") or "brak")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["robustness", "val"])
    ap.add_argument("--labels", required=True, help="labels.jsonl of the evaluation set")
    ap.add_argument("runs", nargs="+", help="name=path/to/raw_predictions.csv (main model last)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.output:
        from ablation_report import _Tee
        sys.stdout = _Tee(sys.stdout, open(args.output, "w", encoding="utf-8"))

    labels = load_labels(Path(args.labels))
    models = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        models[name] = load_predictions(Path(path), labels)
    if args.mode == "robustness":
        robustness(labels, models, args.bootstrap)
    else:
        val(labels, models)


if __name__ == "__main__":
    main()
