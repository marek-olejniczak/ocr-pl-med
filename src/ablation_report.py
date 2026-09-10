"""Compare benchmark runs: CER with bootstrap confidence intervals.

Reads the raw_predictions.csv files the benchmark writes (columns file_name,
document_id, line_id, ground_truth, prediction) and prints, per model:

    CER overall with a 95% bootstrap interval over lines
    CER per source notebook (document_id up to "_page" / "_line")
    CER per ground-truth length bin
    the most frequent character substitutions

One training per variant is all the budget allows, so the uncertainty comes
from resampling the test lines, not from repeated trainings. Two models
whose intervals overlap are not distinguishable on this test set.

Usage:
    python src/ablation_report.py \\
        baseline_v1=wyniki/run_x/surya_lora_ocr800k/handlabeled/raw_predictions.csv \\
        v2_200k=wyniki/run_y/surya_lora_v2_200k/handlabeled/raw_predictions.csv \\
        noA=wyniki/run_y/surya_lora_noA_200k/handlabeled/raw_predictions.csv
"""

import argparse
import csv
import difflib
import random
import re
import sys
from collections import Counter
from pathlib import Path

LENGTH_BINS = [(1, 5), (6, 12), (13, 25), (26, 45), (46, 10_000)]


def normalize(text: str) -> str:
    # Same normalisation as benchmark/src/metrics.py
    return " ".join(str(text).strip().split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def source_of(document_id: str) -> str:
    """Collapse line ids to the notebook they came from.

    "Noga_page04_line15" -> "Noga"; the phone screenshots and the numbered
    anatomia_zmysly / dataN pages are grouped by session, matching the
    breakdown used in the first error analysis.
    """
    base = re.split(r"_(?:page|line)\d", document_id)[0]
    m = re.match(r"Zrzut ekranu 2026-(\d\d)-(\d\d)", base)
    if m:
        return f"Zrzut {m.group(1)}-{m.group(2)}"
    base = re.sub(r"_\d+$", "", base)          # anatomia_zmysly_3 -> anatomia_zmysly
    base = re.sub(r"^data\d+$", "data", base)  # data11 -> data
    return base


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = normalize(r["ground_truth"])
            pred = normalize(r["prediction"])
            rows.append({
                "id": r["document_id"],
                "source": source_of(r["document_id"]),
                "gt": gt, "pred": pred,
                "dist": levenshtein(gt, pred), "len": len(gt),
            })
    return rows


def cer(rows: list[dict]) -> float:
    total = sum(r["len"] for r in rows)
    return sum(r["dist"] for r in rows) / total if total else float("nan")


def bootstrap_ci(rows: list[dict], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(rows)
    samples = []
    for _ in range(n):
        pick = [rows[rng.randrange(k)] for _ in range(k)]
        samples.append(cer(pick))
    samples.sort()
    return samples[int(0.025 * n)], samples[int(0.975 * n)]


def substitutions(rows: list[dict], top: int = 8) -> list[tuple[str, int]]:
    counts: Counter = Counter()
    for r in rows:
        sm = difflib.SequenceMatcher(None, r["gt"], r["pred"], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace" and i2 - i1 == j2 - j1:
                for a, b in zip(r["gt"][i1:i2], r["pred"][j1:j2]):
                    counts[f"{a}->{b}"] += 1
    return counts.most_common(top)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="name=path/to/raw_predictions.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output", default=None, help="Also write the report to this .md file")
    args = ap.parse_args()
    # Windows consoles default to cp1250, which cannot print every character
    # that appears in the substitution table.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.output:
        sys.stdout = _Tee(sys.stdout, open(args.output, "w", encoding="utf-8"))

    models: dict[str, list[dict]] = {}
    for spec in args.runs:
        if "=" not in spec:
            sys.exit(f"expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        models[name] = load(Path(path))

    ids = None
    for rows in models.values():
        cur = {r["id"] for r in rows}
        ids = cur if ids is None else ids & cur
    if any(len(rows) != len(ids) for rows in models.values()):
        print(f"UWAGA: modele maja rozne zestawy linii; porownanie na wspolnych {len(ids)}\n")
        models = {n: [r for r in rows if r["id"] in ids] for n, rows in models.items()}

    print("## CER ogolem (95% bootstrap po liniach)\n")
    print("| model | n | CER | 95% CI | exact |")
    print("|---|---|---|---|---|")
    for name, rows in models.items():
        lo, hi = bootstrap_ci(rows, args.bootstrap)
        exact = sum(r["dist"] == 0 for r in rows) / len(rows)
        print(f"| {name} | {len(rows)} | {cer(rows):.1%} | {lo:.1%} - {hi:.1%} | {exact:.1%} |")

    sources = sorted({r["source"] for rows in models.values() for r in rows})
    print("\n## CER wg zrodla (autora)\n")
    print("| zrodlo | n | " + " | ".join(models) + " |")
    print("|---|---|" + "---|" * len(models))
    for src in sources:
        cells = []
        n = 0
        for rows in models.values():
            sub = [r for r in rows if r["source"] == src]
            n = len(sub)
            cells.append(f"{cer(sub):.1%}" if sub else "-")
        print(f"| {src} | {n} | " + " | ".join(cells) + " |")

    print("\n## CER wg dlugosci GT\n")
    print("| znaki | n | " + " | ".join(models) + " |")
    print("|---|---|" + "---|" * len(models))
    for lo_len, hi_len in LENGTH_BINS:
        cells = []
        n = 0
        for rows in models.values():
            sub = [r for r in rows if lo_len <= r["len"] <= hi_len]
            n = len(sub)
            cells.append(f"{cer(sub):.1%}" if sub else "-")
        label = f"{lo_len}-{hi_len}" if hi_len < 10_000 else f"{lo_len}+"
        print(f"| {label} | {n} | " + " | ".join(cells) + " |")

    print("\n## Najczestsze podmiany znakow\n")
    for name, rows in models.items():
        subs = ", ".join(f"{k} {v}" for k, v in substitutions(rows))
        print(f"- **{name}**: {subs}")


if __name__ == "__main__":
    main()
