"""Generator v3 pilot report: CER and the specific confusions v3 targets.

For every benchmark run, on the working part of handlabeled only (the frozen
test part stays unread until the final model is chosen):

    CER, and its paired change against the reference with a bootstrap CI
    counts of the confusions the glyph families aim at (ci->u, cz->w, k->h,
    s-acute->s ...) and of the opposite directions (h->k, u->ci ...), which
    a shape augmentation can push errors into

The confusions come from Marek's analyze_ngram.py (branch error-analysis):
weighted alignment that counts "cz" read as "w" as one event instead of a
substitution plus a deletion.

Usage:
    python src/v3_report.py --reference v3_base \\
        --split ../ocr-server/benchmark/splits/handlabeled_dev.txt \\
        --analyzer ../ocr-error-analysis/training/ocr/error_analysis/analyze_ngram.py \\
        v3_base=wyniki/ocr-v3-pilot_X/surya_lora_v3_base_200k/handlabeled/raw_predictions.csv \\
        v3=wyniki/ocr-v3-pilot_X/surya_lora_v3_200k/handlabeled/raw_predictions.csv \\
        --output docs/wyniki_v3.md
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_report as AR  # noqa: E402

# (reference, read as): what each glyph family is meant to fix ...
TARGETS = [
    ("ci", "u"), ("cz", "w"), ("rz", "n"), ("ci", "w"), ("oc", "a"), ("ls", "b"), ("rn", "m"),
    ("k", "lu"), ("d", "ol"),
    ("k", "h"), ("a", "o"), ("t", "ł"), ("ł", "t"), ("g", "o"), ("r", "n"), ("o", "a"),
    ("ś", "s"), ("ż", "z"), ("ą", "a"), ("ę", "e"), ("ó", "o"), ("ą", "a,"),
]
# ... and where a shape augmentation could push the error instead.
REVERSE = [("u", "ci"), ("w", "cz"), ("n", "rz"), ("b", "ls"), ("m", "rn"),
           ("h", "k"), ("o", "a"), ("ł", "t"), ("n", "r"), ("s", "ś"), ("z", "ż"), ("a", "ą")]


def load_split(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_rows(path: Path, keep: set[str]) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not keep or r["file_name"] in keep]


def events(rows: list[dict], analyzer: Path, workdir: Path) -> dict[tuple[str, str], int]:
    """Counts of every 1:1 and multi-character confusion, after Marek's alignment."""
    workdir.mkdir(parents=True, exist_ok=True)
    pred = workdir / "predictions.jsonl"
    with open(pred, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"file_name": r["file_name"], "ground_truth": r["ground_truth"],
                                "prediction": r["prediction"] or ""}, ensure_ascii=False) + "\n")
    out = workdir / "ngram"
    subprocess.run([sys.executable, str(analyzer), "--predictions", str(pred),
                    "--metrics-json", "none", "--images-dir", "none", "--out-dir", str(out)],
                   check=True, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    counts: dict[tuple[str, str], int] = {}
    with open(out / "multi_events.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[(r["src"], r["tgt"])] = int(r["count"])
    with open(out / "confusion_counts_clean.csv", encoding="utf-8") as f:
        table = list(csv.reader(f))
    cols = [c.replace("·", " ") for c in table[0][1:]]
    for row in table[1:]:
        src = row[0].replace("·", " ")
        for tgt, n in zip(cols, row[1:]):
            if int(n):
                counts[(src, tgt)] = int(n)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="name=path/to/raw_predictions.csv")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--split", type=Path, help="file names to keep (handlabeled_dev.txt)")
    ap.add_argument("--analyzer", type=Path, required=True, help="analyze_ngram.py")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    keep = load_split(args.split) if args.split else set()
    runs = dict(r.split("=", 1) for r in args.runs)
    if args.reference not in runs:
        sys.exit(f"--reference {args.reference} is not among the runs")

    tmp = Path(tempfile.mkdtemp(prefix="v3_report_"))
    rows = {name: split_rows(Path(p), keep) for name, p in runs.items()}
    models = {}
    for name, p in runs.items():
        loaded = AR.load(Path(p))
        ids = {r["file_name"].rsplit(".", 1)[0] for r in rows[name]}
        models[name] = [r for r in loaded if r["id"] in ids]
    with ThreadPoolExecutor(max_workers=min(8, len(runs))) as pool:
        found = dict(zip(runs, pool.map(lambda n: events(rows[n], args.analyzer, tmp / n), runs)))

    buf = io.StringIO()
    with redirect_stdout(buf):
        part = f"`{args.split.name}`" if args.split else "cały handlabeled"
        n_lines = len(next(iter(models.values())))
        print(f"# Pilot generatora v3: wyniki ({part}, {n_lines} linii)")
        print()
        print("| model | CER | 95% CI |")
        print("|---|---|---|")
        for name, rs in models.items():
            lo, hi = AR.bootstrap_ci(rs, n=args.bootstrap)
            print(f"| {name} | {AR.cer(rs):.2%} | {lo:.2%} .. {hi:.2%} |")
        print()
        AR.paired_vs_reference(models, args.reference, args.bootstrap)
        for title, pairs in (("Pomyłki, w które celuje v3", TARGETS),
                             ("Kierunki odwrotne (czy błąd nie przeszedł na drugą stronę)", REVERSE)):
            print(f"## {title}")
            print()
            print("Liczba zdarzeń; w nawiasie zmiana względem odniesienia.")
            print()
            names = list(runs)
            print("| pomyłka | " + " | ".join(names) + " |")
            print("|---|" + "---:|" * len(names))
            ref = found[args.reference]
            for src, tgt in pairs:
                cells = []
                for n in names:
                    v = found[n].get((src, tgt), 0)
                    d = v - ref.get((src, tgt), 0)
                    cells.append(str(v) if n == args.reference else f"{v} ({d:+d})")
                print(f"| `{src}`→`{tgt}` | " + " | ".join(cells) + " |")
            print()
    text = buf.getvalue()
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
