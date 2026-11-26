"""Wspólne argumenty raportów P1–P3: bez --predictions liczą na dataset-v1 (predykcje w $PP_CACHE)."""

import argparse
from pathlib import Path


def cli(doc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, help="jsonl: file_name, ground_truth, prediction")
    ap.add_argument("--groups", type=Path, help="CSV file_name,group[,split]")
    ap.add_argument("--out-dir", type=Path, required=True, help="katalog wyników")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    return a
