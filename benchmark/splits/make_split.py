"""Podział handlabeled po zeszytach: dev (analiza, wybór checkpointu) i test (zamrożony).

Zeszyty w teście wybrane tak, żeby CER na nim był bliski CER na całości
dla base Surya, v1 i v2 (różnica <0.6 pp) i żeby test miał ~1/3 znaków.
Listy plików są w repo, etykiety nie (testset jest poza gitem): skrypt
dzieli istniejący benchmark/dane/handlabeled/labels.csv.

    python benchmark/splits/make_split.py
    -> benchmark/dane/handlabeled/labels_dev.csv, labels_test.csv
"""
import csv
import re
import sys
from pathlib import Path

TEST_SOURCES = {"anatomia_zmysly_foto", "noga", "miednica"}


def source_group(file_name: str) -> str:
    """Zeszyt z nazwy wycinka: <dokument>_p<NN>[_tytuł]_l<NNN>.jpg (dataset/test/real/v1)."""
    return re.match(r"(.+?)_p\d+", file_name).group(1)


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark/dane/handlabeled/labels.csv")
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames
    parts = {"dev": [], "test": []}
    for row in rows:
        parts["test" if source_group(row["file_name"]) in TEST_SOURCES else "dev"].append(row)
    here = Path(__file__).parent
    for name, part in parts.items():
        expected = (here / f"handlabeled_{name}.txt").read_text(encoding="utf-8").split("\n")
        expected = {f for f in expected if f}
        got = {r["file_name"] for r in part}
        if got != expected:
            sys.exit(f"{name}: {len(got ^ expected)} plików niezgodnych z listą, inny labels.csv?")
        out = src.with_name(f"labels_{name}.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(part)
        print(f"{out}: {len(part)} linii")


if __name__ == "__main__":
    main()
