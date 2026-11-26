"""Grupy handlabeled (jednostka foldu) i generatory foldów (spec §3, §4).

Grupa = źródło notatek (przybliżenie autora; liczba autorów nieznana). Reguły dopasowania
są rozłączne; linia, która nie trafi do dokładnie jednej grupy, przerywa działanie.

    python -m evaluation.folds      # zapisuje data/dataset-v1/groups.csv
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

from evaluation.metrics import BASELINE_DIR, load_predictions

# Reguły „nazwa -> grupa” leżą w data/group_rules.json (nowy zbiór = nowe wpisy tam, bez zmian w kodzie):
#   file_name      dopasowanie po nazwie wycinka,
#   original_name  dla stron, których nazwa nie niesie źródła (dokument "new"): po nazwie pliku z manifest.csv.
GROUP_RULES_JSON = Path(__file__).resolve().parent.parent / "data" / "group_rules.json"


def load_rules(path: Path = GROUP_RULES_JSON) -> dict[str, list[tuple[str, re.Pattern]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: [(g, re.compile(rx)) for g, rx in raw[k]] for k in ("file_name", "original_name")}


def _match(name: str, rules: Sequence[tuple[str, re.Pattern]]) -> str:
    hits = [g for g, rx in rules if rx.match(name)]
    if len(hits) != 1:
        raise ValueError(f"{name!r} matches {hits or 'no group'} (dopisz regułę w data/group_rules.json)")
    return hits[0]


def source_group(original_name: str, rules: dict | None = None) -> str:
    return _match(original_name, (rules or load_rules())["original_name"])


# Test odporności (spec §3): łączymy źródła, które mogą mieć wspólnego autora.
COARSE = {
    "G6_dataN": "C_zrzuty_dataN",
    "G7_zrzuty_0507": "C_zrzuty_dataN",
    "G8_zrzuty_0512": "C_zrzuty_dataN",
}

GROUPS_CSV = BASELINE_DIR / "groups.csv"


def assign_group(file_name: str, rules: dict | None = None) -> str:
    return _match(file_name, (rules or load_rules())["file_name"])


def coarse(group: str) -> str:
    return COARSE.get(group, group)


def line_groups(rows: Sequence[dict], split: str = "fine", group_map: dict[str, str] | None = None) -> list[str]:
    """Grupa linii; `group_map` (file_name -> grupa) ma pierwszeństwo przed GROUP_RULES (np. strony `new_*` z v1.2)."""
    gm = group_map or {}
    gs = [gm.get(r["file_name"]) or assign_group(r["file_name"]) for r in rows]
    if split == "coarse":
        gs = [coarse(g) for g in gs]
    elif split != "fine":
        raise ValueError(split)
    return gs


def logo(groups: Sequence[str]) -> Iterator[tuple[str, list[int], list[int]]]:
    """Leave-one-group-out: (grupa testowa, indeksy treningowe, indeksy testowe)."""
    for g in sorted(set(groups)):
        test = [i for i, x in enumerate(groups) if x == g]
        train = [i for i, x in enumerate(groups) if x != g]
        yield g, train, test


def nested_logo(groups: Sequence[str]) -> Iterator[tuple[str, list[int], list[int], list[tuple[str, list[int], list[int]]]]]:
    """Zewnętrzne LOGO + wewnętrzne LOGO na grupach treningowych (indeksy globalne)."""
    for g, train, test in logo(groups):
        inner = []
        for h in sorted({groups[i] for i in train}):
            itest = [i for i in train if groups[i] == h]
            itrain = [i for i in train if groups[i] != h]
            inner.append((h, itrain, itest))
        yield g, train, test, inner


def write_groups_csv(path: Path = GROUPS_CSV) -> Counter:
    rows = load_predictions()
    fine = line_groups(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "group", "group_coarse"])
        for r, g in zip(rows, fine):
            w.writerow([r["file_name"], g, coarse(g)])
    return Counter(fine)


def read_group_map(path: Path) -> dict[str, str]:
    """CSV z kolumnami file_name, group (inne kolumny pomijane)."""
    with open(path, encoding="utf-8") as f:
        return {r["file_name"]: r["group"] for r in csv.DictReader(f)}


def read_split_map(path: Path) -> dict[str, str] | None:
    """file_name -> "dev" | "test" z kolumny `split`; None, gdy CSV jej nie ma."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "split" not in rows[0]:
        return None
    return {r["file_name"]: r["split"] for r in rows}


def check_splits(groups: Sequence[str], splits: Sequence[str]) -> None:
    """Grupa musi w całości leżeć po jednej stronie podziału, inaczej test przecieka."""
    bad = sorted({g for g in groups if len({s for h, s in zip(groups, splits) if h == g}) > 1})
    if bad:
        raise ValueError(f"grupy rozbite między dev i test: {bad}")
    if set(splits) - {"dev", "test"}:
        raise ValueError(f"nieznane wartości split: {sorted(set(splits) - {'dev', 'test'})}")


def read_groups_csv(path: Path = GROUPS_CSV) -> dict[str, tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return {r["file_name"]: (r["group"], r["group_coarse"]) for r in csv.DictReader(f)}


if __name__ == "__main__":
    c = write_groups_csv()
    for g, n in sorted(c.items()):
        print(f"{n:5d}  {g}")
    print(f"{sum(c.values()):5d}  razem -> {GROUPS_CSV}")
