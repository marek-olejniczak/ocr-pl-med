"""Grupy handlabeled (jednostka foldu) i generatory foldów (spec §3, §4).

Grupa = źródło notatek (przybliżenie autora; liczba autorów nieznana). Reguły dopasowania
są rozłączne; linia, która nie trafi do dokładnie jednej grupy, przerywa działanie.

    PYTHONPATH=. python -m evaluation.folds      # zapisuje data/baseline/groups.csv
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

from evaluation.metrics import BASELINE_DIR, load_predictions

GROUP_RULES: list[tuple[str, re.Pattern]] = [
    ("G1_noga", re.compile(r"^noga_p\d+")),
    ("G2_anatomia_zmysly", re.compile(r"^anatomia_zmysly_foto_p\d+")),
    ("G3_serce", re.compile(r"^serce_p\d+")),
    ("G4_unaczynienie", re.compile(r"^miednica_p\d+")),
    ("G5_elementy_topo", re.compile(r"^elementy_topograficzne_p\d+")),
    ("G6_dataN", re.compile(r"^inne_p\d+")),
    ("G7_zrzuty_0507", re.compile(r"^kosci_p\d+")),
    ("G8_zrzuty_0512", re.compile(r"^miesnie_p\d+")),
]

# test/real/v1.2: strony `new_<hash>` (dokument "new") nie niosą źródła w nazwie; grupę daje plik źródłowy
# z manifest.csv (original_name). Mapę linia -> grupa zapisuje evaluation.import_bench do data/v12/groups.csv.
NEW_SOURCE_RULES: list[tuple[str, re.Pattern]] = [
    ("G9_karta", re.compile(r"^(Kopia KARTA PACJENTA \d|Recepta okulistyczna)")),
    ("G10_nerw_twarzowy", re.compile(r"^Nerw Twarzowy")),
    ("G11_zyly_glebokie", re.compile(r"^Żyły Głębokie")),
]


def source_group(original_name: str) -> str:
    hits = [g for g, rx in NEW_SOURCE_RULES if rx.match(original_name)]
    if len(hits) != 1:
        raise ValueError(f"{original_name!r} matches {hits or 'no group'}")
    return hits[0]


# Test odporności (spec §3): łączymy źródła, które mogą mieć wspólnego autora.
COARSE = {
    "G6_dataN": "C_zrzuty_dataN",
    "G7_zrzuty_0507": "C_zrzuty_dataN",
    "G8_zrzuty_0512": "C_zrzuty_dataN",
}

GROUPS_CSV = BASELINE_DIR / "groups.csv"


def assign_group(file_name: str) -> str:
    hits = [g for g, rx in GROUP_RULES if rx.match(file_name)]
    if len(hits) != 1:
        raise ValueError(f"{file_name!r} matches {hits or 'no group'}")
    return hits[0]


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


def read_groups_csv(path: Path = GROUPS_CSV) -> dict[str, tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return {r["file_name"]: (r["group"], r["group_coarse"]) for r in csv.DictReader(f)}


if __name__ == "__main__":
    c = write_groups_csv()
    for g, n in sorted(c.items()):
        print(f"{n:5d}  {g}")
    print(f"{sum(c.values()):5d}  razem -> {GROUPS_CSV}")
