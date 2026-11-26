"""Składa SUMMARY.md z liczb w katalogu wyników jednego przebiegu (rebuild.py). Brakujące części pomija.

    python -m evaluation.report ../results/NAZWA
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pp(x) -> str:
    return f"{float(x) * 100:+.2f}"


def pct(x) -> str:
    return f"{float(x) * 100:.2f}%"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(str(c) for c in r) + " |" for r in rows), ""]


def corr_row(name: str, r: dict, ci_key: str = "lines") -> list:
    return [name, f"{float(r['base_cer']):.4f} → {float(r['cer']):.4f}", pp(r["d_cer"]),
            f"[{pp(r[f'ci_{ci_key}_lo'])}, {pp(r[f'ci_{ci_key}_hi'])}]",
            f"{float(r['base_wer']):.4f} → {float(r['wer']):.4f}",
            f"{r['fixed']} / {r['broken']} / {r['changed_wrong']}", pct(r["harm"])]


HEAD = ["wariant", "CER", "ΔCER p.p.", "95% CI (linie)", "WER", "fixed / broken / changed-wrong", "harm"]


def build(d: Path) -> str:
    out = [f"# Wyniki przebiegu `{d.name}`", "",
           "Wygenerowane przez `evaluation/report.py` z plików w tym katalogu. Nie edytować ręcznie.", ""]
    raw = read_csv(d / "raw_ocr.csv")
    if raw:
        models = list(dict.fromkeys(r["model"] for r in raw))
        cell = {(r["model"], r["subset"]): r for r in raw}
        subsets = [s for s in ("all", "dev", "test") if (models[0], s) in cell]
        out += ["## Surowy OCR (bez korekty), CER / WER", ""]
        out += table(["model", *subsets], [[m, *(f"{float(cell[m, s]['cer']):.4f} / {float(cell[m, s]['wer']):.4f}"
                                                 for s in subsets)] for m in models])
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else {}
    main_model = cfg.get("model", "")

    logo_rows, test_rows = [], []
    for name, label in (("rescore_surya.json", "top-5 + ocena Suryi (docelowy)"),
                        ("rescore_w0.json", "top-5 bez Suryi (statyczny, w paczce)")):
        p = d / name
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            logo_rows.append(corr_row(label, r["logo"]))
            test_rows.append(corr_row(label, r["test"]))
    for r in read_csv(d / "p4_summary.csv"):
        if r["variant"] == main_model:
            logo_rows.append(corr_row("pełna lista kandydatów, bez Suryi", r))
    p = d / f"holdout_{main_model}.json"
    if p.exists() and "test" in (h := json.loads(p.read_text(encoding="utf-8"))):
        test_rows.append(corr_row("pełna lista kandydatów, bez Suryi", h["test"]))
    if logo_rows:
        out += [f"## Korektor na `{main_model}`: LOGO (liczba główna)", ""] + table(HEAD, logo_rows)
    if test_rows:
        out += ["## Ten sam model: protokół dev → test", ""] + table(HEAD, test_rows)

    others = [r for r in read_csv(d / "p4_summary.csv") if r["variant"] != main_model]
    if others:
        out += ["## Inne modele OCR, korektor z pełną listą kandydatów (LOGO)", ""]
        out += table(HEAD, [corr_row(r["variant"], r) for r in others])

    abl = read_csv(d / "ablations" / "p4_summary.csv")
    if abl:
        out += ["## Ablacje (LOGO, pełna lista kandydatów; porównywać z wierszem `main`)", "",
                "Różnice rzędu 0.1 p.p. to szum strojenia.", ""]
        out += table(HEAD, [corr_row(r["variant"], r) for r in abl])
    return "\n".join(out)


def main() -> None:
    d = Path(sys.argv[1])
    (d / "SUMMARY.md").write_text(build(d), encoding="utf-8")
    print(d / "SUMMARY.md")


if __name__ == "__main__":
    main()
