"""Składa tabelę wyników (spec §7) z results/p4_summary.csv, p2_option0.csv, oracle.csv
i results/p4_*_per_group.json → results/TABLE.md (bez tekstu linii).

    python -m evaluation.make_table      # tylko dataset-v1; dla nowych zbiorów evaluation/report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dataset-v1"  # tylko archiwum dataset-v1

LABELS = [
    # (variant, split, opis)
    ("v1_ta", "fine", "**główny: dataset-v1, lista TA en,la,pl, progi strojone bez listy**"),
    ("v1_base", "fine", "dataset-v1, bez listy"),
    ("v1_lapl", "fine", "dataset-v1, lista TA la,pl"),
    ("v1_ta_tuned", "fine", "ablacja dataset-v1: lista także w strojeniu (luźniejsze progi)"),
    ("v1_no_cap", "fine", "ablacja dataset-v1: bez limitu udziału grupy"),
    ("v1_no_multi", "fine", "ablacja dataset-v1: tylko pomyłki 1:1, bez jednostek wieloznakowych"),
    ("v1_uniform", "fine", "ablacja dataset-v1: jednostajny Levenshtein zamiast wyuczonych pomyłek"),
    ("option1_harm03_gtfix", "fine", "hist.: opcja 1, predykcje z 1.10, GT po poprawce; baseline 0.12727"),
    ("option1_harm03", "fine", "hist.: opcja 1, predykcje z 1.10, GT przed poprawką; baseline 0.12769"),
    ("option1", "fine", "hist.: harm ≤ 0.5%"),
    ("abl_no_multi", "fine", "hist. ablacja: bez jednostek wieloznakowych"),
    ("abl_no_protect", "fine", "hist. ablacja: bez klas chronionych P2–P8"),
    ("abl_option2_only", "fine", "hist.: opcja 2 osobno (tylko diakrytyki)"),
    ("abl_sjp", "fine", "hist. ablacja: + sjp odmiany"),
    ("abl_forward", "fine", "hist. ablacja: forward zamiast Viterbi"),
    ("option1_harm03", "coarse", "hist.: test odporności, podział gruby (6 grup), harm ≤ 0.3%"),
    ("option1", "coarse", "hist.: podział gruby, harm ≤ 0.5%"),
]
ORACLES = [
    ("O1_oov->GT_inlex_lev<=2", "oracle: OOV → GT (GT w słowniku, lev ≤ 2)"),
    ("O1_oov->GT_inlex_lev<=3", "oracle: OOV → GT (GT w słowniku, lev ≤ 3)"),
    ("O2_diakrytyki_wszystkie->GT", "oracle: diakrytyki"),
]


def pp(x: str, scale: float = 100, nd: int = 2) -> str:
    return f"{float(x) * scale:+.{nd}f}"


def main() -> None:
    summ = {}
    with open(RESULTS / "p4_summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            summ[(r["variant"], r["split"])] = r  # ostatni wpis wygrywa
    base = summ[("v1_ta", "fine")]
    head = ("| wariant | CER | CER pomocn. | WER | ΔCER p.p. | Δ wzgl. | 95% CI grupy | 95% CI linie | p (znaki) "
            "| fixed / broken / ch-wrong / net | prec. | recall | harm |")
    lines = ["# Tabela wyników (generowana: `python -m evaluation.make_table`)", "",
             "Predykcje out-of-fold, zagnieżdżone LOGO, 2160 linii handlabeled. ΔCER względem baseline swojego wiersza. "
             "Wiersze „hist.” dotyczą predykcji z 2026-10-01 i starszego GT; oracle i opcja 0 także.", "",
             head, "|" + "---|" * 13,
             f"| baseline (Surya v2_200k, dataset-v1) | {float(base['base_cer']):.5f} | {float(base['base_cer_aux']):.5f} "
             f"| {float(base['base_wer']):.4f} | — | — | — | — | — | — | — | — | — |"]
    o0 = RESULTS / "p2_option0.csv"
    if o0.exists():
        r = next(csv.DictReader(open(o0, encoding="utf-8")))
        lines.append(f"| opcja 0 (higiena) | {float(r['cer']):.5f} | {float(r['cer_aux']):.5f} | {float(r['wer']):.4f} "
                     f"| {pp(r['d_cer'])} | — | — | — | — | 0 / 0 / 0 / 0 | — | — | 0 |")
    for key in LABELS:
        r = summ.get(key[:2])
        if not r:
            lines.append(f"| {key[2]} | *brak przebiegu* |" + " |" * 11)
            continue
        lines.append(
            f"| {key[2]} | {float(r['cer']):.5f} | {float(r['cer_aux']):.5f} | {float(r['wer']):.4f} | {pp(r['d_cer'])} "
            f"| {pp(r['d_cer_rel'], 100, 1)}% | [{pp(r['ci_groups_lo'])}, {pp(r['ci_groups_hi'])}] "
            f"| [{pp(r['ci_lines_lo'])}, {pp(r['ci_lines_hi'])}] "
            f"| {'<1e-6' if float(r['p_sign']) < 1e-6 else format(float(r['p_sign']), '.1e')} "
            f"| {r['fixed']} / {r['broken']} / {r['changed_wrong']} / {r['net']} | {float(r['precision']):.3f} "
            f"| {float(r['recall']):.3f} | {float(r['harm']) * 100:.2f}% |")
    with open(RESULTS / "oracle.csv", encoding="utf-8") as f:
        orc = {r["variant"]: r for r in csv.DictReader(f)}
    for v, label in ORACLES:
        r = orc[v]
        lines.append(f"| {label} | {float(r['cer']):.5f} | {float(r['cer_aux']):.5f} | {float(r['wer']):.4f} "
                     f"| {pp(r['d_cer'])} | {pp(r['d_cer_rel'], 100, 1)}% | — | — | — | {r['fixed']} / 0 / 0 / {r['net']} "
                     f"| 1.000 | {float(r['recall']):.3f} | 0 |")

    lines += ["", "## Aneks: CER per grupa", ""]
    pg = {}
    for key in LABELS:
        p = RESULTS / f"p4_{key[0]}_{key[1]}_per_group.json"
        if p.exists():
            pg[key] = json.loads(p.read_text())
    for split in ("fine", "coarse", "gtfix"):
        if split == "gtfix":
            keys = [k for k in pg if k[0].endswith("_gtfix")]
        else:
            keys = [k for k in pg if k[1] == split and not k[0].endswith("_gtfix")]
        if not keys:
            continue
        groups = list(pg[keys[0]]["baseline"])
        lines += [f"### podział {split}", "", "| wariant | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups),
                  "| baseline | " + " | ".join(f"{pg[keys[0]]['baseline'][g]:.4f}" for g in groups) + " |"]
        for k in keys:
            d = pg[k][k[0]]
            lines.append(f"| {k[2]} | " + " | ".join(f"{d[g]:.4f}" for g in groups) + " |")
        lines.append("")
    (RESULTS / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
