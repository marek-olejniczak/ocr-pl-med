"""Składa tabelę wyników (spec §7) z results/p4_summary.csv, p2_option0.csv, oracle.csv
i results/p4_*_per_group.json → results/TABLE.md (bez tekstu linii).

    PYTHONPATH=. python -m evaluation.make_table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

LABELS = [
    # (variant, split, opis)
    ("option1_harm03_gtfix", "fine", "**opcja 1 (główny, harm ≤ 0.3%), GT po poprawce 39 tokenów; baseline 0.12727**"),
    ("option1_harm03", "fine", "**opcja 1 (główny, harm ≤ 0.3%)**, GT przed poprawką"),
    ("option1", "fine", "opcja 1, harm ≤ 0.5%"),
    ("abl_no_multi", "fine", "ablacja: bez jednostek wieloznakowych"),
    ("abl_no_protect", "fine", "ablacja: bez klas chronionych P2–P8"),
    ("abl_option2_only", "fine", "opcja 2 osobno (tylko diakrytyki)"),
    ("abl_sjp", "fine", "ablacja: + sjp odmiany"),
    ("abl_forward", "fine", "ablacja: forward zamiast Viterbi"),
    ("option1_harm03", "coarse", "test odporności: podział gruby (6 grup), harm ≤ 0.3%"),
    ("option1", "coarse", "podział gruby, harm ≤ 0.5%"),
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
    base = next(iter(summ.values()))
    head = ("| wariant | CER | CER pomocn. | WER | ΔCER p.p. | Δ wzgl. | 95% CI grupy | 95% CI linie | p (znaki) "
            "| fixed / broken / ch-wrong / net | prec. | recall | harm |")
    lines = ["# Tabela wyników (generowana: `python -m evaluation.make_table`)", "",
             "Predykcje out-of-fold, zagnieżdżone LOGO, 2160 linii handlabeled. ΔCER względem baseline.", "",
             head, "|" + "---|" * 13,
             f"| baseline (Surya v2_200k) | {float(base['base_cer']):.5f} | {float(base['base_cer_aux']):.5f} "
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
