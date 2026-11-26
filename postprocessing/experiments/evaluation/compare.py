"""Porównanie baseline vs po korekcie: CER/WER, tabela tokenów, istotność (spec §7, ticket 07).

Tabela tokenów jest liczona po tokenach GT. Każdy token GT dostaje token z wyrównania
Levenshteina na słowach (GT↔baseline i osobno GT↔korekta); brak odpowiednika = None.
  fixed          błędny przed, poprawny po
  broken         poprawny przed, błędny po
  changed_wrong  błędny przed i po, ale korekta zmieniła odpowiednik
  unchanged_wrong / unchanged_correct
precision = fixed / zmienione (odpowiednik przed ≠ po), recall = fixed / błędne przed,
harm = broken / poprawne przed.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from rapidfuzz.distance import Levenshtein

from evaluation import sig
from evaluation.metrics import line_char_errors, normalize_aux, normalize_ws, score


def align_to_gt(gt_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> list[str | None]:
    """Dla każdego tokenu GT: odpowiadający token hipotezy (lub None)."""
    out: list[str | None] = [None] * len(gt_tokens)
    for op in Levenshtein.opcodes(list(gt_tokens), list(hyp_tokens)):
        if op.tag in ("equal", "replace"):
            for k in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                out[op.src_start + k] = hyp_tokens[op.dest_start + k]
    return out


def token_table(gts: Sequence[str], base: Sequence[str], corr: Sequence[str]) -> dict:
    c = Counter()
    for gt, b, k in zip(gts, base, corr, strict=True):
        g = normalize_ws(gt).split()
        ab = align_to_gt(g, normalize_ws(b).split())
        ak = align_to_gt(g, normalize_ws(k).split())
        for t, x, y in zip(g, ab, ak):
            ok_b, ok_k = x == t, y == t
            if ok_b and ok_k:
                c["unchanged_correct"] += 1
            elif ok_b:
                c["broken"] += 1
            elif ok_k:
                c["fixed"] += 1
            elif x != y:
                c["changed_wrong"] += 1
            else:
                c["unchanged_wrong"] += 1
            c["changed"] += x != y
    correct_before = c["unchanged_correct"] + c["broken"]
    wrong_before = c["fixed"] + c["changed_wrong"] + c["unchanged_wrong"]
    out = {k: c[k] for k in ("fixed", "broken", "changed_wrong", "unchanged_wrong", "unchanged_correct", "changed")}
    out["net"] = c["fixed"] - c["broken"]
    out["precision"] = c["fixed"] / c["changed"] if c["changed"] else float("nan")
    out["recall"] = c["fixed"] / wrong_before if wrong_before else float("nan")
    out["harm"] = c["broken"] / correct_before if correct_before else float("nan")
    return out


def compare(gts: Sequence[str], base: Sequence[str], corr: Sequence[str], groups: Sequence[str],
            B_groups: int = 10000, B_lines: int = 2000, seed: int = 0) -> dict:
    sb, sk = score(gts, base), score(gts, corr)
    ab, ak = score(gts, base, normalize_aux), score(gts, corr, normalize_aux)
    ea = [line_char_errors(g, p)[0] for g, p in zip(gts, base)]
    eb = [line_char_errors(g, p)[0] for g, p in zip(gts, corr)]
    n = [line_char_errors(g, g)[1] for g in gts]
    dg, cig, pg = sig.paired_bootstrap(ea, eb, n, list(groups), B=B_groups, seed=seed)
    _, cil, pl = sig.paired_bootstrap(ea, eb, n, list(range(len(gts))), B=B_lines, seed=seed)
    wins, losses, p_sign = sig.sign_test(ea, eb)
    row = {
        "cer": sk.cer, "cer_aux": ak.cer, "wer": sk.wer,
        "d_cer": sk.cer - sb.cer, "d_cer_rel": (sk.cer - sb.cer) / sb.cer if sb.cer else float("nan"),
        "ci_groups_lo": cig[0], "ci_groups_hi": cig[1], "p_boot_groups": pg,
        "ci_lines_lo": cil[0], "ci_lines_hi": cil[1], "p_boot_lines": pl,
        "lines_better": wins, "lines_worse": losses, "p_sign": p_sign,
        "base_cer": sb.cer, "base_cer_aux": ab.cer, "base_wer": sb.wer,
    }
    row.update(token_table(gts, base, corr))
    return row


def per_group_cer(gts: Sequence[str], preds: Sequence[str], groups: Sequence[str]) -> dict[str, float]:
    acc: dict[str, list[int]] = {}
    for g, p, gr in zip(gts, preds, groups):
        e, n = line_char_errors(g, p)
        x = acc.setdefault(gr, [0, 0])
        x[0] += e
        x[1] += n
    return {k: v[0] / v[1] for k, v in sorted(acc.items())}
