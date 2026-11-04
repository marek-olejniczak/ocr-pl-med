#!/usr/bin/env python
"""Analiza pomyłek wieloznakowych (1:2, 2:1, 2:2) dla OCR (Surya) na handlabeled.

Uzupełnienie `analyze.py`. Zwykły Levenshtein (sub=del=ins=1) rozbija zdarzenie
typu „cz” → „w” na substytucję + usunięcie, co zanieczyszcza macierz 1:1
(np. `c→w`, `z→n`). Tutaj wyrównujemy z operacjami wieloznakowymi i uczymy
ich koszty z danych.

Metoda:
  * normalizacja 1:1 z analyze.py (eval_cer + NFC),
  * iteracja 0: zwykły Levenshtein; kandydaci na zdarzenia wieloznakowe =
    sąsiadujące pary edycji (sub+ins -> 1:2, sub+del -> 2:1, sub+sub -> 2:2),
  * iteracje 1..K (hard-EM / Viterbi): koszt operacji = -log P(pred | ref-unit),
      - 1:1, usunięcie:  (N + α) / (wystąpienia znaku w GT + α·(V+1)),
      - wstawienie:      (N + α) / (liczba slotów + α·V),
      - wieloznakowe:    N / wystąpienia źródła (znak / bigram) w GT, + kara λ;
    tylko kandydaci z N ≥ --min-inventory są dopuszczeni; wyrównanie, zliczenie,
    nowe koszty, aż do zbieżności,
  * test lift: P(zdarzenia) / P(najlepszej dekompozycji na niezależne 1:1/indel);
    raportujemy zdarzenia z N ≥ --min-count i lift ≥ --min-lift,
  * zdarzenia trywialne (zawierające znak źródłowy po stronie predykcji, np.
    „rz”→„r” = zwykłe usunięcie) oraz obejmujące spację są wykluczone,
  * rozbicie na źródła (zeszyt/dokument, proxy autora) — czy pomyłka jest
    ogólna, czy specyficzna dla jednego charakteru pisma,
  * oczyszczona macierz 1:1 = substytucje, które zostały po wyjęciu zdarzeń
    wieloznakowych.

Artefakty (--out-dir, domyślnie out_ngram/):
    multi_events.csv           wszystkie zdarzenia wieloznakowe (N, P, lift, źródła)
    sub_before_after.csv       substytucje 1:1: Levenshtein vs po oczyszczeniu
    unit_error_stats.csv       per znak (po oczyszczeniu): wystąpienia, błędy, udział
    confusion_counts_clean.csv oczyszczona macierz 1:1
    heatmap_clean.png / heatmap_clean_top.png
    examples/*.png             wycinki dla top zdarzeń
    RAPORT.md

Uruchomienie (z katalogu narzędzia, po analyze.py / infer.sh):
    python analyze_ngram.py      # data/predictions.jsonl -> out_ngram/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as A

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]

# Zdarzenia o podłożu ortograficznym/fonetycznym (nie optycznym) — wskazują
# raczej na „język” dekodera niż na wizję. Oba kierunki.
_ORTHO = [
    ("rz", "ż"), ("ch", "h"), ("ó", "u"), ("ci", "ć"), ("si", "ś"),
    ("ni", "ń"), ("zi", "ź"), ("om", "ą"), ("on", "ą"), ("em", "ę"), ("en", "ę"),
]
ORTHO = {p for a, b in _ORTHO for p in ((a, b), (b, a))}


def kind_of(src: str, tgt: str) -> str:
    return f"{len(src)}:{len(tgt)}"


def source_group(file_name: str) -> str:
    """Zeszyt/dokument źródłowy (proxy autora) z nazwy pliku linii."""
    stem = re.sub(r"_line\d+\.\w+$", "", file_name)
    stem = re.sub(r"_page\d+$", "", stem)
    stem = re.sub(r"^(Zrzut ekranu \d{4}-\d{2}-\d{2}).*$", r"\1", stem)
    stem = re.sub(r"^data\d+$", "data", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def word_at(text: str, idx: int) -> str:
    """Słowo (ciąg bez spacji) obejmujące pozycję idx."""
    lo = text.rfind(" ", 0, idx) + 1
    hi = text.find(" ", idx)
    return text[lo:hi if hi >= 0 else len(text)].strip(".,:;()-–>")


def is_valid_multi(src: str, tgt: str) -> bool:
    if (len(src), len(tgt)) not in {(1, 2), (2, 1), (2, 2)}:
        return False
    if " " in src or " " in tgt:
        return False
    if len(src) == 2 and len(tgt) == 2:
        # ab->cd: każda pozycja musi być błędem (inaczej to pojedyncza substytucja)
        return src[0] != tgt[0] and src[1] != tgt[1]
    if len(src) == 1:
        return src not in tgt          # a->ab to zwykłe wstawienie
    return tgt not in src              # ab->a to zwykłe usunięcie


# --------------------------------------------------------------------------
# Model kosztów
# --------------------------------------------------------------------------


class CostModel:
    """Koszty operacji = -log P. `unit=True` -> zwykły Levenshtein (iteracja 0)."""

    def __init__(self, occ1, occ2, n_slots, vocab, n1=None, n_ins=None,
                 multi=None, alpha=0.1, lam=0.5, unit=False):
        self.unit = unit
        self.occ1, self.occ2 = occ1, occ2
        self.n1 = n1 or Counter()
        self.n_ins = n_ins or Counter()
        self.alpha, self.lam = alpha, lam
        self.V = len(vocab)
        self.n_slots = n_slots
        self.multi: dict[tuple[str, str], float] = {}
        if multi and not unit:
            for (src, tgt), n in multi.items():
                occ = occ1[src] if len(src) == 1 else occ2[src]
                if occ and n:
                    self.multi[(src, tgt)] = -math.log(min(n / occ, 1.0)) + lam
        self.multi_by_shape = defaultdict(dict)
        for (src, tgt), c in self.multi.items():
            self.multi_by_shape[(len(src), len(tgt))][(src, tgt)] = c
        self._c1: dict[tuple[str, str], float] = {}
        self._ci: dict[str, float] = {}

    def p1(self, a: str, b: str) -> float:
        """P(b | a) dla 1:1 / usunięcia (b == '')."""
        return (self.n1.get((a, b), 0) + self.alpha) / (self.occ1.get(a, 0) + self.alpha * (self.V + 1))

    def p_ins(self, b: str) -> float:
        return (self.n_ins.get(b, 0) + self.alpha) / (self.n_slots + self.alpha * self.V)

    def c1(self, a: str, b: str) -> float:
        if self.unit:
            return 0.0 if a == b else 1.0
        key = (a, b)
        v = self._c1.get(key)
        if v is None:
            v = self._c1[key] = -math.log(self.p1(a, b))
        return v

    def cins(self, b: str) -> float:
        if self.unit:
            return 1.0
        v = self._ci.get(b)
        if v is None:
            v = self._ci[b] = -math.log(self.p_ins(b))
        return v


def align_weighted(ref: str, pred: str, cm: CostModel):
    """Viterbi z operacjami 1:1, del, ins, 1:2, 2:1, 2:2.

    Zwraca listę (src, tgt, i_ref, j_pred) w kolejności; src/tgt to napisy
    (puste przy indelach), i/j = indeks początku w ref/pred.
    """
    n, m = len(ref), len(pred)
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bk = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    m12 = cm.multi_by_shape.get((1, 2), {})
    m21 = cm.multi_by_shape.get((2, 1), {})
    m22 = cm.multi_by_shape.get((2, 2), {})
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = INF, None
            if i and j:
                v = dp[i - 1][j - 1] + cm.c1(ref[i - 1], pred[j - 1])
                if v < best:
                    best, arg = v, (1, 1)
            if i:
                v = dp[i - 1][j] + cm.c1(ref[i - 1], "")
                if v < best:
                    best, arg = v, (1, 0)
            if j:
                v = dp[i][j - 1] + cm.cins(pred[j - 1])
                if v < best:
                    best, arg = v, (0, 1)
            if i >= 1 and j >= 2 and m12:
                c = m12.get((ref[i - 1], pred[j - 2:j]))
                if c is not None and dp[i - 1][j - 2] + c < best:
                    best, arg = dp[i - 1][j - 2] + c, (1, 2)
            if i >= 2 and j >= 1 and m21:
                c = m21.get((ref[i - 2:i], pred[j - 1]))
                if c is not None and dp[i - 2][j - 1] + c < best:
                    best, arg = dp[i - 2][j - 1] + c, (2, 1)
            if i >= 2 and j >= 2 and m22:
                c = m22.get((ref[i - 2:i], pred[j - 2:j]))
                if c is not None and dp[i - 2][j - 2] + c < best:
                    best, arg = dp[i - 2][j - 2] + c, (2, 2)
            dp[i][j], bk[i][j] = best, arg
    ops = []
    i, j = n, m
    while i or j:
        di, dj = bk[i][j]
        ops.append((ref[i - di:i], pred[j - dj:j], i - di, j - dj))
        i, j = i - di, j - dj
    ops.reverse()
    return ops, dp[n][m]


def merge_candidates(ops) -> Counter:
    """Sąsiadujące pary jednoznakowych edycji -> kandydaci wieloznakowi."""
    cand: Counter = Counter()
    k = 0
    while k < len(ops) - 1:
        s1, t1, _, _ = ops[k]
        s2, t2, _, _ = ops[k + 1]
        e1 = s1 != t1 and len(s1) <= 1 and len(t1) <= 1
        e2 = s2 != t2 and len(s2) <= 1 and len(t2) <= 1
        if e1 and e2:
            src, tgt = s1 + s2, t1 + t2
            if is_valid_multi(src, tgt):
                cand[(src, tgt)] += 1
                k += 2
                continue
        k += 1
    return cand


# --------------------------------------------------------------------------
# Pętla EM
# --------------------------------------------------------------------------


def run_em(lines, occ1, occ2, n_slots, vocab, args):
    cm = CostModel(occ1, occ2, n_slots, vocab, unit=True)
    history = []
    prev_total = None
    for it in range(args.iterations + 1):
        n1: Counter = Counter()
        n_ins: Counter = Counter()
        multi: Counter = Counter()
        cand: Counter = Counter()
        aligned = []
        total = 0.0
        for fn, ref, pred in lines:
            ops, cost = align_weighted(ref, pred, cm)
            total += cost
            aligned.append(ops)
            for src, tgt, _, _ in ops:
                if len(src) == 1 and len(tgt) <= 1:
                    n1[(src, tgt)] += 1
                elif not src:
                    n_ins[tgt] += 1
                else:
                    multi[(src, tgt)] += 1
            cand.update(merge_candidates(ops))
        n_multi = sum(multi.values())
        history.append({"iter": it, "total_cost": total, "multi_events": n_multi,
                        "inventory": len(cm.multi)})
        print(f"[ngram] iter {it}: koszt={total:.1f}  zdarzenia wieloznakowe={n_multi}  "
              f"inwentarz={len(cm.multi)}", flush=True)
        if it == args.iterations:
            break
        if (prev_total is not None and it >= 2
                and abs(prev_total - total) / max(prev_total, 1e-9) < 1e-4):
            break
        prev_total = total
        pool = multi + cand
        inventory = {k: v for k, v in pool.items() if v >= args.min_inventory}
        cm = CostModel(occ1, occ2, n_slots, vocab, n1=n1, n_ins=n_ins,
                       multi=inventory, alpha=args.alpha, lam=args.lam)
    return cm, aligned, n1, n_ins, multi, history


# --------------------------------------------------------------------------
# Raportowanie
# --------------------------------------------------------------------------


def lift_of(src, tgt, p_ev, cm: CostModel) -> float:
    if len(src) == 2 and len(tgt) == 1:
        a, b = src
        indep = max(cm.p1(a, tgt) * cm.p1(b, ""), cm.p1(a, "") * cm.p1(b, tgt))
    elif len(src) == 1 and len(tgt) == 2:
        x, y = tgt
        indep = max(cm.p1(src, x) * cm.p_ins(y), cm.p_ins(x) * cm.p1(src, y))
    else:
        indep = cm.p1(src[0], tgt[0]) * cm.p1(src[1], tgt[1])
    return p_ev / indep if indep else float("inf")


def _window_span(text: str, idx: int, length: int, half: int = 26) -> str:
    lo, hi = max(0, idx - half), min(len(text), idx + length + half)
    left = ("…" if lo > 0 else "") + text[lo:idx]
    right = text[idx + length:hi] + ("…" if hi < len(text) else "")
    return f"{left}[{text[idx:idx + length]}]{right}"


def plot_multi_examples(out_dir, events, lines, aligned, images_dir: Path, per_event: int):
    from PIL import Image

    ex_dir = out_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    index = ["# Przykłady zdarzeń wieloznakowych", ""]
    for rank, ev in enumerate(events, start=1):
        src, tgt = ev["src"], ev["tgt"]
        hits = []
        for (fn, ref, pred), ops in zip(lines, aligned):
            for s, t, i, j in ops:
                if s == src and t == tgt:
                    hits.append((fn, ref, pred, i, j))
                    break
            if len(hits) >= per_event:
                break
        if not hits:
            continue
        nrows = int(np.ceil(len(hits) / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (fn, ref, pred, i, j) in zip(axes, hits):
            try:
                with Image.open(images_dir / fn) as img:
                    ax.imshow(img.convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"brak obrazu: {exc}", ha="center")
            ax.axis("off")
            ax.set_title(f"GT:  {_window_span(ref, i, len(src))}\n"
                         f"OCR: {_window_span(pred, j, len(tgt))}", fontsize=7, loc="left")
        for ax in axes[len(hits):]:
            ax.axis("off")
        fig.suptitle(f"{rank}. „{src}” → „{tgt}”  ({ev['kind']})   n={ev['count']}, "
                     f"P={ev['p']:.1%}, lift={ev['lift']:.0f}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        name = f"examples/{rank:02d}_{'_'.join(f'{ord(c):04x}' for c in src)}-" \
               f"{'_'.join(f'{ord(c):04x}' for c in tgt)}.png"
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        index.append(f"- `{name}` — „{src}” → „{tgt}” (n={ev['count']}, P={ev['p']:.1%})")
    (ex_dir / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def md_table(header, rows):
    def numeric(v):
        return re.fullmatch(r"[+-]?[\d.]+%?", str(v)) is not None
    first = rows[0] if rows else [""] * len(header)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---:" if numeric(v) else "---" for v in first) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(BASE / "data" / "predictions.jsonl"))
    ap.add_argument("--metrics-json", default=str(BASE / "data" / "metrics.json"))
    ap.add_argument("--images-dir",
                    default=str(REPO / "benchmark" / "dane" / "handlabeled" / "images"))
    ap.add_argument("--out-dir", default=str(BASE / "out_ngram"))
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.1, help="wygładzanie 1:1/indel")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="kara (nats) za operację wieloznakową")
    ap.add_argument("--min-inventory", type=int, default=3,
                    help="min. liczba wystąpień, by kandydat trafił do inwentarza")
    ap.add_argument("--min-count", type=int, default=5, help="próg raportowania")
    ap.add_argument("--min-lift", type=float, default=3.0, help="próg raportowania")
    ap.add_argument("--top-examples", type=int, default=15)
    ap.add_argument("--examples-per-pair", type=int, default=6)
    ap.add_argument("--min-errors", type=int, default=10)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = A.load_predictions(Path(args.predictions))
    lines = [(r.get("file_name", ""), A.normalize(r.get("ground_truth", "")),
              A.normalize(r.get("prediction", ""))) for r in rows]
    groups = [source_group(fn) for fn, _, _ in lines]

    occ1: Counter = Counter()
    occ2: Counter = Counter()
    vocab = set()
    for _, ref, pred in lines:
        occ1.update(ref)
        occ2.update(ref[k:k + 2] for k in range(len(ref) - 1))
        vocab |= set(ref) | set(pred)
    n_slots = sum(len(ref) + 1 for _, ref, _ in lines)
    total_chars = sum(occ1.values())

    cm, aligned, n1, n_ins, multi, history = run_em(lines, occ1, occ2, n_slots, vocab, args)

    # --- baseline (zwykły Levenshtein z analyze.py)
    base = A.accumulate(rows)
    base_sub = Counter({(r, p): c for r, cnt in base["sub_counts"].items() for p, c in cnt.items()})
    base_edits = sum(base_sub.values()) + sum(base["del_counts"].values()) + sum(base["ins_counts"].values())

    # --- zdarzenia wieloznakowe
    ev_groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    ev_words: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for g, (_, ref, _), ops in zip(groups, lines, aligned):
        for s, t, i, _ in ops:
            if (s, t) in multi:
                ev_groups[(s, t)][g] += 1
                ev_words[(s, t)][word_at(ref, i).lower()] += 1
    group_lines = Counter(groups)
    events = []
    for (src, tgt), n in multi.items():
        occ = occ1[src] if len(src) == 1 else occ2[src]
        p = n / occ if occ else 0.0
        gcount = ev_groups[(src, tgt)]
        top_g, top_n = gcount.most_common(1)[0]
        wcount = ev_words[(src, tgt)]
        top_w, top_wn = wcount.most_common(1)[0]
        events.append({
            "src": src, "tgt": tgt, "kind": kind_of(src, tgt), "count": n,
            "src_occ": occ, "p": p, "lift": lift_of(src, tgt, p, cm),
            "category": "ortograficzne" if (src, tgt) in ORTHO else "optyczne?",
            "n_groups": len(gcount), "top_group": top_g, "top_group_share": top_n / n,
            "n_words": len(wcount), "top_word": top_w, "top_word_share": top_wn / n,
        })
    events.sort(key=lambda e: -e["count"])
    reported = [e for e in events if e["count"] >= args.min_count and e["lift"] >= args.min_lift]

    with (out_dir / "multi_events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "tgt", "kind", "count", "src_occ_in_gt", "p_given_src", "lift",
                    "category", "n_groups", "top_group", "top_group_share",
                    "n_words", "top_word", "top_word_share", "reported"])
        for e in events:
            w.writerow([e["src"], e["tgt"], e["kind"], e["count"], e["src_occ"],
                        f"{e['p']:.4f}", f"{e['lift']:.1f}", e["category"], e["n_groups"],
                        e["top_group"], f"{e['top_group_share']:.2f}", e["n_words"],
                        e["top_word"], f"{e['top_word_share']:.2f}", int(e in reported)])

    # --- oczyszczona macierz 1:1
    clean_sub = Counter({k: v for k, v in n1.items() if k[1] and k[0] != k[1]})
    clean_del = Counter({k[0]: v for k, v in n1.items() if not k[1]})
    clean_edits = sum(clean_sub.values()) + sum(clean_del.values()) + sum(n_ins.values())
    multi_edits = sum(multi.values())
    sub_counts = defaultdict(Counter)
    for (a, b), v in clean_sub.items():
        sub_counts[a][b] = v
    acc_clean = {"sub_counts": sub_counts, "n_lines": len(lines)}
    alphabet = A.alphabet_of(occ1, Counter(c for _, _, p in lines for c in p))
    A.plot_heatmap(out_dir, alphabet, acc_clean, args.min_errors, fname="heatmap_clean.png")
    A.plot_heatmap(out_dir, alphabet, acc_clean, args.min_errors, top_n=30, annotate=True,
                   fname="heatmap_clean_top.png")
    with (out_dir / "confusion_counts_clean.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref\\pred"] + [A.display(s) for s in alphabet])
        for r in alphabet:
            w.writerow([A.display(r)] + [clean_sub.get((r, p), 0) for p in alphabet])

    keys = sorted(set(base_sub) | set(clean_sub), key=lambda k: -base_sub.get(k, 0))
    with (out_dir / "sub_before_after.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "pred", "levenshtein", "clean", "delta", "delta_pct"])
        for k in keys:
            b, c = base_sub.get(k, 0), clean_sub.get(k, 0)
            w.writerow([A.display(k[0]), A.display(k[1]), b, c, c - b,
                        f"{(c - b) / b:.2f}" if b else ""])

    # --- per znak: błędy przypisane znakowi (1:1 + del + udział w zdarzeniach wieloznakowych)
    char_err: Counter = Counter()
    for (a, b), v in clean_sub.items():
        char_err[a] += v
    for a, v in clean_del.items():
        char_err[a] += v
    multi_err: Counter = Counter()
    for (src, tgt), v in multi.items():
        for ch in src:
            multi_err[ch] += v
    unit_rows = []
    for ch, occ in occ1.most_common():
        e1, em = char_err[ch], multi_err[ch]
        unit_rows.append((ch, occ, e1, em, (e1 + em) / occ))
    with (out_dir / "unit_error_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "gt_count", "err_1to1_or_del", "err_in_multi", "error_rate",
                    "share_of_all_char_errors"])
        tot_err = sum(r[2] + r[3] for r in unit_rows) or 1
        for ch, occ, e1, em, rate in unit_rows:
            w.writerow([A.display(ch), occ, e1, em, f"{rate:.4f}", f"{(e1 + em) / tot_err:.4f}"])

    print("[ngram] wycinki przykładów...", flush=True)
    plot_multi_examples(out_dir, reported[:args.top_examples], lines, aligned,
                        Path(args.images_dir), args.examples_per_pair)

    # --- raport
    metrics = {}
    if Path(args.metrics_json).is_file():
        metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    rep_edits = sum(e["count"] for e in reported)
    by_kind = Counter()
    for e in reported:
        by_kind[e["kind"]] += e["count"]
    rep = [
        "# Pomyłki wieloznakowe (1:2, 2:1, 2:2) — Surya LoRA r64 / v2_200k",
        "",
        f"- Adapter: `{metrics.get('adapter', '?')}`",
        f"- Zbiór: handlabeled, **{len(lines)}** linii, **{total_chars}** znaków referencyjnych, "
        f"{len(group_lines)} źródeł (zeszytów)",
        f"- Wyrównanie: Viterbi z kosztami -log P, hard-EM, {len(history) - 1} iteracji; "
        f"α={args.alpha}, λ={args.lam}, inwentarz N≥{args.min_inventory}",
        f"- Raportowane: N ≥ {args.min_count} i lift ≥ {args.min_lift}",
        "",
        "> Zastrzeżenia: handlabeled = zbiór selekcji checkpointu (in-sample). "
        "„Źródło” to zeszyt/dokument z nazwy pliku — przybliżenie autora, nie jego "
        "identyfikator. Kategoria „optyczne?” to heurystyka (wszystko spoza listy "
        "reguł ortograficznych) — weryfikować na wycinkach w `examples/`.",
        "",
        "## Bilans edycji",
        "",
        md_table(["wyrównanie", "1:1 sub", "usunięcia", "wstawienia", "wieloznakowe", "edycje (jednostki)"], [
            ("Levenshtein (analyze.py)", sum(base_sub.values()), sum(base["del_counts"].values()),
             sum(base["ins_counts"].values()), 0, base_edits),
            ("ważone + wieloznakowe", sum(clean_sub.values()), sum(clean_del.values()),
             sum(n_ins.values()), multi_edits, clean_edits + multi_edits),
        ]),
        "",
        f"Zdarzenia wieloznakowe łącznie: **{multi_edits}** (w tym raportowane: {rep_edits}; "
        f"wg typu: " + ", ".join(f"{k} = {v}" for k, v in sorted(by_kind.items())) + ").",
        "",
        "## Zdarzenia wieloznakowe (raportowane)",
        "",
        "P = N / wystąpienia źródła w GT. lift = P / P(najlepszej dekompozycji na "
        "niezależne 1:1/indel). „top źródło” / „top słowo” = udział zeszytu / słowa "
        "z największą liczbą zdarzeń (koncentracja: pomyłka ogólna vs jednego autora/słowa).",
        "",
        md_table(["#", "ref → pred", "typ", "n", "wyst. źródła", "P", "lift", "kategoria",
                  "n źródeł", "top źródło", "n słów", "top słowo"],
                 [(i, f"`{e['src']}` → `{e['tgt']}`", e["kind"], e["count"], e["src_occ"],
                   f"{e['p']:.1%}", f"{e['lift']:.0f}", e["category"], e["n_groups"],
                   f"{e['top_group']} ({e['top_group_share']:.0%})", e["n_words"],
                   f"{e['top_word']} ({e['top_word_share']:.0%})")
                  for i, e in enumerate(reported[:40], start=1)]),
        "",
        "## Substytucje 1:1 — przed vs po oczyszczeniu (top-20 wg Levenshteina)",
        "",
        md_table(["ref → pred", "Levenshtein", "po oczyszczeniu", "Δ"],
                 [(f"`{A.display(k[0])}` → `{A.display(k[1])}`", base_sub[k], clean_sub.get(k, 0),
                   f"{(clean_sub.get(k, 0) - base_sub[k]) / base_sub[k]:+.0%}")
                  for k in keys[:20]]),
        "",
        "## Top-20 substytucji 1:1 po oczyszczeniu",
        "",
        md_table(["ref → pred", "liczba", "P(pred | ref)"],
                 [(f"`{A.display(a)}` → `{A.display(b)}`", v, f"{v / occ1[a]:.1%}")
                  for (a, b), v in clean_sub.most_common(20)]),
        "",
        "## Znaki o największym udziale w błędach (po oczyszczeniu)",
        "",
        "Błędy znaku = 1:1 + usunięcie + udział w zdarzeniu wieloznakowym (każdy znak źródła).",
        "",
        md_table(["znak", "wyst. w GT", "błędy 1:1/del", "w wieloznakowych", "% błędów znaku"],
                 [(f"`{A.display(ch)}`", occ, e1, em, f"{rate:.1%}")
                  for ch, occ, e1, em, rate in sorted(unit_rows, key=lambda r: -(r[2] + r[3]))[:20]]),
        "",
        "## Zbieżność EM",
        "",
        md_table(["iteracja", "koszt całkowity", "zdarzenia wieloznakowe", "inwentarz"],
                 [(h["iter"], f"{h['total_cost']:.1f}", h["multi_events"], h["inventory"])
                  for h in history]),
        "",
        "## Pliki",
        "",
        "- `multi_events.csv`, `sub_before_after.csv`, `unit_error_stats.csv`, "
        "`confusion_counts_clean.csv`",
        "- `heatmap_clean.png`, `heatmap_clean_top.png`",
        "- `examples/INDEX.md`",
        "",
    ]
    (out_dir / "RAPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"[ngram] gotowe -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
