#!/usr/bin/env python
"""Analiza pomyłek znak-po-znaku dla OCR na zbiorze handlabeled (dowolny model; predykcje Surya z infer.sh).

Wejście:
    predictions.jsonl  ->  {"file_name", "ground_truth", "prediction"}  (per linia)

Co liczymy:
  * normalizacja tekstu 1:1 z hp_search/eval_cer.py (`" ".join(t.strip().split())`)
    plus NFC (scala znak + łączący diakrytyk w jeden znak),
  * wyrównanie Levenshteina z backtrace, per linia,
  * zbieramy TYLKO substytucje (znak referencyjny != przewidziany); usunięcia
    i wstawienia liczymy osobno (nie wchodzą do macierzy pomyłek),
  * macierz pomyłek (liczby) i "intensywność" = P(pred | ref) bez przekątnej
    (rozkład błędów danego znaku, wiersz sumuje się do 1).

Artefakty (--out-dir):
    confusion_counts.csv           pełna macierz (liczby substytucji)
    confusion_error_prob.csv       P(pred|ref) bez przekątnej (rozkład błędów)
    symbol_stats.csv               per znak: częstość, poprawność, błędy, top-5
    top_confusions_per_symbol.csv  long format: ref, rank, pred, count, prob
    del_ins_stats.csv              usunięcia/wstawienia per znak
    heatmap.png / heatmap_top.png  macierze pomyłek
    small_multiples.png            top litery -> paski top-5 pomyłek
    diacritics.png                 panel diakrytyczny (polski)
    examples/*.png                 wycinki linii dla top par
    confusions.html                interaktywna heatmapa (plotly, offline)
    RAPORT.md                      podsumowanie

Uruchomienie (z katalogu narzędzia, venv z requirements.txt):
    python analyze.py            # data/predictions.jsonl -> out/
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]

# Pary "baza -> diakrytyk" charakterystyczne dla polskiego; używane w panelu.
DIACRITIC_PAIRS = [
    ("a", "ą"), ("c", "ć"), ("e", "ę"), ("l", "ł"), ("n", "ń"),
    ("o", "ó"), ("s", "ś"), ("z", "ź"), ("z", "ż"),
    ("A", "Ą"), ("C", "Ć"), ("E", "Ę"), ("L", "Ł"), ("N", "Ń"),
    ("O", "Ó"), ("S", "Ś"), ("Z", "Ź"), ("Z", "Ż"),
]
# Mylące się "gołe" znaki (bez ogonka) i warianty wielkości liter.
CASE_LETTERS = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż"


# --------------------------------------------------------------------------
# Wczytanie i normalizacja
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Ta sama normalizacja co eval_cer (_normalize_text) + NFC."""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.strip().split())


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Wyrównanie (Levenshtein z backtrace) — tylko do zbierania operacji
# --------------------------------------------------------------------------

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


def align(ref: str, pred: str) -> list[tuple[str, str, int, int]]:
    """Zwraca operacje edycyjne jako (typ, znak, idx_ref, idx_pred).

    Koszt substytucji = usunięcia = wstawienia = 1. Tie-break deterministyczny:
    najpierw dopasowanie/substytucja (po przekątnej), potem usunięcie,
    na końcu wstawienie.
    """
    n, m = len(ref), len(pred)
    # dp[i][j] = koszt wyrównania ref[:i] z pred[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            pj = pred[j - 1]
            if ri == pj:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j - 1], prev[j], row[j - 1])

    ops: list[tuple[str, str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == pred[j - 1]:
            ops.append((MATCH, ref[i - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append((SUB, ref[i - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((DEL, ref[i - 1], i - 1, -1))
            i -= 1
        else:
            ops.append((INS, pred[j - 1], -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# Agregacja
# --------------------------------------------------------------------------


def accumulate(rows: list[dict]):
    gt_counter: Counter = Counter()
    pred_counter: Counter = Counter()
    sub_counts: dict[str, Counter] = defaultdict(Counter)
    del_counts: Counter = Counter()
    ins_counts: Counter = Counter()
    # (file_name, ref, pred, [(ref_ch, pred_ch, idx_ref, idx_pred), ...])
    per_line_subs: list[tuple[str, str, str, list[tuple[str, str, int, int]]]] = []

    for row in rows:
        ref = normalize(row.get("ground_truth", ""))
        pred = normalize(row.get("prediction", ""))
        gt_counter.update(ref)
        pred_counter.update(pred)
        subs_this: list[tuple[str, str, int, int]] = []
        for kind, ch, ri, pi in align(ref, pred):
            if kind == MATCH:
                continue
            if kind == SUB:
                if ch != pred[pi]:
                    sub_counts[ch][pred[pi]] += 1
                    subs_this.append((ch, pred[pi], ri, pi))
            elif kind == DEL:
                del_counts[ch] += 1
            elif kind == INS:
                ins_counts[ch] += 1
        per_line_subs.append((row.get("file_name", ""), ref, pred, subs_this))

    return {
        "gt_counter": gt_counter,
        "pred_counter": pred_counter,
        "sub_counts": sub_counts,
        "del_counts": del_counts,
        "ins_counts": ins_counts,
        "per_line_subs": per_line_subs,
        "n_lines": len(rows),
    }


def error_total(sub_counts: dict[str, Counter], sym: str) -> int:
    return int(sum(sub_counts.get(sym, Counter()).values()))


def alphabet_of(gt_counter: Counter, pred_counter: Counter) -> list[str]:
    """Symbole z GT + predykcji, uporządkowane malejąco po częstości w GT."""
    symbols = set(gt_counter) | set(pred_counter)
    return sorted(symbols, key=lambda c: (-gt_counter.get(c, 0), -pred_counter.get(c, 0), c))


def display(sym: str) -> str:
    if sym == " ":
        return "·"
    if sym == "\t":
        return "\\t"
    if sym == "\n":
        return "\\n"
    return sym


# --------------------------------------------------------------------------
# Zapis CSV
# --------------------------------------------------------------------------


def write_matrices(out_dir: Path, alphabet: list[str], acc) -> None:
    sub = acc["sub_counts"]
    gt_counter = acc["gt_counter"]

    # confusion_counts.csv
    with (out_dir / "confusion_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref\\pred"] + [display(s) for s in alphabet])
        for r in alphabet:
            row = [display(r)]
            for p in alphabet:
                row.append(sub.get(r, Counter()).get(p, 0))
            w.writerow(row)

    # confusion_error_prob.csv  (P(pred|ref), bez przekątnej, wiersz sumuje do 1)
    with (out_dir / "confusion_error_prob.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref\\pred"] + [display(s) for s in alphabet])
        for r in alphabet:
            tot = error_total(sub, r)
            row = [display(r)]
            for p in alphabet:
                if p == r or tot == 0:
                    row.append("")
                else:
                    row.append(f"{sub.get(r, Counter()).get(p, 0) / tot:.4f}")
            w.writerow(row)

    # symbol_stats.csv
    with (out_dir / "symbol_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "symbol", "gt_count", "matches", "substitution_errors", "deletions",
            "error_rate", "top1", "top2", "top3", "top4", "top5",
        ])
        for s in alphabet:
            gt_n = gt_counter.get(s, 0)
            subs = error_total(sub, s)
            dels = acc["del_counts"].get(s, 0)
            matches = max(gt_n - subs - dels, 0)
            rate = (subs + dels) / gt_n if gt_n else 0.0
            tops = sub.get(s, Counter()).most_common(5)
            top_cells = []
            for p, c in tops:
                tot = subs or 1
                top_cells.append(f"{display(p)}:{c}({c / tot:.0%})")
            top_cells += [""] * (5 - len(top_cells))
            w.writerow([display(s), gt_n, matches, subs, dels, f"{rate:.4f}", *top_cells])

    # top_confusions_per_symbol.csv
    with (out_dir / "top_confusions_per_symbol.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "rank", "pred", "count", "prob_given_ref_error"])
        for s in alphabet:
            subs = error_total(sub, s)
            if subs == 0:
                continue
            for rank, (p, c) in enumerate(sub.get(s, Counter()).most_common(5), start=1):
                w.writerow([display(s), rank, display(p), c, f"{c / subs:.4f}"])

    # del_ins_stats.csv
    with (out_dir / "del_ins_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "gt_count", "pred_count", "substitution_errors",
                    "deletions", "insertions", "deletion_rate", "insertion_rate"])
        for s in alphabet:
            gt_n = acc["gt_counter"].get(s, 0)
            pd_n = acc["pred_counter"].get(s, 0)
            dels = acc["del_counts"].get(s, 0)
            inss = acc["ins_counts"].get(s, 0)
            w.writerow([
                display(s), gt_n, pd_n, error_total(sub, s), dels, inss,
                f"{dels / gt_n:.4f}" if gt_n else "",
                f"{inss / pd_n:.4f}" if pd_n else "",
            ])


# --------------------------------------------------------------------------
# Wizualizacje
# --------------------------------------------------------------------------


def _filtered(alphabet, acc, min_errors, min_col=3):
    sub = acc["sub_counts"]
    rows = [s for s in alphabet if error_total(sub, s) >= min_errors]
    row_set = set(rows)
    incoming: Counter = Counter()
    for r in rows:
        for p, c in sub.get(r, Counter()).items():
            incoming[p] += c
    cols = [p for p in alphabet if incoming.get(p, 0) >= min_col or p in row_set]
    cols.sort(key=lambda c: (-incoming.get(c, 0), c))
    return rows, cols


def plot_heatmap(out_dir: Path, alphabet, acc, min_errors: int, top_n=None,
                 annotate=False, fname="heatmap.png"):
    rows, cols = _filtered(alphabet, acc, min_errors)
    if top_n:
        rows = rows[:top_n]
        row_set = set(rows)
        incoming: Counter = Counter()
        for r in rows:
            for p, c in acc["sub_counts"].get(r, Counter()).items():
                incoming[p] += c
        cols = [p for p in cols if incoming.get(p, 0) >= 2 or p in row_set]
        cols.sort(key=lambda c: (-incoming.get(c, 0), c))
    if not rows or not cols:
        return

    sub = acc["sub_counts"]
    mat = np.zeros((len(rows), len(cols)))
    cnt = np.zeros((len(rows), len(cols)), dtype=int)
    for i, r in enumerate(rows):
        tot = error_total(sub, r)
        for j, c in enumerate(cols):
            if c == r:
                continue
            n = sub.get(r, Counter()).get(c, 0)
            cnt[i, j] = n
            if tot:
                mat[i, j] = n / tot
    mat_masked = np.ma.masked_where(mat == 0, mat)

    fig_w = max(8.0, len(cols) * 0.42)
    fig_h = max(6.0, len(rows) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat_masked, cmap="Reds", aspect="auto", vmin=0,
                   vmax=max(mat.max(), 0.01))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([display(c) for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([display(r) for r in rows], fontsize=8)
    ax.set_xlabel("przewidziany znak")
    ax.set_ylabel("znak referencyjny")
    ax.set_title(
        f"Macierz pomyłek — P(pred|ref) bez przekątnej\n"
        f"handlabeled, {acc['n_lines']} linii; wiersze z liczbą błędów ≥ {min_errors}, "
        f"kolumny z ≥3 pomyłkami (pełna macierz w CSV)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(pomyłka na ten znak)")
    if annotate:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if cnt[i, j]:
                    ax.text(j, i, str(cnt[i, j]), ha="center", va="center",
                            fontsize=6, color="black")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def plot_small_multiples(out_dir: Path, alphabet, acc, min_errors: int, top_k=5, n_panels=16):
    sub = acc["sub_counts"]
    letters = [s for s in alphabet if s not in (" ",) and error_total(sub, s) >= min_errors]
    letters = letters[:n_panels]
    if not letters:
        return
    ncols = 4
    nrows = int(np.ceil(len(letters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, letters):
        tot = error_total(sub, s)
        tops = sub.get(s, Counter()).most_common(top_k)
        labels = [display(p) for p, _ in tops][::-1]
        vals = [c / tot for _, c in tops][::-1]
        ax.barh(range(len(vals)), vals, color="#c0392b")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(f"„{display(s)}”  (błędów: {tot})", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(min(v + 0.02, 0.9), i, f"{v:.0%}", va="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(letters):]:
        ax.axis("off")
    fig.suptitle("Z czym mylona jest dana litera (top-%d, intensywność = P(pred|ref,błąd))" % top_k,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "small_multiples.png", dpi=170)
    plt.close(fig)


def plot_diacritics(out_dir: Path, acc):
    sub = acc["sub_counts"]
    pairs = [(b, d) for b, d in DIACRITIC_PAIRS
             if acc["gt_counter"].get(b, 0) + acc["gt_counter"].get(d, 0) > 0]
    # unikamy zdublowanych par (z->ź i z->ż są osobne, to OK)
    if not pairs:
        return
    labels = [f"{b}→{d}" for b, d in pairs]
    fwd, bwd, fwd_n, bwd_n = [], [], [], []
    for b, d in pairs:
        tb, td = error_total(sub, b), error_total(sub, d)
        c_fwd = sub.get(b, Counter()).get(d, 0)
        c_bwd = sub.get(d, Counter()).get(b, 0)
        fwd.append(c_fwd / tb if tb else 0.0)
        bwd.append(c_bwd / td if td else 0.0)
        fwd_n.append(c_fwd)
        bwd_n.append(c_bwd)

    x = np.arange(len(pairs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(9, len(pairs) * 0.5), 5))
    b1 = ax.bar(x - w / 2, fwd, w, label="baza→diakrytyk (np. a→ą)", color="#2980b9")
    b2 = ax.bar(x + w / 2, bwd, w, label="diakrytyk→baza (np. ą→a)", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("P(pomyłka) bez przekątnej")
    ax.set_title("Panel diakrytyczny — mylenie liter z ogonkami (polski)")
    ax.legend(fontsize=8)
    for rect, n in list(zip(b1, fwd_n)) + list(zip(b2, bwd_n)):
        if n:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
                    str(n), ha="center", fontsize=7)
    ax.set_ylim(0, max(max(fwd + bwd, default=0.05) * 1.25, 0.05))
    fig.tight_layout()
    fig.savefig(out_dir / "diacritics.png", dpi=170)
    plt.close(fig)


def _window(text: str, idx: int, half: int = 26) -> str:
    if idx < 0 or idx >= len(text):
        return text[: 2 * half]
    lo, hi = max(0, idx - half), min(len(text), idx + half + 1)
    left = ("…" if lo > 0 else "") + text[lo:idx]
    right = text[idx + 1:hi] + ("…" if hi < len(text) else "")
    return f"{left}[{text[idx]}]{right}"


def plot_examples(out_dir: Path, acc, images_dir: Path, top_pairs: int, per_pair: int):
    sub = acc["sub_counts"]
    all_pairs = [(r, p, c) for r, counter in sub.items() for p, c in counter.items()]
    all_pairs.sort(key=lambda t: -t[2])
    chosen = all_pairs[:top_pairs]
    if not chosen:
        return

    index_lines = ["# Przykłady pomyłek", ""]
    from PIL import Image

    for rank, (ref_ch, pred_ch, count) in enumerate(chosen, start=1):
        # linie, w których wystąpiła dokładnie ta para
        hits = []  # (file_name, ref, pred, ri, pi)
        for file_name, ref, pred, subs in acc["per_line_subs"]:
            for r, p, ri, pi in subs:
                if r == ref_ch and p == pred_ch:
                    hits.append((file_name, ref, pred, ri, pi))
                    break
            if len(hits) >= per_pair:
                break
        if not hits:
            continue
        tot = error_total(sub, ref_ch)
        prob = count / tot if tot else 0.0
        ncols, nrows = 2, int(np.ceil(len(hits) / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (file_name, ref, pred, ri, pi) in zip(axes, hits):
            img_path = images_dir / file_name
            try:
                with Image.open(img_path) as img:
                    ax.imshow(img.convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"brak obrazu: {exc}", ha="center")
            ax.axis("off")
            ax.set_title(f"GT:  {_window(ref, ri)}\nOCR: {_window(pred, pi)}",
                         fontsize=7, loc="left")
        for ax in axes[len(hits):]:
            ax.axis("off")
        fig.suptitle(
            f"{rank}. „{display(ref_ch)}” → „{display(pred_ch)}”   "
            f"n={count}, P={prob:.0%} (z {tot} błędów „{display(ref_ch)}”)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        name = f"examples/{rank:02d}_{ord(ref_ch):04x}_{ord(pred_ch):04x}.png"
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        index_lines.append(f"- `{name}` — „{display(ref_ch)}” → „{display(pred_ch)}” (n={count}, P={prob:.0%})")
    (out_dir / "examples" / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def plot_html(out_dir: Path, alphabet, acc, min_errors: int):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] plotly niedostępny ({exc}) — pomijam confusions.html")
        return

    sub = acc["sub_counts"]
    rows, cols = _filtered(alphabet, acc, min_errors)
    if not rows or not cols:
        return
    z, cd = [], []
    for r in rows:
        tot = error_total(sub, r)
        zr, cdr = [], []
        for c in cols:
            n = 0 if c == r else sub.get(r, Counter()).get(c, 0)
            zr.append((n / tot) if tot else 0.0)
            cdr.append(n)
        z.append(zr)
        cd.append(cdr)

    top_pairs = sorted(
        ((r, p, c) for r, counter in sub.items() for p, c in counter.items()),
        key=lambda t: -t[2],
    )[:25]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.68, 0.32],
        subplot_titles=("Macierz pomyłek P(pred|ref)", "Top-25 pomyłek (liczby)"),
    )
    fig.add_trace(
        go.Heatmap(
            z=z, x=[display(c) for c in cols], y=[display(r) for r in rows],
            customdata=cd, colorscale="Reds",
            hovertemplate="ref „%{y}” → pred „%{x}”<br>P=%{z:.1%}, n=%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[c for _, _, c in top_pairs][::-1],
            y=[f"{display(r)}→{display(p)}" for r, p, _ in top_pairs][::-1],
            orientation="h",
            marker_color="#c0392b",
            hovertemplate="%{y}: %{x}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title=f"Pomyłki znaków — Surya LoRA, handlabeled (wiersze z błędów ≥ {min_errors})",
        height=max(600, len(rows) * 22), width=1500, template="plotly_white",
        margin=dict(l=80, r=40, t=80, b=60),
    )
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.write_html(out_dir / "confusions.html", include_plotlyjs=True)
    print(f"[analyze] zapisano {out_dir / 'confusions.html'}")


# --------------------------------------------------------------------------
# Raport
# --------------------------------------------------------------------------


def write_report(out_dir: Path, acc, metrics: dict, min_errors: int):
    sub = acc["sub_counts"]
    alphabet = alphabet_of(acc["gt_counter"], acc["pred_counter"])
    n_lines = acc["n_lines"]
    total_chars = sum(acc["gt_counter"].values())
    total_sub = sum(error_total(sub, s) for s in sub)
    total_del = sum(acc["del_counts"].values())
    total_ins = sum(acc["ins_counts"].values())

    top_pairs = sorted(
        ((r, p, c) for r, counter in sub.items() for p, c in counter.items()),
        key=lambda t: -t[2],
    )[:20]

    worst = sorted(
        [s for s in alphabet if acc["gt_counter"].get(s, 0) >= 10],
        key=lambda s: -((error_total(sub, s) + acc["del_counts"].get(s, 0)) / acc["gt_counter"][s]),
    )[:15]

    lines: list[str] = []
    lines.append("# Pomyłki znaków — Surya LoRA r64 / v2_200k (best checkpoint)")
    lines.append("")
    lines.append(f"- Adapter: `{metrics.get('adapter', '?')}`")
    lines.append(f"- Zbiór: handlabeled, **{n_lines}** linii, **{total_chars}** znaków referencyjnych")
    if metrics:
        m = metrics.get("metrics", {})
        lines.append(f"- CER = `{m.get('cer'):.6f}`  WER = `{m.get('wer'):.6f}`  EMA = `{m.get('ema'):.4f}`"
                     if m else "- (brak metrics.json)")
    lines.append("")
    lines.append("> Zastrzeżenie: handlabeled to **ten sam** zbiór, po którym wybierano checkpoint")
    lines.append("> (callback `eval_cer`), więc analiza jest „in-sample” względem selekcji. Struktura")
    lines.append("> pomyłek jest mimo to diagnostyczna. Rzadkie znaki (Ą 13, Ć 4, Ź 5 wystąpień)")
    lines.append("> mają zbyt mało prób, by ich statystyki były stabilne. Spacja pokazywana jako `·`.")
    lines.append("")
    lines.append("## Bilans edycji (całe 2160 linii)")
    lines.append("")
    lines.append(f"- substytucje: **{total_sub}**")
    lines.append(f"- usunięcia: **{total_del}**")
    lines.append(f"- wstawienia: **{total_ins}**")
    lines.append(f"- suma edycji: **{total_sub + total_del + total_ins}**")
    lines.append("")
    lines.append("## Top-20 pomyłek (substytucje)")
    lines.append("")
    lines.append("| # | ref → pred | liczba | P(pred \\| błąd ref) |")
    lines.append("|---|---|---:|---:|")
    for i, (r, p, c) in enumerate(top_pairs, start=1):
        tot = error_total(sub, r) or 1
        lines.append(f"| {i} | `{display(r)}` → `{display(p)}` | {c} | {c / tot:.1%} |")
    lines.append("")
    lines.append("## Najgorsze znaki (min. 10 wystąpień, wg odsetka błędów)")
    lines.append("")
    lines.append("| znak | wystąpień | odsetek błędów (sub+del) | top pomyłka |")
    lines.append("|---|---:|---:|---|")
    for s in worst:
        gt_n = acc["gt_counter"][s]
        errs = error_total(sub, s) + acc["del_counts"].get(s, 0)
        top = sub.get(s, Counter()).most_common(1)
        top_s = f"`{display(top[0][0])}` ({top[0][1]})" if top else "—"
        lines.append(f"| `{display(s)}` | {gt_n} | {errs / gt_n:.1%} | {top_s} |")
    lines.append("")
    lines.append("## Diakrytyki")
    lines.append("")
    lines.append("| para | baza→diakrytyk | diakrytyk→baza |")
    lines.append("|---|---:|---:|")
    for b, d in DIACRITIC_PAIRS:
        if acc["gt_counter"].get(b, 0) + acc["gt_counter"].get(d, 0) == 0:
            continue
        tb, td = error_total(sub, b), error_total(sub, d)
        cf = sub.get(b, Counter()).get(d, 0)
        cb = sub.get(d, Counter()).get(b, 0)
        lines.append(
            f"| `{b}`↔`{d}` | {cf} ({cf / tb:.0%}) | {cb} ({cb / td:.0%}) |"
            if tb and td else f"| `{b}`↔`{d}` | {cf} | {cb} |"
        )
    lines.append("")
    lines.append("## Pliki")
    lines.append("")
    for f in ["heatmap.png", "heatmap_top.png", "small_multiples.png", "diacritics.png",
              "confusions.html", "confusion_counts.csv", "confusion_error_prob.csv",
              "symbol_stats.csv", "top_confusions_per_symbol.csv", "del_ins_stats.csv",
              "examples/INDEX.md"]:
        lines.append(f"- `{f}`")
    lines.append("")
    (out_dir / "RAPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(BASE / "data" / "predictions.jsonl"))
    ap.add_argument("--metrics-json", default=str(BASE / "data" / "metrics.json"))
    ap.add_argument("--images-dir",
                    default=str(REPO / "benchmark" / "dane" / "handlabeled" / "images"))
    ap.add_argument("--out-dir", default=str(BASE / "out"))
    ap.add_argument("--min-errors", type=int, default=10,
                    help="Min. liczba błędów znaku, by trafił do heatmapy")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--top-examples", type=int, default=15)
    ap.add_argument("--examples-per-pair", type=int, default=6)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples").mkdir(exist_ok=True)

    metrics = {}
    metrics_path = Path(args.metrics_json)
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    rows = load_predictions(Path(args.predictions))
    if not rows:
        print("BŁĄD: brak predykcji", flush=True)
        return 1
    print(f"[analyze] wczytano {len(rows)} linii predykcji", flush=True)

    acc = accumulate(rows)
    alphabet = alphabet_of(acc["gt_counter"], acc["pred_counter"])
    print(f"[analyze] alfabet: {len(alphabet)} symboli; "
          f"substytucje={sum(error_total(acc['sub_counts'], s) for s in acc['sub_counts'])}, "
          f"usunięcia={sum(acc['del_counts'].values())}, "
          f"wstawienia={sum(acc['ins_counts'].values())}", flush=True)

    print("[analyze] CSV...", flush=True)
    write_matrices(out_dir, alphabet, acc)

    print("[analyze] heatmapy...", flush=True)
    plot_heatmap(out_dir, alphabet, acc, args.min_errors, fname="heatmap.png")
    plot_heatmap(out_dir, alphabet, acc, args.min_errors, top_n=30, annotate=True,
                 fname="heatmap_top.png")
    plot_small_multiples(out_dir, alphabet, acc, args.min_errors, top_k=args.top_k)
    plot_diacritics(out_dir, acc)

    print("[analyze] wycinki przykładów...", flush=True)
    plot_examples(out_dir, acc, Path(args.images_dir), args.top_examples, args.examples_per_pair)

    print("[analyze] HTML...", flush=True)
    plot_html(out_dir, alphabet, acc, args.min_errors)

    print("[analyze] raport...", flush=True)
    write_report(out_dir, acc, metrics, args.min_errors)

    print(f"[analyze] gotowe -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
