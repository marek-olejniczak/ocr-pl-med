"""Ticket 20 krok C: kandydaci do oceny przez Suryę (wejście dla evaluation/surya_lp.py na serwerze).

    python -m evaluation.rescore_cands --predictions P.jsonl --groups data/v12/groups.csv \
        --images-prefix benchmark/dane/real_v1.2/images/ --out KATALOG      # poza repo: zawiera tekst linii

 Dla każdej grupy g model kanału z pozostałych grup (bez przecieku),
dla tokenów kwalifikujących się w liniach g top-K kandydatów wg ch + lam*lp. Wyjście:
  req.jsonl  (na serwer: id = indeks linii, image, texts = [oryginał, warianty...])
  meta.json  (lokalnie: dla linii -> lista (pozycja tokenu, kandydat) w kolejności wariantów)
"""
import argparse, json
from pathlib import Path
from evaluation import run_cv as R
from evaluation.folds import logo
from ocr_postprocess.error_model import Params
from ocr_postprocess.lexicon import Lexicon, domain_terms
from ocr_postprocess.lm import UnigramLM
from ocr_postprocess.tokenize import restore_case

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", type=Path, required=True)
ap.add_argument("--groups", type=Path)
ap.add_argument("--images-prefix", required=True, help="ścieżka obrazów linii względem /app w kontenerze")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--lam-gen", type=float, default=0.75)
a = ap.parse_args()
K, LAM, out = a.k, a.lam_gen, a.out
out.mkdir(parents=True, exist_ok=True)
lex = Lexicon(); lm = UnigramLM(lex.targets())
d = R.prepare(lex, "fine", predictions=a.predictions, groups_csv=a.groups)
terms = frozenset(domain_terms(["en", "la", "pl"]))
em = Params(multi=True)
meta = {}
for g, train, test in logo(d.groups):
    f = R.Fold(d, lex, lm, train, em, protect=terms)
    for i in test:
        vs = []
        for pi, t in enumerate(d.toks[i]):
            if not R.eligible(d, i, pi, f.abbrev):
                continue
            sc = f.scored(d, t.key)
            order = sorted((j for j in range(len(sc.cands)) if sc.lev[j] <= 3), key=lambda j: -(sc.ch[j] + LAM * sc.lp[j]))[:K]
            vs += [(pi, sc.cands[j]) for j in order]
        if vs:
            meta[i] = vs
    print(g, len(meta), flush=True)
with open(out / "req.jsonl", "w", encoding="utf-8") as fo:
    for i, vs in meta.items():
        texts = [d.prs[i]]
        for pi, c in vs:
            parts = [t.raw for t in d.toks[i]]
            t = d.toks[i][pi]
            parts[pi] = t.rebuild(restore_case(t.core, c))
            texts.append(" ".join(parts))
        fo.write(json.dumps({"id": i, "image": a.images_prefix + d.rows[i]["file_name"], "texts": texts},
                            ensure_ascii=False) + "\n")
json.dump({str(i): vs for i, vs in meta.items()}, open(out / "meta.json", "w"), ensure_ascii=False)
print("linii", len(meta), "wariantów", sum(len(v) for v in meta.values()))
