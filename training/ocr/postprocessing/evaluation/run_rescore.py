"""Ticket 20 krok C: korektor + ocena kandydatów przez Suryę (teacher forcing).
s(c) = ch + lam*lp + w*sd(c), sd(c) = logP_surya(linia z c | obraz) - logP_surya(oryginał | obraz); s_keep bez zmian.
Kandydaci ograniczeni do top-K z gen_cands (tylko dla nich jest sd). w=0 w tej samej siatce = punkt odniesienia.
Zagnieżdżone LOGO (11 grup) + dev -> test (TEST_GROUPS z run_holdout).

    PYTHONPATH=. python -m evaluation.run_rescore KATALOG --predictions P.jsonl --groups data/v12/groups.csv
        [--fixed-w 0] [--deploy results/v12/deploy_params_surya.json]
KATALOG = wyjście rescore_cands (meta.json) + resC.jsonl z serwera (evaluation/surya_lp.py).
Wyniki 2026-11-25 (v4_200k_25k, v1.2): results/v12/README.md.
"""
import argparse, itertools, json, math, sys
from pathlib import Path
import numpy as np
from evaluation import run_cv as R
from evaluation.compare import compare
from evaluation.folds import logo, nested_logo
from evaluation.run_holdout import split_lines, sub_logo
from pp.error_model import Params
from pp.lexicon import Lexicon, domain_terms
from pp.lm import UnigramLM
from pp.tokenize import restore_case

ap = argparse.ArgumentParser()
ap.add_argument("lpdir", type=Path)
ap.add_argument("--predictions", type=Path, required=True)
ap.add_argument("--groups", type=Path)
ap.add_argument("--harm", type=float, default=0.003)
ap.add_argument("--ws", default="0,0.25,0.5,1,2")
ap.add_argument("--fixed-w", type=float, default=None, help="tylko ta waga (bez strojenia w)")
ap.add_argument("--deploy", type=Path, help="progi do wdrożenia: LOGO na wszystkich grupach -> ten plik json")
a = ap.parse_args()
WS = [a.fixed_w] if a.fixed_w is not None else [float(x) for x in a.ws.split(",")]
GRID = {"lam": [0.5, 0.75, 1.0], "f_min": [0, 5], "delta": [x / 2 for x in range(-32, 21)], "mu": [0.0, 0.5, 1.0, 2.0, 3.0],
        "L_min": [3, 4, 5, 6]}

lex = Lexicon(); lm = UnigramLM(lex.targets()); lp_unseen = math.log(lm.unseen) - lm.logz
d = R.prepare(lex, "fine", predictions=a.predictions, groups_csv=a.groups)
terms = frozenset(domain_terms(["en", "la", "pl"]))
em = Params(multi=True)
meta = json.load(open(a.lpdir / "meta.json"))
sd = {}
for line in open(a.lpdir / "resC.jsonl", encoding="utf-8"):
    r = json.loads(line); i = r["id"]; base = r["scores"][0]["lp"]
    for (pi, c), s in zip(meta[str(i)], r["scores"][1:]):
        sd[(i, pi, c)] = s["lp"] - base
print("sd:", len(sd), flush=True)


def recs(f, lines):
    out = []
    for i in lines:
        for pi, t in enumerate(d.toks[i]):
            if not R.eligible(d, i, pi, f.abbrev):
                continue
            sc = f.scored(d, t.key)
            idx = [j for j in range(len(sc.cands)) if (i, pi, sc.cands[j]) in sd and sc.lev[j] <= 3]
            if idx:
                out.append((i, pi, sc, idx))
    return out


def pick(sc, idx, i, pi, lam, fmin, w):
    b, sb, sb2 = -1, -math.inf, -math.inf
    for j in idx:
        if sc.cnt[j] < fmin:
            continue
        s = sc.ch[j] + lam * sc.lp[j] + w * sd[(i, pi, sc.cands[j])]
        if s > sb:
            b, sb, sb2 = j, s, sb
        elif s > sb2:
            sb2 = s
    return b, sb, sb2


def tune(rs, n_ok):
    best, info = None, None
    for lam, fm, w in itertools.product(GRID["lam"], GRID["f_min"], WS):
        g, m, L, dl, br = [], [], [], [], []
        for i, pi, sc, idx in rs:
            b, sb, sb2 = pick(sc, idx, i, pi, lam, fm, w)
            if b < 0:
                continue
            dd, broken, _ = R.iso_delta(d, i, pi, sc.cands[b])
            g.append(sb - (sc.keep_ch + lam * lp_unseen)); m.append(sb - sb2); L.append(len(sc.key)); dl.append(dd); br.append(broken)
        if not g:
            continue
        g, m, L, dl, br = map(np.asarray, (g, m, L, dl, br))
        for lmin in GRID["L_min"]:
            for mu in GRID["mu"]:
                base = (L >= lmin) & (m >= mu)
                for delta in GRID["delta"]:
                    acc = base & (g >= delta)
                    harm = br[acc].sum() / n_ok
                    if harm > a.harm:
                        continue
                    tot = int(dl[acc].sum())
                    if best is None or tot < info["d_edits"]:
                        best = dict(lam=lam, f_min=fm, w=w, L_min=lmin, mu=mu, delta=delta)
                        info = {"d_edits": tot, "harm": float(harm), "n": int(acc.sum())}
    return best, info


def apply(f, lines, p):
    for i, pi, sc, idx in recs(f, lines):
        if len(sc.key) < p["L_min"]:
            continue
        b, sb, sb2 = pick(sc, idx, i, pi, p["lam"], p["f_min"], p["w"])
        if b >= 0 and sb - (sc.keep_ch + p["lam"] * lp_unseen) >= p["delta"] and sb - sb2 >= p["mu"]:
            t = d.toks[i][pi]
            out_tok[(i, pi)] = t.rebuild(restore_case(t.core, sc.cands[b]))


def render(lines, out):
    for i in lines:
        parts = [out_tok.get((i, pi), t.raw) for pi, t in enumerate(d.toks[i])]
        out[i] = " ".join(parts)


def summary(lines, out, label):
    r = compare([d.gts[i] for i in lines], [d.prs[i] for i in lines], [out[i] for i in lines], [d.groups[i] for i in lines])
    print(f"{label}: CER {r['base_cer']:.5f} -> {r['cer']:.5f} (Δ {r['d_cer'] * 100:+.3f} p.p.), CI linie "
          f"[{r['ci_lines_lo'] * 100:+.2f}, {r['ci_lines_hi'] * 100:+.2f}], fixed {r['fixed']} broken {r['broken']} "
          f"changed_wrong {r['changed_wrong']} harm {r['harm'] * 100:.2f}%", flush=True)
    return r


res = {"ws": WS, "harm": a.harm}
if a.deploy:
    rs, ok = [], 0
    for g, itr, ite in logo(d.groups):
        rs += recs(R.Fold(d, lex, lm, itr, em), ite); ok += R.n_correct(d, ite)
    p, info = tune(rs, ok)
    p.update(d_max=3, k=5, lam_gen=0.75, topk_only=True)
    json.dump({"harm_max": a.harm, "params": p, "logo_all_groups": info,
               "note": f"progi z oceną Suryi (w), LOGO {len(set(d.groups))} grup; k=5 kandydatów wg ch+0.75*lp"},
              open(a.deploy, "w"), indent=1)
    print("deploy:", p, info)
    sys.exit(0)
# LOGO 11 grup
out_tok = {}
for g, train, test, inner in nested_logo(d.groups):
    rs, ok = [], 0
    for h, itr, ite in inner:
        rs += recs(R.Fold(d, lex, lm, itr, em), ite); ok += R.n_correct(d, ite)
    p, info = tune(rs, ok)
    apply(R.Fold(d, lex, lm, train, em, protect=terms), test, p)
    print(g, p, info, flush=True)
out = list(d.prs); render(range(len(d.prs)), out)
res["logo"] = summary(range(len(d.prs)), out, "LOGO")
# dev -> test
dev, test = split_lines(d.groups)
out_tok = {}
rs, ok = [], 0
for h, itr, ite in sub_logo(d.groups, dev):
    rs += recs(R.Fold(d, lex, lm, itr, em), ite); ok += R.n_correct(d, ite)
p, info = tune(rs, ok)
apply(R.Fold(d, lex, lm, dev, em, protect=terms), test, p)
out = list(d.prs); render(test, out)
print("progi z dev:", p, info)
res["params_dev"] = p
res["test"] = summary(test, out, "TEST")
tag = f"w{a.fixed_w}" if a.fixed_w is not None else "wtuned"
json.dump(res, open(a.lpdir / f"eval_{tag}_h{a.harm}.json", "w"), indent=1, default=float)
