# Postprocessing wyjścia Surya (statyczny, bez ponownej inferencji)

Korekta linii z Surya v2_200k (LoRA r64) słownikiem (PoliMorf 2026 + NKJP) i modelem kanału
Brill & Moore uczonym na parach (GT, predykcja). Ewaluacja na handlabeled (2160 linii)
w zagnieżdżonym leave-one-group-out. Wyniki i wnioski: [`results/RAPORT.md`](results/RAPORT.md),
pełna tabela: [`results/TABLE.md`](results/TABLE.md).

**Wynik główny** (harm ≤ 0.3% na walidacji wewnętrznej): CER 0.1277 → 0.1048 (−2.29 p.p., −18%,
95% CI po grupach [−2.93, −1.48]), WER 0.482 → 0.377, harm na teście 0.40%.

## Uruchomienie

```bash
cd training/ocr/postprocessing
python3 -m venv .venv          # na tej maszynie: ~/tools/miniconda3/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_resources.py        # PoliMorf 2026-10-18, NKJP 1grams, sjp odmiany → $PP_CACHE
.venv/bin/python -m pytest -q tests

export PYTHONPATH=.
.venv/bin/python -m evaluation.metrics      # CER/WER baseline (= eval_cer.py)
.venv/bin/python -m evaluation.folds        # data/baseline/groups.csv
.venv/bin/python -m evaluation.oracle       # pułapy zysku, lista podejrzanych etykiet GT
.venv/bin/python -m evaluation.p2_report    # pokrycie słownika, opcja 0, klasy chronione
.venv/bin/python -m evaluation.p3_report    # model błędu per fold vs Levenshtein
.venv/bin/python -m evaluation.run_cv --harm-max 0.003 --tag option1_harm03   # wariant główny (~10 min)
.venv/bin/python -m evaluation.make_table   # results/TABLE.md
```

`$PP_CACHE` domyślnie `~/.cache/ocr-pl-med/postprocessing` (poza repo; tam też cache prefiltra).
Ablacje `run_cv`: `--no-multi`, `--no-protect`, `--diacritics-only`, `--sjp`, `--forward`,
`--split coarse`. `--dump PLIK` zapisuje predykcje z tekstem linii: tylko poza repo.

## API

```python
from pp.correct import Corrector
from pp.decide import DecisionParams

corr = Corrector.from_training(gt_lines, pred_lines, groups,
                               DecisionParams(lam=1.0, delta=1.0, mu=3.0, d_max=3, L_min=3, f_min=20))
corr.correct_lines(["Tętnica podholanowy biegnie (pawierzchnia)"])
# -> ["Tętnica podkolanowy biegnie (powierzchnia)"]
```

Inicjalizacja ok. 26 s (słownik 4.7 mln form, LM, model kanału). Korekta działa partią:
prefiltr kosztuje ok. 11 ms/token w partii, ok. 340 ms dla pojedynczego tokenu, więc lepiej
korygować całą stronę naraz. Parametry powyżej to progi do wdrożenia z
`run_cv --deploy --harm-max 0.003` (LOGO na wszystkich 8 grupach → `results/deploy_params.json`;
to nie jest wynik testowy).

## Układ

| Ścieżka | Co |
|---|---|
| `data/baseline/predictions.jsonl` | predykcje Surya v2_200k na handlabeled (z etykietami), 2026-10-01 |
| `data/baseline/metrics.json`, `groups.csv` | wynik `eval_cer.py` (CER 0.12769, WER 0.48174); mapa linia → grupa |
| `pp/normalize.py`, `tokenize.py`, `protect.py` | opcja 0, tokenizacja + wielkość liter, klasy chronione P1–P8 |
| `pp/lexicon.py`, `lm.py` | słownik `known` / `target`, P(w) z NKJP + Simple Good-Turing |
| `pp/error_model.py` | model kanału Brill & Moore (hard-EM, jednostki do 2 znaków, Viterbi/forward) |
| `pp/candidates.py`, `decide.py`, `correct.py` | prefiltr rapidfuzz, reguła decyzji, API |
| `evaluation/` | metryki, istotność (`sig.py`), foldy, oracle, raporty P2–P4, `run_cv.py`, `make_table.py` |
| `results/` | CSV/JSON/MD bez tekstu linii |

## Ograniczenia

- Grupa (fold) = źródło notatek, nie zweryfikowany autor; liczba autorów handlabeled nieznana.
- 8 grup: przedziały po grupach są szerokie; raportujemy też przedziały po liniach.
- Checkpoint Surya był wybierany na tym samym zbiorze: bezwzględny CER jest „in-sample”, delta
  korekty mniej (dotyczy obu stron porównania jednakowo).
- Część „zepsuć” to błędy etykiet GT (brak diakrytyków, ticket 09); oficjalny GT bez zmian.

## Atrybucja zasobów

- PoliMorf (2026-10-18), Zespół Inżynierii Lingwistycznej IPI PAN / SGJP, licencja BSD 2-clause.
- NKJP 1-gramy (podkorpus zrównoważony), IPI PAN, CC-BY.
- sjp.pl „odmiany” (2026-11-01), CC BY 4.0 (jedna z licencji do wyboru). Tylko ablacja.
- Skróty anatomiczne: Terminologia Anatomica (TA98), lista ręczna 12 skrótów.
