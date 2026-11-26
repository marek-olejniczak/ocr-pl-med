# Postprocessing wyjścia Surya (statyczny, bez ponownej inferencji)

Korekta linii z Surya v2_200k (LoRA r64) słownikiem (PoliMorf 2026 + NKJP) i modelem kanału
Brill & Moore uczonym na parach (GT, predykcja). Ewaluacja na handlabeled (2160 linii)
w zagnieżdżonym leave-one-group-out. Wyniki i wnioski: [`results/RAPORT.md`](results/RAPORT.md),
pełna tabela: [`results/TABLE.md`](results/TABLE.md).

**Wynik główny** (dataset-v1, harm ≤ 0.3% na walidacji, lista terminów `en,la,pl`): na zamrożonym teście
CER 0.1238 → 0.1062 (−1.76 p.p., 95% CI po liniach [−2.05, −1.44]), harm 0.00%; w LOGO na 2160 liniach
CER 0.1270 → 0.0987 (−2.83 p.p., 95% CI po grupach [−3.38, −1.89]), harm 0.31%.

**Stan na 2026-11-24** (Surya v4_200k_25k, test/real/v1.2, 2904 linie bez `[BLANK]`): LOGO po 11 grupach
CER 0.1098 → 0.0975 (−1.23 p.p.), harm 0.27%; zamrożony test 0.1040 → 0.0956 (−0.84 p.p.), harm 0.05%.
Na v2/v3 ten sam korektor dawał ok. −2.5 p.p. Szczegóły i progi dla v4: [`results/v12/`](results/v12/README.md).
Wybór tylko spośród 5 najlepszych kandydatów daje −1.59 p.p., a z oceną kandydatów przez Suryę (log P linii przy obrazie)
−2.40 p.p. w LOGO i −2.56 p.p. na teście. Integracja z aplikacją: [`INTEGRACJA.md`](INTEGRACJA.md), paczka `deploy/v4_200k_25k`.

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
.venv/bin/python -m evaluation.run_cv --harm-max 0.003 --tag v1_ta --domain en,la,pl   # LOGO, wariant główny (~10 min)
.venv/bin/python -m evaluation.make_table   # results/TABLE.md
.venv/bin/python -m evaluation.run_holdout --tag ta --domain en,la,pl          # warianty porównuj na dev
.venv/bin/python -m evaluation.run_holdout --tag ta --domain en,la,pl --test   # test: raz, po wyborze wariantu
```

`--domain` wybiera języki listy terminów chronionych przed korektą (`data/domain/ta_terms.tsv`), np. `la,pl`
dla tekstów bez angielskiego; bez flagi lista jest wyłączona. Progi stroimy bez listy, więc zmiana języków
nie wymaga ponownego strojenia. `--domain-in-tuning` to ablacja z listą także w strojeniu.

`$PP_CACHE` domyślnie `~/.cache/ocr-pl-med/postprocessing` (poza repo; tam też cache prefiltra).
Ablacje `run_cv`: `--uniform`, `--no-multi`, `--no-group-cap`, `--no-protect`, `--diacritics-only`, `--sjp`,
`--forward`, `--split coarse`. `--dump PLIK` zapisuje predykcje z tekstem linii: tylko poza repo.
Inne predykcje (np. Surya v3, test v1.1) w formacie `data/baseline/predictions.jsonl`: `--predictions PLIK`,
wyniki wtedy poza `results/`: `--out-dir KATALOG`.
Predykcje z przebiegu benchmarku importuje `evaluation.import_bench` (jsonl poza repo + mapa grup `data/v12/groups.csv`).
Dla test/real/v1.2 podawaj `--groups data/v12/groups.csv` (strony `new_*` nie mają źródła w nazwie). Linie `[BLANK]`
(zostają w zbiorze dla detekcji) `load_predictions` pomija.

Higiena wejścia korektora (`pp/normalize.hygiene`) zwija zapętlenia modelu (`deloop`): n-gramy do 3 słów
powtórzone ≥ 3 razy z rzędu oraz powtórzone segmenty `<br>`. Surya v3 na RxHandBD: CER 0.710 → 0.600;
na notatkach dataset-v1 i v1.1 zmienia 0–1 linii.

## API

```python
from pp.correct import Corrector
from pp.decide import DecisionParams

corr = Corrector.from_training(gt_lines, pred_lines, groups,
                               DecisionParams(lam=0.75, delta=-3.0, mu=3.0, d_max=3, L_min=3, f_min=1),
                               domain_langs=("la", "pl"))
corr.correct_lines(["Tętnica podholanowy biegnie (pawierzchnia)"])
# -> ["Tętnica podkolanowy biegnie (pawierzchnia)"]; drugi token odpada na marginesie μ
```

Inicjalizacja ok. 26 s (słownik 4.7 mln form, LM, model kanału). Korekta działa partią:
prefiltr kosztuje ok. 11 ms/token w partii, ok. 340 ms dla pojedynczego tokenu, więc lepiej
korygować całą stronę naraz. Parametry powyżej to progi do wdrożenia z
`run_cv --deploy --harm-max 0.003` (LOGO na wszystkich 8 grupach → `results/deploy_params.json`;
to nie jest wynik testowy).

## Układ

| Ścieżka | Co |
|---|---|
| `data/baseline/predictions.jsonl` | predykcje Surya v2_200k na dataset-v1 (z etykietami), 2026-11-05 |
| `data/baseline/metrics.json`, `groups.csv` | wynik `eval_cer.py` (CER 0.12701, WER 0.48307); mapa linia → grupa |
| `deploy/v4_200k_25k/` | paczka do wdrożenia: model pomyłek + progi (statyczne i z oceną Suryi) |
| `pp/deploy.py`, `pp_worker.py`, `pp/surya_score.py` | ładowanie paczki, serwis HTTP, ocena tekstu przez Suryę |
| `data/v12/groups.csv` | test/real/v1.2: linia → grupa (G1–G11) i split dev/test |
| `data/domain/` | lista terminów TA z Wikidata (CC0) i kwerenda SPARQL |
| `pp/normalize.py`, `tokenize.py`, `protect.py` | opcja 0, tokenizacja + wielkość liter, klasy chronione P1–P8 |
| `pp/lexicon.py`, `lm.py` | słownik `known` / `target`, P(w) z NKJP + Simple Good-Turing |
| `pp/error_model.py` | model kanału Brill & Moore (hard-EM, jednostki do 2 znaków, Viterbi/forward) |
| `pp/candidates.py`, `decide.py`, `correct.py` | prefiltr rapidfuzz, reguła decyzji, API |
| `evaluation/` | metryki, istotność (`sig.py`), foldy, oracle, raporty P2–P4, `run_cv.py`, `run_holdout.py`, `make_table.py` |
| `results/` | CSV/JSON/MD bez tekstu linii |

## Ograniczenia

- Grupa (fold) = źródło notatek, nie zweryfikowany autor; liczba autorów handlabeled nieznana.
- 8 grup: przedziały po grupach są szerokie; raportujemy też przedziały po liniach.
- Checkpoint Surya był wybierany na tym samym zbiorze: bezwzględny CER jest „in-sample”, delta
  korekty mniej (dotyczy obu stron porównania jednakowo).
- Część „zepsuć” to diakrytyki pominięte przez autorkę; GT jest wierny obrazowi.
- Strojenie ogranicza harm, ale nie błędne podmiany (changed-wrong): RAPORT §4b, §5.
- `oracle`, `p2_report`, `p3_report` i ablacje z RAPORT §2 mają zapisane wyniki z predykcji 2026-10-01.

## Atrybucja zasobów

- PoliMorf (2026-10-18), Zespół Inżynierii Lingwistycznej IPI PAN / SGJP, licencja BSD 2-clause.
- NKJP 1-gramy (podkorpus zrównoważony), IPI PAN, CC-BY.
- sjp.pl „odmiany” (2026-11-01), CC BY 4.0 (jedna z licencji do wyboru). Tylko ablacja.
- Skróty anatomiczne: Terminologia Anatomica (TA98), lista ręczna 12 skrótów.
