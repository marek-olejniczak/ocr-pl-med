# Postprocessing wyjścia Surya (statyczny, bez ponownej inferencji)

Korekta linii z Surya v2_200k (LoRA r64) słownikiem + modelem kanału (Brill & Moore),
ewaluacja na handlabeled (2160 linii) w protokole leave-one-group-out.

## Uruchomienie

```bash
cd training/ocr/postprocessing
python3 -m venv .venv          # na tej maszynie: ~/tools/miniconda3/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_resources.py        # PoliMorf 2026-10-18, NKJP 1grams, sjp odmiany → $PP_CACHE
.venv/bin/python -m pytest -q tests
PYTHONPATH=. .venv/bin/python -m evaluation.metrics   # CER/WER baseline
```

`$PP_CACHE` domyślnie `~/.cache/ocr-pl-med/postprocessing` (poza repo).

## Układ

| Ścieżka | Co |
|---|---|
| `data/baseline/predictions.jsonl` | predykcje Surya v2_200k na handlabeled (z etykietami), 2026-10-01 |
| `data/baseline/metrics.json` | wynik `eval_cer.py` dla tych predykcji: CER 0.12769, WER 0.48174 |
| `evaluation/metrics.py` | CER/WER mikro zgodne 1:1 z `surya/hp_search/eval_cer.py` (+ metryka pomocnicza) |
| `evaluation/sig.py` | paired bootstrap (po grupach / liniach), permutacja, test znaków |
| `fetch_resources.py` | pobranie zasobów z pinowanymi wersjami i SHA-256 |
| `pp/` | korektor (normalizacja, tokenizacja, słownik, P(w), model błędu, kandydaci, decyzja) |
| `results/` | wyniki zagregowane (bez tekstu linii) |

## Baseline

| metryka | wartość |
|---|---|
| CER (główny, jak eval_cer.py) | 0.12769 (5402 / 42305) |
| WER | 0.48174 (2902 / 6024) |
| CER pomocniczy (casefold, bez interpunkcji) | 0.11953 |

Wartość 0.127243 z ablacji checkpointów pochodzi z innego przebiegu inferencji.

## Atrybucja zasobów

- PoliMorf (2026-10-18), Zespół Inżynierii Lingwistycznej IPI PAN / SGJP, licencja BSD 2-clause.
- NKJP 1-gramy (podkorpus zrównoważony), IPI PAN, CC-BY.
- sjp.pl „odmiany” (2026-11-01), CC BY 4.0 (jedna z licencji do wyboru). Tylko ablacja.
