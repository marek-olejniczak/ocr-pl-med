# Integracja korektora z aplikacją (opcjonalny krok po OCR)

Korektor dostaje linie tekstu z Suryi i zwraca je poprawione: słowa spoza słownika zamienia na najbardziej
prawdopodobne słowo słownikowe, gdy model pomyłek OCR i częstość słowa dają wyraźną przewagę. Nie zmienia słów
słownikowych, liczb, skrótów ani terminów anatomicznych (lista TA `en,la,pl`). Działa za Suryą, na liniach,
bez trenowania w aplikacji: wszystko, czego się nauczył, jest w paczce `deploy/v4_200k_25k/` (6 KB).

## Dwa tryby

| tryb | co potrzebuje | CER na v1.2 (LOGO, 11 grup) | zamrożony test | harm |
|---|---|---|---|---|
| bez korekty | — | 0.1098 | 0.1040 | — |
| statyczny (`correct_lines`) | sam tekst | 0.0939 (−1.59 p.p.) | 0.0887 (−1.53) | 0.32% / 0.20% |
| z oceną Suryi (`correct_lines_rescored`) | tekst + obraz linii + model Surya z adapterem | 0.0858 (−2.40 p.p.) | 0.0784 (−2.56) | 0.40% / 0.31% |

Harm to odsetek słów poprawnych przed korektą, które korektor zepsuł. Szczegóły i zastrzeżenia: `results/v12/README.md`.
Wartości są dla Suryi `v4_200k_25k`: przy innym adapterze trzeba przebudować paczkę (niżej).

## Wymagania

- Python ≥ 3.10, `pip install -r requirements.txt` (rapidfuzz, numpy).
- Zasoby słownikowe (raz, ok. 70 MB pobrania, ok. 170 MB po rozpakowaniu): `python fetch_resources.py`;
  katalog z `$PP_CACHE` (domyślnie `~/.cache/ocr-pl-med/postprocessing`). PoliMorf: BSD 2-clause, NKJP 1-gramy: CC-BY.
- Po załadowaniu ok. 1.9 GB RAM, ładowanie ok. 23 s (raz na proces). Strona 30 linii: ok. 3.5 s na CPU.
  Korygować całą stronę jednym wywołaniem (kandydaci liczeni są partią).
- Tryb z oceną Suryi: ten sam proces co Surya (model już w pamięci), dodatkowo ok. 7 wariantów linii na linię
  w jednym przebiegu modelu bez dekodowania; na RTX 3090 ok. 100 wariantów/s. Na CPU/MPS wolniej, nie mierzone.

## Użycie w kodzie

```python
import sys; sys.path.insert(0, "training/ocr/postprocessing")
from pp.deploy import load

corr = load("training/ocr/postprocessing/deploy/v4_200k_25k")     # raz, przy starcie
corr.correct_lines(["WZGORZE - impulsy z", "receptonów ciała"])   # -> ["WZGÓRZE - impulsy z", "receptorów ciała"]
corr.correct_lines_detailed([...])  # -> [{"text": ..., "changes": [{"token": 0, "from": "WZGORZE", "to": "WZGÓRZE"}]}]
```

`changes` pozwala podświetlić w interfejsie słowa zmienione przez korektor (indeks tokenu po podziale po spacjach).

### Z oceną Suryi (w procesie, który już trzyma Suryę z adapterem)

```python
from pp.surya_score import SuryaScorer
scorer = SuryaScorer(recognition_predictor.foundation_predictor)   # ten sam FoundationPredictor z PeftModel
corr.correct_lines_rescored(lines, line_images, scorer)            # line_images: PIL, te same wycinki co do OCR
```

Surya ocenia każdego kandydata (top-5 na słowo) jako log P(linia z kandydatem | obraz), czyli patrzy na obraz,
czego model pomyłek nie robi. `SuryaScorer` liczy to tak samo jak strata w treningu LoRA (zgodność sprawdzona
z eksperymentem: mediana różnicy 5e-7, maks. 0.55 w 1% tekstów przez bf16 w partiach, zero zmian decyzji).

## Jako osobny serwis HTTP

```bash
cd training/ocr/postprocessing
PP_BUNDLE=deploy/v4_200k_25k PP_WORKER_PORT=5056 PP_PRELOAD=1 .venv/bin/python pp_worker.py
curl -s localhost:5056/health
curl -s -X POST localhost:5056/correct -d '{"lines": ["WZGORZE - impulsy z"], "detailed": true}'
```

Tylko tryb statyczny (serwis nie ma obrazu ani Suryi). Bez zależności poza modułem (biblioteka standardowa),
nasłuchuje na 127.0.0.1. Błąd korekty zwraca HTTP 500 z opisem, aplikacja powinna wtedy pokazać tekst z OCR.

## Przebudowa paczki dla nowego modelu OCR

Potrzebne predykcje nowego modelu na zbiorze z etykietami (np. benchmark na test/real/v1.2):

```bash
export PYTHONPATH=.
P=~/.cache/ocr-pl-med/postprocessing/preds-NOWY; C=~/.cache/ocr-pl-med/postprocessing/lp-NOWY   # poza repo (tekst linii)
.venv/bin/python -m evaluation.import_bench --run <benchmark/wyniki/run> --dataset-dir <dataset/test/real/v1.2> --out $P
# kandydaci -> serwer: log P z Suryi (GPU, kontener surya-training) -> z powrotem do $C/resC.jsonl
.venv/bin/python -m evaluation.rescore_cands --predictions $P/<model>.v12.jsonl --groups data/v12/groups.csv \
    --images-prefix benchmark/dane/real_v1.2/images/ --out $C
#   na serwerze: evaluation/surya_lp.py --adapter <adapter> --inp req.jsonl --out resC.jsonl (polecenie w nagłówku pliku)
.venv/bin/python -m evaluation.run_rescore $C --predictions $P/<model>.v12.jsonl --groups data/v12/groups.csv \
    --fixed-w 0 --deploy params_top5.json
.venv/bin/python -m evaluation.run_rescore $C --predictions $P/<model>.v12.jsonl --groups data/v12/groups.csv \
    --deploy params_surya.json
.venv/bin/python -m pp.deploy build --predictions $P/<model>.v12.jsonl --groups data/v12/groups.csv \
    --params params_top5.json --params-rescored params_surya.json --out deploy/<model>
```

Bez serwera da się zbudować tylko tryb statyczny z pełną listą kandydatów:
`python -m evaluation.run_cv --deploy --harm-max 0.003 ...` (−1.22 p.p. zamiast −1.59).

Paczka zawiera model pomyłek (pickle) i progi. Ładować tylko paczki zbudowane u nas: pickle wykonuje kod przy wczytaniu.

## Ograniczenia

- Zysk zmierzony na odręcznych notatkach anatomicznych. Na kartach pacjenta (141 linii) korektor prawie nic nie zmienia:
  liczby, nazwiska, daty i skróty dawkowania są poza jego zakresem. Na angielskich nazwach leków (RxHandBD) brak zysku.
- Korektor nie dodaje brakujących słów i nie łączy rozbitych; poprawia słowa po jednym.
- Lista TA chroni terminy łacińskie i angielskie; dla tekstów bez angielskiego można zbudować paczkę z `--domain la,pl`.
