# Analiza pomyłek znaków OCR (handlabeled)

Diagnostyka *jakie znaki i ciągi znaków myli model* na ręcznie olabelowanym zbiorze
`benchmark/dane/handlabeled` (2160 linii). Wyniki zasiliły wnioski do danych v3:
`notes/vault/Generator_v3_wskazowki.md`.

Analizowany model: Surya LoRA r64 / `v2_200k` (best checkpoint, `checkpoint-7908`),
CER **0.1277** · WER 0.4817 · EMA 0.2528. Analizy działają na dowolnym `predictions.jsonl`
(`{"file_name", "ground_truth", "prediction"}`), więc można je puścić na innym modelu.

## Pliki

| plik | co robi |
|---|---|
| `infer.sh` | GPU, kontener `surya-training`: `hp_search/eval_cer.py --dump-predictions` → `data/` (metryki 1:1 z benchmarkiem) |
| `analyze.py` | pomyłki 1:1: Levenshtein, macierz `P(pred \| ref)`, heatmapy, panel diakrytyków, wycinki → `out/` |
| `analyze_ngram.py` | pomyłki wieloznakowe 1:2 / 2:1 / 2:2 (wyrównanie ważone, koszty hard-EM), lift, koncentracja per źródło/słowo, oczyszczona macierz 1:1 → `out_ngram/` |
| `diacritics.py` | `model-input`: wejście modelu po przetwarzaniu Surya + gubienie diakrytyków vs rozdzielczość; `sample`: losowa próbka linii do ręcznej oceny |
| `run_all.sh` | wszystko po kolei |
| `results/` | zagregowany snapshot wyników (RAPORT.md + CSV, bez tekstu/obrazów linii) |

## Uruchomienie

```bash
python -m venv .venv && .venv/bin/pip install -r training/ocr/error_analysis/requirements.txt

# cały pipeline (inferencja GPU + analizy CPU)
PYTHON=.venv/bin/python bash training/ocr/error_analysis/run_all.sh

# inny checkpoint / GPU
ADAPTER=training/results/ocr/surya/<run>/checkpoint-N GPU=1 bash training/ocr/error_analysis/infer.sh

# same analizy na istniejących predykcjach
PYTHON=.venv/bin/python SKIP_INFER=1 bash training/ocr/error_analysis/run_all.sh
cd training/ocr/error_analysis && ../../../.venv/bin/python diacritics.py sample --char ż
```

`data/`, `out/`, `out_ngram/` są w `.gitignore`: zawierają etykiety testsetu i wycinki
linii (testset jest poza gitem). Do repo trafia tylko `results/`.

## Metoda

**`analyze.py`**: normalizacja jak w `eval_cer.py` (`" ".join(t.split())`) + NFC;
Levenshtein (sub=del=ins=1) z backtrace; do macierzy idą tylko substytucje,
usunięcia/wstawienia osobno; intensywność = `P(pred | ref)` bez przekątnej.

**`analyze_ngram.py`**: zwykły Levenshtein rozbija np. `cz → w` na `c→w` + usunięcie `z`
i zanieczyszcza macierz 1:1. Tutaj:
1. kandydaci = sąsiadujące pary edycji z Levenshteina (sub+ins → 1:2, sub+del → 2:1, sub+sub → 2:2),
2. Viterbi z operacjami 1:1, del, ins, 1:2, 2:1, 2:2; koszt = −log P(pred | ref-jednostka),
   uczony hard-EM (1:1 wygładzane α=0.1; wieloznakowe N / wystąpienia źródła w GT + kara λ=0.5;
   dopuszczone tylko N ≥ 3); zbieżność po ~5 iteracjach,
3. raport: N ≥ 5 i lift ≥ 3, gdzie lift = P(zdarzenia) / P(najlepszej dekompozycji na niezależne 1:1/indel),
4. wykluczone zdarzenia trywialne (np. `rz→r` = zwykłe usunięcie) i obejmujące spację,
5. koncentracja: udział jednego źródła (zeszyt z nazwy pliku, proxy autora) i jednego słowa.

**`diacritics.py`**: odtwarza przetwarzanie wejścia Surya 0.17.1 (`scale_to_fit` dla
`img_size=(1024, 512)` + zaokrąglenie do wielokrotności 28 = patch 14 × merge 2).
Kod skalowania skopiowany, bo kontener nie ma matplotlib.

## Najważniejsze wyniki (snapshot w `results/`)

- **Pomyłki 1:1** (Levenshtein): 4014 substytucji + 914 usunięć + 474 wstawień = 5402 edycji;
  top: `k→h` 448, `a→o` 244, `i→u` 123, `ś→s` 108, `t→ł` 102.
- **Wieloznakowe**: 609 zdarzeń, wszystkie optyczne, **0 ortograficznych** (`rz↔ż`, `ch↔h`, `ó↔u` …).
  Top: `ci→u` 94, `cz→w` 87, `rz→n` 65, `ci→w` 47, `oc→a` 26, `ls→b` 23 (55% wystąpień „ls”),
  `k→lu` 22, `d→ol` 16. Większość rozłożona na 5–8 z 8 źródeł, więc to ogólne słabości modelu.
- **Macierz 1:1 była zniekształcona**: po oczyszczeniu `z→w` −92%, `z→n` −94%, `i→w` −91%,
  `i→u` −84%, `c→s` −100% (to połówki zdarzeń wieloznakowych). Prawdziwe 1:1: `k→h` 439
  (29% wszystkich „k”), `a→o` 244, `ś→s` 143 (37% „ś”), `ż→z` 29% „ż”.
- **Wielkość liter**: tokenizer Surya (`GreedyMathUTF16Tokenizer`) rozróżnia `A`/`a`;
  pomyłki tylko w wielkości liter to 2.6% edycji (CER bez rozróżniania 0.1243).
- **Diakrytyki a rozdzielczość**: żadna linia nie jest zmniejszana (wysokość: mediana 44 px,
  79% linii jest powiększanych). Gubienie diakrytyków nie zależy od wysokości (8–15%, tak jak
  zwykłe litery). W ręcznie przejrzanej próbce duża część `ś→s`/`ż→z` to linie, w których autor
  **nie postawił** diakrytyku, a etykieta ma poprawną ortografię.

## Zastrzeżenia

- `handlabeled` to ten sam zbiór, po którym wybierano checkpoint (callback `eval_cer`), więc
  analiza jest in-sample względem selekcji.
- Rzadkie znaki (Ą 13, Ć 4, Ź 5 wystąpień) mają niestabilne statystyki.
- „Źródło” to zeszyt/dokument z nazwy pliku, nie identyfikator autora.
- α, λ i progi raportowania (N ≥ 5, lift ≥ 3) są dobrane ręcznie; ranking jest stabilny
  między iteracjami EM.
- Ocena próbek diakrytyków jest wizualna i ręczna.
