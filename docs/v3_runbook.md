# Pilot generatora v3: jak uruchomić

Co i dlaczego: `docs/ocr_generator_v3.md`. Dane są już wygenerowane i wysłane
na DagsHub, a wskaźnik `dataset.dvc` i skrypty są na gałęzi `server`.
Zostaje serwer.

## Serwer, jedna sesja

```bash
tmux new -s v3
cd ~/ocr-pl-med
git checkout server && git pull
source .venv/bin/activate
dvc remote modify --local origin verify true

# tylko nowe zbiory, nie cały dataset/
dvc pull dataset/v3_base_200k dataset/v3_200k dataset/v3_no_cursive_joins_200k \
  dataset/v3_no_glyph_variants_200k dataset/v3_no_diacritic_placement_200k

# 7 treningów, potem benchmark 8 modeli na prawdziwych notatkach
bash training/ocr/surya/ablation/run_v3.sh
```

`Ctrl+b`, potem `d` odłącza. Czas: około 3 h treningów (7 × ~25 min) i
~40 min benchmarku. Kolejka pomija gotowe adaptery, więc po przerwie wystarczy
odpalić `run_v3.sh` jeszcze raz. Przed każdym treningiem sprawdza shardy
i liczbę próbek po konwersji.

Kontrola z zewnątrz:

```bash
grep "^==" ~/ocr-pl-med/training/results/ocr/surya/ablation_queue.log | tail
ls ~/ocr-pl-med/benchmark/wyniki/
```

## Wyniki na laptopa

```powershell
scp -r "studenci-pp-2026-1-ocr@172.16.16.1:~/ocr-pl-med/benchmark/wyniki/ocr-v3-pilot_*" "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\"
```

Analizę (CER na części roboczej, pomyłki `ci→u`, `k→h`, `ś→s` …, szum między
seedami) robi Claude z tych plików.

## Podział testsetu (do treningów Marka)

Lista linii części roboczej i zamrożonej jest w `benchmark/splits/`. Na
serwerze jedno polecenie robi z niej pliki dla `eval_cer.py` / callbacku CER:

```bash
python benchmark/splits/make_split.py
# -> benchmark/dane/handlabeled/labels_dev.csv (1212), labels_test.csv (948)
```

W treningu: `--cer-labels-csv benchmark/dane/handlabeled/labels_dev.csv`.
