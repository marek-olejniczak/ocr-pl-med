# Ablacja, runda 2: każda augmentacja osobno

Dwa pytania o każdą augmentację z osobna:

1. **Czy przeszkadza modelowi czytać?** Odpowiada zbiór kontrolny `robustness`:
   1500 linii, każda narysowana w 31 wersjach (czysta, plus każda augmentacja
   osobno na 1–3 poziomach siły). Ten sam tekst, ręka i tusz w każdej wersji,
   więc różnica w CER to koszt samej augmentacji. Bez treningu, tylko odczyt.
2. **Czy pomaga w treningu?** Odpowiada 9 nowych treningów, każdy bez jednej
   rodziny augmentacji, porównanych z `v2_200k` na prawdziwych notatkach.

Przy okazji `v2_val10k`: 10 000 linii walidacyjnych z `v2_800k` z pełną
metryką, co daje CER osobno dla każdego kroju pisma i rodzaju treści.

## Co powstaje

| Zbiór | Co to jest | Rozmiar |
|---|---|---|
| `robustness` | zbiór kontrolny, 1500 linii × 31 warunków | 46 500 obrazów |
| `v2_val10k` | podzbiór walidacyjny `v2_800k` | 10 000 obrazów |
| `no_short_words_200k` … `no_phone_photo_200k` | 9 zbiorów treningowych bez jednej rodziny | 9 × 200 000 linii |

Wszystkie zbiory treningowe mają seed 2030 jak `v2_200k`. Generator po
dodaniu zapisu parametrów odtwarza `v2_200k` linia w linię (tekst i metryka
300/300, obrazy bez albumentations bajt w bajt), więc `v2_200k` pozostaje
ważnym punktem odniesienia i nie trzeba go powtarzać.

## A. Laptop

Generacja i pakowanie idą jednym poleceniem (log: `output/runda2_generacja.log`).
Po nich wysyłka:

```powershell
cd C:\Users\tomek\Desktop\dane\inzynierka
dvc add dataset
git add dataset.dvc
git commit -m "Runda 2 ablacji: robustness, v2_val10k, 9 zbiorow bez jednej rodziny"
dvc push
git push
```

Potem wskaźnik `dataset.dvc` trzeba przenieść na gałąź `server` w repo
`ocr-pl-med`, tak jak przy pierwszej rundzie.

## B. Serwer, jedna sesja

```bash
tmux new -s runda2
cd ~/ocr-pl-med
git checkout server && git pull
source .venv/bin/activate

# DVC domyślnie nie sprawdza sum po pobraniu; w pierwszej rundzie przez to
# przeszły ucięte archiwa. Z tym ustawieniem ucięty plik zostanie wykryty.
dvc remote modify --local origin verify true

# dane: tylko nowe katalogi, nie cały dataset/
dvc pull dataset/robustness dataset/v2_val10k \
  dataset/no_short_words_200k dataset/no_caps_200k dataset/no_arrows_bullets_200k \
  dataset/no_anatomy_vocab_200k dataset/no_neighbour_glyphs_200k dataset/no_grid_paper_200k \
  dataset/no_morphology_200k dataset/no_elastic_200k dataset/no_phone_photo_200k

# zbiory kontrolne do katalogu benchmarku, z kontrolą kompletności
bash training/ocr/surya/ablation/prepare_eval_data.sh
```

Potem wszystko po kolei jednym poleceniem. Każdy etap ma własny log, a awaria
jednego nie zatrzymuje następnych:

```bash
( cd benchmark && python -u autorunner.py --config ../training/ocr/surya/ablation/experiments_odpornosc.yaml 2>&1 | tee wyniki/runda2_odpornosc.log ) ; \
bash training/ocr/surya/ablation/run_ablation_queue.sh 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log ; \
( cd benchmark && python -u autorunner.py --config ../training/ocr/surya/ablation/experiments_rodziny.yaml 2>&1 | tee wyniki/runda2_rodziny.log )
```

`Ctrl+b`, potem `d` odłącza. Czas łącznie około 7 godzin:

| Etap | Czas |
|---|---|
| odczyt zbiorów kontrolnych przez v1 i v2 | ~2,5 h |
| 9 treningów po ~25 min | ~4 h |
| benchmark 10 modeli na prawdziwych notatkach | ~45 min |

Kolejka treningów pomija modele, które już mają adapter, więc trenuje tylko
9 nowych. Przed każdym treningiem sprawdza, czy shardy nie są ucięte i czy
po konwersji jest tyle próbek, ile w etykietach.

Kontrola z zewnątrz:

```bash
grep "^==" ~/ocr-pl-med/training/results/ocr/surya/ablation_queue.log | tail
ls ~/ocr-pl-med/benchmark/wyniki/
nvidia-smi
```

### Opcjonalnie: czy model bez danej augmentacji jest na nią mniej odporny

Te same 10 modeli na pierwszych 300 liniach zbioru kontrolnego, około 2 godzin:

```bash
( cd benchmark && python -u autorunner.py --config ../training/ocr/surya/ablation/experiments_rodziny_odpornosc.yaml 2>&1 | tee wyniki/runda2_rodziny_odpornosc.log )
```

## C. Wyniki na laptopa

```powershell
scp -r "studenci-pp-2026-1-ocr@172.16.16.1:~/ocr-pl-med/benchmark/wyniki/ocr-odpornosc_*" "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\"
scp -r "studenci-pp-2026-1-ocr@172.16.16.1:~/ocr-pl-med/benchmark/wyniki/ocr-ablacja-rodziny_*" "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\"
```

## D. Raporty

```powershell
cd C:\Users\tomek\Desktop\text-gen
$W = "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\ocr-odpornosc_<data>"
python src/robustness_report.py robustness --labels output/robustness/labels.jsonl `
  "v1=$W\surya_lora_ocr800k\robustness\raw_predictions.csv" `
  "v2=$W\surya_lora_v2_800k\robustness\raw_predictions.csv" --output docs\odpornosc.md
python src/robustness_report.py val --labels output/v2_val10k/labels.jsonl `
  "v1=$W\surya_lora_ocr800k\v2_val10k\raw_predictions.csv" `
  "v2=$W\surya_lora_v2_800k\v2_val10k\raw_predictions.csv" --output docs\val_breakdown.md

$R = "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\ocr-ablacja-rodziny_<data>"
python src/ablation_report.py "v2_200k=$R\surya_lora_v2_200k\handlabeled\raw_predictions.csv" `
  "bez_short_words=$R\surya_lora_no_short_words_200k\handlabeled\raw_predictions.csv" `
  "bez_caps=$R\surya_lora_no_caps_200k\handlabeled\raw_predictions.csv" `
  "bez_arrows_bullets=$R\surya_lora_no_arrows_bullets_200k\handlabeled\raw_predictions.csv" `
  "bez_anatomy_vocab=$R\surya_lora_no_anatomy_vocab_200k\handlabeled\raw_predictions.csv" `
  "bez_neighbour_glyphs=$R\surya_lora_no_neighbour_glyphs_200k\handlabeled\raw_predictions.csv" `
  "bez_grid_paper=$R\surya_lora_no_grid_paper_200k\handlabeled\raw_predictions.csv" `
  "bez_morphology=$R\surya_lora_no_morphology_200k\handlabeled\raw_predictions.csv" `
  "bez_elastic=$R\surya_lora_no_elastic_200k\handlabeled\raw_predictions.csv" `
  "bez_phone_photo=$R\surya_lora_no_phone_photo_200k\handlabeled\raw_predictions.csv" `
  --output docs\wyniki_ablacji_rodziny.md
```

## Warunki zbioru kontrolnego

Poziomy 1–3 to łagodny koniec, środek i ostry koniec zakresu, z którego
losuje generator treningowy.

| Rodzina | Warunki |
|---|---|
| sąsiednie linie | `neighbours_s1..s3`: oba sąsiedzi, wcięcie 12% / 28% / 45% rozmiaru fontu |
| kratka | `grid_s1..s3`: krycie 0,45 / 0,72 / 1,0; `ruled_s3`: same linie |
| morfologia | `erosion_s1..s3`: domieszka 0,5 / 0,75 / 1,0; `dilation_s1`, `dilation_s3`: 1 i 2 px |
| zniekształcenie | `elastic_s1..s3`: alfa 0,6 / 1,1 / 1,6 wysokości; `griddist_s1..s3`: limit 0,08 / 0,15 / 0,22 |
| telefon | `phone_s1..s3`: wszystkie składowe razem; `phone_shadow`, `_noise`, `_blur`, `_motion`, `_downscale`, `_jpeg`: każda składowa osobno na najostrzejszym poziomie |
| skaner (v1) | `scan_clean_color`, `scan_grayscale`, `scan_photocopy` |

## Uwaga o ucinaniu plików (sprawdzone 2026-10-07)

DagsHub przechowuje pełne archiwa, także te po 1,4 GB z pierwszej rundy
(rozmiar zdalny równy lokalnemu co do bajta). Pliki ucinały się przy
pobieraniu na serwer, w okolicy 1,09 GB, a DVC tego nie zauważył, bo bez
`verify true` nie liczy sum po pobraniu. Od rundy 2 archiwa mają najwyżej
0,9 GiB (`shard_for_dvc.py` pilnuje rzeczywistego rozmiaru przy każdym
pliku), a kolejka i `prepare_eval_data.sh` dodatkowo sprawdzają, czy każde
archiwum otwiera się do końca.
