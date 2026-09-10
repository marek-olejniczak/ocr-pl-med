# Ablacja OCR — komendy krok po kroku

Plan: `docs/plan.md` (faza 2 + 3). Co jest w generatorze v2: `docs/ocr_generator_v2.md`.
Wszystko poniżej odpala człowiek; nic nie startuje samo.

## Co powstaje

| Zbiór | Linie | Seed | Wyłączone | Trening | Po co |
|---|---|---|---|---|---|
| `v2_800k` | 800 000 | 2029 | nic | 40 000 kroków (~1 dzień) | model właściwy, porównanie z `ocr_800k` (15,9% CER) |
| `v2_200k` | 200 000 | 2030 | nic | 10 000 kroków (~6 h) | punkt odniesienia ablacji |
| `noA_200k` | 200 000 | 2030 | A: `short_words,caps,arrows_bullets,anatomy_vocab` | 10 000 | wkład treści |
| `noB_200k` | 200 000 | 2030 | B: `neighbour_glyphs,grid_paper` | 10 000 | wkład kadru i tła |
| `noC_200k` | 200 000 | 2030 | C: `morphology,elastic,phone_photo` | 10 000 | wkład degradacji obrazu |

Baseline v1 = istniejący model `surya_lora_ocr800k`, nie trenujemy go ponownie.
Cztery zbiory 200k mają ten sam seed, różnią się tylko wyłączoną grupą.
Hiperparametry treningu identyczne z baseline (r=64, α=64, dropout 0,2,
q/v/o_proj, lr 2e-5, wd 0,05, cosine, warmup 0,1, bf16, batch 8).

Serwer: 1 dzień na `v2_800k` + 1 dzień na cztery 200k. Można rozdzielić:
najpierw tylko `v2_800k` (faza 2), ablację dopiero gdy wynik jest lepszy niż 15,9%.

---

## A. Laptop (Windows, PowerShell) — generacja i pakowanie

Generator zapisuje etykiety na bieżąco, więc przerwany run nie psuje danych;
skrypt pomija warianty, które już mają `dataset_card.json`, więc po przerwie
wystarczy odpalić go ponownie.

```powershell
cd C:\Users\tomek\Desktop\text-gen

# 1. Wszystkie pięć zbiorów po kolei (~3 h dla 800k + ~3 h dla 4x200k).
#    Osobne okno PowerShell, nie zamykać laptopa (uśpienie przerywa).
python src/build_ablation_sets.py generate

#    Albo tylko faza 2 na początek:
python src/build_ablation_sets.py generate --only v2_800k

# 2. Pakowanie do repo DVC: shardy po 1 GB + labels + karta zbioru
python src/build_ablation_sets.py pack --dvc-repo "C:\Users\tomek\Desktop\dane\inzynierka"

# 3. Wysyłka na DagsHub
cd C:\Users\tomek\Desktop\dane\inzynierka
dvc add dataset
git add dataset.dvc
git commit -m "OCR v2 + zbiory ablacji (v2_800k, v2_200k, noA/noB/noC_200k)"
dvc push
git push
```

Żeby generacja przeżyła zamknięcie okna, można ją odpalić jako proces w tle
z logiem:

```powershell
Start-Process python -ArgumentList "src/build_ablation_sets.py generate" `
  -RedirectStandardOutput output\ablation_gen.log -RedirectStandardError output\ablation_gen.err -NoNewWindow
Get-Content output\ablation_gen.log -Tail 5 -Wait     # podgląd
```

---

## B. Serwer RTX 3090 (Linux) — tmux, dane, treningi

### B.1 tmux — żeby trening przeżył rozłączenie SSH

```bash
# instalacja (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y tmux
# bez sudo: conda install -c conda-forge tmux   albo   pip install --user tmuxp (nie to samo; wolimy apt)

tmux new -s ablacja          # nowa sesja o nazwie "ablacja"
# ... odpalasz komendy w środku ...
# Ctrl+b, potem d             -> odłączenie (sesja żyje dalej)
tmux attach -t ablacja       # powrót po ponownym zalogowaniu
tmux ls                      # lista sesji
# Ctrl+b, potem [             -> przewijanie w górę (q kończy)
tmux kill-session -t ablacja # gdy wszystko skończone
```

### B.2 Kod i dane

```bash
cd ~/ocr-pl-med            # albo gdzie leży checkout
git fetch && git checkout server && git pull          # jest tu: poprawka <br> w Surya, kolejka ablacji

# dane: katalog dataset/<nazwa>/ z images_shards/ i labels.jsonl
# jeśli dataset/ to klon repo DagsHub:
cd dataset_repo && git pull && dvc pull dataset.dvc && cd ..
ls dataset/v2_800k/images_shards | head            # sprawdzenie
```

Jeżeli `dataset/` w checkout'cie ocr-pl-med to inny katalog, ustaw
`DATA_ROOT=/sciezka/do/dataset` przy uruchamianiu kolejki.

### B.3 Kolejka treningów (w tmux)

```bash
tmux new -s ablacja
cd ~/ocr-pl-med
mkdir -p training/results/ocr/surya

# faza 2: tylko model właściwy (1 dzień)
ONLY="v2_800k" bash training/ocr/surya/ablation/run_ablation_queue.sh 2>&1 \
  | tee -a training/results/ocr/surya/ablation_queue.log

# faza 3: cztery ablacje (razem ~1 dzień)
ONLY="v2_200k noA_200k noB_200k noC_200k" bash training/ocr/surya/ablation/run_ablation_queue.sh 2>&1 \
  | tee -a training/results/ocr/surya/ablation_queue.log

# wszystko naraz (2 dni):
bash training/ocr/surya/ablation/run_ablation_queue.sh 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log
```

Skrypt sam robi konwersję (`convert_data.py ocr800k`) i trening (`cli.py`),
pomija runy z gotowym `adapter/`, a padnięty run nie zatrzymuje kolejki.
`REPORT_TO=wandb` włącza W&B (wymaga `WANDB_API_KEY`).

Podgląd z drugiego terminala:

```bash
tail -f ~/ocr-pl-med/training/results/ocr/surya/ablation_queue.log
nvidia-smi
```

Wyniki: `training/results/ocr/surya/<nazwa>-lora-r64/adapter/`.

### B.4 Benchmark

1. Do `benchmark/experiments.yaml` w sekcji `models:` wklej wpisy z
   `training/ocr/surya/ablation/experiments_ablation.yaml` (jeden na adapter).
   Zostaw włączony `surya_lora_ocr800k` (baseline) i `surya` (baza).
2. Zmień `experiment.name` na np. `ocr-ablacja-v2`.
3. W tmux:

```bash
cd ~/ocr-pl-med/benchmark
python autorunner.py 2>&1 | tee wyniki/ablacja_autorunner.log
```

Wyniki: `benchmark/wyniki/<experiment.name>_<data>/<model>/handlabeled/raw_predictions.csv`.

### B.5 Kopia wyników na laptop

```powershell
scp -r <login>@172.16.16.1:~/ocr-pl-med/benchmark/wyniki/ocr-ablacja-v2_* "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\"
```

---

## C. Laptop — raport

```powershell
cd C:\Users\tomek\Desktop\text-gen
$W = "C:\Users\tomek\Desktop\ocr-main\benchmark\wyniki\ocr-ablacja-v2_<data>"
python src/ablation_report.py `
  "baseline_v1=$W\surya_lora_ocr800k\handlabeled\raw_predictions.csv" `
  "v2_800k=$W\surya_lora_v2_800k\handlabeled\raw_predictions.csv" `
  "v2_200k=$W\surya_lora_v2_200k\handlabeled\raw_predictions.csv" `
  "noA=$W\surya_lora_noA_200k\handlabeled\raw_predictions.csv" `
  "noB=$W\surya_lora_noB_200k\handlabeled\raw_predictions.csv" `
  "noC=$W\surya_lora_noC_200k\handlabeled\raw_predictions.csv" `
  --output docs\wyniki_ablacji.md
```

Raport: CER z 95% przedziałem bootstrap po liniach, CER po autorach, po
długości GT, najczęstsze podmiany. Wkład grupy = CER(bez grupy) − CER(v2_200k).
Różnica liczy się, gdy przedziały ufności się nie nakładają; efekt widoczny
u wszystkich autorów jest wiarygodny nawet przy małej średniej.

## Decyzje po drodze

- `v2_800k` gorszy lub równy 15,9% → nie odpalać ablacji; analiza błędów
  nowego modelu tym samym raportem i poprawki generatora.
- Grupa C znacząca, a promotor pyta o elastic → jeden dodatkowy run
  `--disable elastic` (6 h), ręcznie przez `generate_ocr_lines.py`.
