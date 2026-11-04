# Sweep hiperparametrów LoRA Suryi (`v2_200k`)

Ablacja datasetów pokazała, że najlepsze dane to **`v2_200k`** (CER 0.127243 vs 0.129394 dla
`v2_800k`), a cały zmierzony efekt pochodzi z treści augmentacji — powyżej 200k próbek / 10k kroków
nie ma już mierzalnego zysku. Dane są więc na optimum i zostaje strojenie **samych
hiperparametrów treningu**, których nikt dotąd nie ruszał (receptura `lr 2e-5 / r64 / a64 / drop 0.2`
została odziedziczona po baselinie `ocr800k`).

## Runda 1a: learning rate × LoRA rank (9 runów)

Siatka (`run_hp_queue.sh`): `lr ∈ {5e-5, 1e-4, 2e-4}` × `rank ∈ {32, 64, 128}`, `alpha = 2·rank`,
`dropout 0.1`. Reszta przepisu bez zmian względem ablacji: 10 000 kroków, batch 8, wd 0.05,
cosine, warmup 0.1, bf16, target `q_proj v_proj o_proj`, eval co 2500 kroków. Dzięki temu różnice
w CER można przypisać zmienianym parametrom, a nie reszcie przepisu.

Dwie dźwignie są najbardziej obiecujące: **learning rate** (2e-5 jest konserwatywne dla LoRA —
standard PEFT to ~1e-4) i **LoRA rank** (pojemność adaptera). Jeden trial ≈ 25 min treningu
+ ~3 min ewaluacji, więc cała siatka ≈ 4,5 h z budżetu 12 h.

Punkt odniesienia: `v2_200k-lora-r64` = **CER 0.127243 / WER 0.479250 / EMA 0.262500**
(2160 linii, poziom „line”).

## Jak uruchomić

Na serwerze, w tmux (kolejkę odpalasz Ty, nie Claude):

```bash
tmux new -s hp
bash training/ocr/surya/hp_search/run_hp_queue.sh 2>&1 \
    | tee -a training/results/ocr/surya/hp/queue.log
```

Zmienne: `ONLY="hp_lr1e-4_r64"` (podzbiór), `REPORT_TO=wandb` (logowanie), `DUMP_PREDICTIONS=1`
(predykcje per linia do diagnozy). Kolejka jest **restartowalna**: trening jest pomijany, gdy
`adapter/` istnieje, a eval — gdy wynik już jest.

Wyniki: adaptery `training/results/ocr/surya/hp/<name>/adapter`, metryki
`training/results/ocr/surya/hp/results/<name>.json`, tabela na końcu kolejki
(`summarize.py`, można też puścić osobno):

```bash
docker compose -f training/docker-compose.yml run --rm surya-training \
    python training/ocr/surya/hp_search/summarize.py training/results/ocr/surya/hp/results
```

## Smoke-test parity (zrób PRZED pełną kolejką)

`eval_cer.py` liczy to samo co benchmark (`benchmark/docker/surya/app.py` + `benchmark/src/metrics.py`),
ale trzeba to potwierdzić na liczbach, bo od tego zależy sensowność porównań z baseline'em.
Na **istniejącym** adapterze baseline'owym:

```bash
docker compose -f training/docker-compose.yml run --rm surya-training \
    python training/ocr/surya/hp_search/eval_cer.py \
    --adapter training/results/ocr/surya/v2_200k-lora-r64/adapter \
    --output-json training/results/ocr/surya/hp/results/SMOKE-baseline.json
```

Oczekiwane (co do cyfry, tak jak zapisał benchmark): **CER 0.127243, WER 0.479250, EMA 0.262500**.
Zgodność = `eval_cer.py` powtarza predykcję i metryki benchmarku, można ufać 9 runom.

## Co jest odwzorowane 1:1

| Element | Źródło w benchmarku |
| --- | --- |
| budowa modelu (FoundationPredictor → PeftModel → RecognitionPredictor) | `docker/surya/app.py::_build_surya_state` |
| wywołanie predykcji (batch po 8, `ocr_without_boxes`, `math_mode=True`, `sort_lines=False`) | `app.py::predict_batch` + `predict` |
| czyszczenie tekstu (ucięcie po `<br>`, strip tagów, `html.unescape`) | `app.py::_clean_surya_text`, `_lines_to_text` |
| metryki (CER/WER/EMA/near-perfect/lev, mikro na sumach) | `src/metrics.py::HTRMetricsEvaluator` |
| poziom agregacji „line”, normalizacja `" ".join(text.strip().split())` | `src/metrics.py`, `orchestrator/benchmark.py` |

Sprawdzone lokalnie testem parity (metryki na 2004 parach, agregacja „line” z kolizjami kluczy,
`_extract_ids` na 2160 nazwach, czyszczenie tekstu względem źródła `app.py`) — wszystko zgodne.

`eval_cer.py` czyta `benchmark/dane/handlabeled/` wprost z bind-mountu (`..:/app`), więc **nie
potrzeba przebudowy obrazu** ani wpisu w `experiments.yaml` dla każdego wariantu.

## Wyniki rundy 1a (odpalonej 2026-09-16/17)

**Smoke parity zdał co do cyfry**: `SMOKE-baseline` = CER **0.127243** / WER 0.479250 / EMA 0.262500,
czyli dokładnie to, co zapisał benchmark `ocr-ablacja-v2`. Ścieżka ewaluacji (`eval_cer.py`) liczy
tak samo jak benchmark, więc liczby z kolejki są porównywalne z baseline'em.

**Żadna komórka siatki nie pobiła baseline'u** (CER 0.127243):

| run | CER | Δ vs baseline |
| --- | --- | --- |
| baseline `v2_200k-lora-r64` (lr 2e-5 / r64 / a64 / d0.2) | 0.127243 | — |
| `hp_lr5e-5_r64` (najlepsza) | 0.129512 | +0.23 p.p. |
| `hp_lr5e-5_r32` | 0.129725 | +0.25 p.p. |
| `hp_lr2e-4_r128` | 0.129796 | +0.26 p.p. |
| `hp_lr2e-4_r32` | 0.130434 | +0.32 p.p. |
| `hp_lr1e-4_r64` | 0.131781 | +0.45 p.p. |
| `hp_lr2e-4_r64` | 0.132608 | +0.54 p.p. |
| `hp_lr1e-4_r32` | 0.132679 | +0.54 p.p. |
| `hp_lr1e-4_r128` | 0.133247 | +0.60 p.p. |
| `hp_lr5e-5_r128` (najgorsza) | 0.133365 | +0.61 p.p. |

Trzy rzeczy każą czytać to ostrożnie:

1. **Cała siatka zmieniła naraz trzy parametry.** Każdy run 1a miał `alpha = 2·rank` (baseline
   `alpha = rank`) i `dropout 0.1` (baseline `0.2`) — więc degradacji **nie da się przypisać ani
   lr, ani rankowi**: równie dobrze odpowiada za nią alpha albo dropout. To był błąd projektu siatki.
2. **Różnice są w granicach szumu.** Trzy runy o tym samym lr, a różnych rankach to przy braku
   efektu ranku quasi-replikaty: rozrzut wewnątrz-lr daje pooled sd **0,157 p.p.**, czyli SE różnicy
   dwóch pojedynczych runów **0,22 p.p.** Najlepsza komórka (+0,23 p.p.) to ~1σ, średnia siatki
   (+0,42 p.p.) ~1,9σ. Rank nie ma przy tym żadnego monotonicznego efektu przy żadnym lr
   (przy lr 2e-4 najlepszy jest r128, przy 1e-4 — r64, przy 5e-5 — r64), co samo w sobie wygląda
   na szum. Bez replikatu z innym seedem nie wiemy, czy „baseline lepszy o 0,23 p.p." to sygnał.
3. **`eval_loss` nie tylko nie śledzi CER — jest z nim odwrotnie skorelowany.** Baseline ma przy
   10k kroków `eval_loss` **0,0417**, a wszystkie runy siatki **0,0147–0,0307** (2–3× niżej), mimo że
   baseline ma najlepszy CER. Przy ustalonym r64: lr 5e-5 → 1e-4 → 2e-4 obniża `eval_loss`
   0,0259 → 0,0202 → 0,0164 (−37%), a CER **rośnie** 0,129512 → 0,131781 → 0,132608. Walidacja jest
   z tego samego generatora co trening, więc niższy loss znaczy „lepiej wpasował się w syntetyczne
   artefakty", nie „lepiej czyta pismo". **Selekcja checkpointu po `eval_loss`**
   (`cli.py: metric_for_best_model="eval_loss"`) jest więc w tym zadaniu zawodna — w tej kolejce
   wyszło nieszkodliwie tylko dlatego, że wszystkie krzywe były monotoniczne i „najlepszy" wypadł
   na ostatnim kroku.

Przy okazji wyszło, że `cli.py` **nie seedował niczego przed założeniem LoRA** (`set_seed` odpalał
się dopiero w konstruktorze Trainera, czyli po inicjalizacji macierzy A), więc każdy dotychczasowy
run — łącznie z baseline'em — startował z innego losowego init. Commit z kolejką 1b to naprawia
(flaga `--seed`, domyślnie 42).

## Runda 1b: baseline + jedna zmiana na run

Skoro 1a zmieniła trzy parametry naraz, 1b zmienia **dokładnie jeden** względem baseline'u
(`lr 2e-5 / r64 / a64 / d0.2`), żeby różnica w CER miała jednoznaczne przypisanie:

| run | co zmienione vs baseline | po co |
| --- | --- | --- |
| `hp1b_seed1234` | seed 1234 | **miara szumu** — ten sam config, inny seed |
| `hp1b_d0.1` | dropout 0.1 | izoluje dropout (podejrzany nr 1 z 1a) |
| `hp1b_d0.3` | dropout 0.3 | strona przeciwna — jedyny strzał w górę po stronie regularyzacji |
| `hp1b_a128` | alpha 128 | izoluje `alpha = 2·rank` |
| `hp1b_lr5e-5` | lr 5e-5 | izoluje lr (najlepszy lr z 1a, ale z baseline'owymi alpha i dropoutem) |
| `hp1b_steps20k` | 20 000 kroków | jedyna nietknięta dźwignia (10k = 0,42 epoki, nigdy nie strojone) |

6 runów ≈ **3,1 h** (5 × ~28 min + 20k kroków ≈ 1 h). Razem z 1a ≈ 7,8 h z budżetu 12 h.
`hp1b_steps20k` to jedyny zakład w górę — można go pominąć (`ONLY=` bez tej nazwy), reszta to
diagnostyka, która rozstrzyga, czy 1a w ogóle coś zmierzyła. Uwaga: przy 20k kroków cosine rozciąga
się na cały run, więc to nie jest czyste „więcej kroków przy tym samym schedule".

Wiersze 1a zostają w `RUNS` (kolejka je pomija — adaptery istnieją), więc jeden przebieg skryptu
dolicza tylko 1b. Po zakończeniu `summarize.py` sam dopisze linię o szumie i powie, czy zwycięzca
jest poza nim.

Opcjonalnie (linia zakomentowana w `RUNS`): `hp1b_seed42` — baseline z seedem 42, czyli druga noga
pary replikatów w tym samym reżimie seedowania. Daje czystszy pomiar szumu niż porównanie
`hp1b_seed1234` ze starym baseline'em (trenowanym jeszcze bez seeda).
