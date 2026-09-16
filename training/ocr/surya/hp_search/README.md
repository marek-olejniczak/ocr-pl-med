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

## Runda 1b (jeśli zostanie budżet)

Wokół najlepszej komórki z 1a: `dropout {0.1, 0.2}` × `alpha/rank {1, 2}`; jeśli wygra `lr 2e-4`,
dorzucić `3e-4`. Round 1a + 1b ≈ 6,5 h, czyli w budżecie 12 h zostaje zapas na wpadki.
