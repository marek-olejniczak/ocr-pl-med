# Fine-tuning TrOCR (VisionEncoderDecoderModel) — polski wariant

Dotrenowanie **`PET3R12/trocr-base-polish-handwriting`** (czerwiec 2026, MIT,
~334M param.) na **tych samych danych** co Surya: `ocr_800k`
(`training/data/processed/surya800k/`). Cel: porównanie w benchmarku na handlabeled
testsecie — Surya base / Surya+LoRA / TrOCR base / **TrOCR po naszym treningu** / RysOCR.

Różnica vs trening Suryi:
- Surya = **LoRA** (PEFT, `task_type=None`) — zostawiamy bazę, trenujemy adaptery.
- TrOCR = **pełny fine-tuning** (`Seq2SeqTrainer`) — dekoder TrOCR ma angielski
  słownik (RoBERTa vocab 50265), który kaleczy polskie znaki; pełny trening nadpisuje
  wagi dekodera i uczy polskich transkrypcji. Checkpoint = zwykły katalog (bez peft),
  bezpośrednio czytelny przez benchmark jako `model_id`.

## Stack

| Komponent | Wersja | Uwagi |
|---|---|---|
| transformers | **5.0.0** | polski TrOCR zapisany z 5.0.0 (`processor_config.json`); obraz Suryi (4.56.1) go nie wczyta |
| torch | 2.7.1 (cu118) | z obrazu `training-surya-training:latest` |
| model | `PET3R12/trocr-base-polish-handwriting` | encoder ViT 384, decoder TrOCR d_model 1024/12 warstw, vocab 50265 |

## Dane

Bez nowej konwersji — ten sam format co Surya (`metadata.jsonl` + płaskie obrazy linii):

```
training/data/processed/surya800k/train/metadata.jsonl  # 760 066 próbek
training/data/processed/surya800k/val/metadata.jsonl    #  39 934 próbek
```

## Użycie

Build obrazu i trening odpala się z repo (`training/docker-compose.yml`):

```bash
# 1. Build
docker compose -f training/docker-compose.yml build trocr-training

# 2. Smoke test (100 kroków) — sprawdza ładowanie modelu, loss, checkpoint, wandb
docker compose -f training/docker-compose.yml run --rm trocr-training \
  python training/ocr/trocr/cli.py train \
  --model-id PET3R12/trocr-base-polish-handwriting \
  --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
  --train-images-dir training/data/processed/surya800k/train/images \
  --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
  --val-images-dir training/data/processed/surya800k/val/images \
  --output-dir training/results/ocr/trocr/smoke \
  --max-steps 100 --report-to wandb

# 3. Pełny trening (40k kroków @ bs 8 ≈ 0.42 epoki — mirror Suryi r64)
docker compose -f training/docker-compose.yml run --rm trocr-training \
  python training/ocr/trocr/cli.py train \
  --model-id PET3R12/trocr-base-polish-handwriting \
  --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
  --train-images-dir training/data/processed/surya800k/train/images \
  --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
  --val-images-dir training/data/processed/surya800k/val/images \
  --output-dir training/results/ocr/trocr/ocr800k-full \
  --max-steps 40000 --report-to wandb
```

Po treningu pełny checkpoint (model + processor + `meta.json`) ląduje w
`training/results/ocr/trocr/<run>/` — ten katalog wskazuje się w benchmarku jako
`model_id` (`trocr_polish_ft`).

## Hiperparametry (mirror Suryi r64)

`lr 2e-5`, `bs 8`, `grad_accum 1`, `40k kroków`, `bf16`, `cosine`, `warmup 0.1`,
`eval_steps 5000` (loss + generowanie), `save_total_limit 3`, `load_best_model_at_end`.
Flaga `--max-steps` pozwala robić smoke testy.

## Cache modelu

Domyślnie `--cache-dir modele/cache/trocr` — ten sam katalog, którego używa
benchmark (`benchmark/docker/trocr`), więc model pobrany przy treningu jest od razu
dostępny w benchmarku (bez ponownego downloadu).

## Serwer — uwagi

- Trening odpala **Ty** w tmux (`docker compose -f training/docker-compose.yml run --rm trocr-training ...`).
- Obraz bazuje na `training-surya-training:latest` (już na serwerze) — delta mała,
  bezpieczna przy 99% pełnej partycji root `/var/lib/docker`.
- W&B: `WANDB_API_KEY` z `.env` repo (serwis `trocr-training` ma `WANDB_ENTITY=ocr-pl-med`,
  `WANDB_PROJECT=ocr-finetune`). Logujemy krzywe treningu; **nie** uploadujemy modelu jako
  artefaktu W&B (checkpoint ~1.3 GB zostaje na dysku, benchmark czyta go z
  `training/results/ocr/trocr/<run>/`).
