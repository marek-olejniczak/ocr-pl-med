# Fine-tuning PARSeq (docTR) na liniach ocr_800k

Dotrenowanie **docTR PARSeq** (`Felix92/doctr-torch-parseq-multilingual-v1`) na tych
samych liniach ocr_800k co TrOCR i Surya, z rozszerzonym vocabem o polskie
diakrytyki + **spację** i z rozbudowanym logowaniem do W&B.

Dlaczego w ogóle PARSeq tu wyglądał „tragicznie" w benchmarku: to był **surowy
docTR** z domyślnym (francuskim) vocabem — bez polskich diakrytyków, model
word-level (bez spacji), a dostawał całe linie. Ten pipeline dotrenowuje go na
liniach z właściwym vocabem.

## Stack

- **Model**: docTR PARSeq (ViT-S backbone), multilingual base z HF hub
  `Felix92/doctr-torch-parseq-multilingual-v1` — ma już wszystkie polskie
  diakrytyki (`ąćęłńóśźż` + wersaliki), brakuje tylko spacji.
- **Input**: `--input-size HxW` (domyślnie tyle, co base: 32x128). Obrazy linii są
  **letterboxowane** (aspect-preserve, centrowane, białe tło) — jak w oficjalnym
  treningu docTR, nie squashowane.

  **Dlaczego warto podnieść szerokość:** linie w ocr_800k są bardzo szerokie
  (mediana 475x37 px, aspect 12.4:1), a letterbox do 32x128 zachowuje proporcje,
  więc wąskim gardłem jest szerokość: `scale = min(128/475, 32/37) = 0.27` i tekst
  zajmuje ~10 px wysokości, z 22 z 32 wierszy canvasu jako biały padding — zostaje
  ~7% pikseli linii (~5 px na znak przy medianie 26 znaków). Przy 32x512 jest to
  ~32 px wysokości (pełny canvas) i ~58% pikseli. Porównanie (te same 400 próbek):

  | input | tekst (mediana) | p10 | zachowane piksele | tokeny ViT |
  |---|---|---|---|---|
  | 32x128 | 10 px | 5 px | 7% | 129 |
  | 32x384 | 31 px | 15 px | 46% | 385 |
  | 32x512 | 32 px | 20 px | 58% | 513 |

  docTR sam pozycji **nie** interpoluje (`PatchEmbedding.interpolate=False` dla
  niekwadratowego patcha, a jego `interpolate_pos_encoding` zakłada siatkę
  kwadratową), więc przy zmianie rozmiaru robi to `_interpolate_positions` w
  `transfer_from_base` (bicubic, siatka patchy 8x16 -> 8x64, wiersz cls 1:1).
  Gdyby tego nie było, wagi pozycyjne ViT zostałyby **losowe**.

  Koszt: tokeny ViT 129 -> 513 (~4x), więc `--batch-size` trzeba zejść (64 -> 32).
- **Obraz bazowy**: `training-surya-training:latest` (torch 2.7.1+cu118, wandb)
  + `python-doctr==1.1.0` (od 1.1.0 torch/torchvision to twarde zależności, bez
  extra `[torch]`). Osobna pętla treningowa — docTR to nie transformers.

## Vocab / diakrytyki

`build_vocab(base_vocab, data_counts, ...)` = base + `" "` + znaki z danych o
częstości ≥ `--min-add-count` (domyślnie 2). Porządek base zachowany — rzędy
istniejących znaków nie zmieniają indeksów, więc `transfer_from_base` kopiuje je
1:1 po identyczności znaku, a rzędy specjalne `EOS/SOS/PAD` przenosi **wg roli**
(przesuwają się wraz z `len(vocab)`). `pos_queries` startuje ciepło z base
(pierwsze `base.max_length+1` rzędów).

DocTR odrzuca próbki z tekstem dłuższym niż `--max-label-length` **−2** (domyślnie
178; najdłuższa linia w danych ma 169 znaków): `encode_sequences` musi zmieścić
`[SOS | znaki | EOS]` w `target_size=max_label_length`, a przy długości `M−1`
EOS jest wypychany rolką i wewnętrzne maski PARSeq się rozjeżdżają (RuntimeError
`96 vs 95`). Próbki z nieznanym znakiem też są odrzucane (raport co pominął).

## Dane

Ten sam format co Surya/TrOCR (linie, nie słowa):

    training/data/processed/surya800k/{train,val}/metadata.jsonl   # {"file_name": ..., "text": ...}
    training/data/processed/surya800k/{train,val}/images/          # płaskie obrazy linii

## Logowanie W&B (nie tylko loss)

Własna pętla daje dostęp do `param.grad`, `optimizer.param_groups` i
`optimizer.state[param]`. Per `--log-every` (10) kroków:

| metryka | co pokazuje |
|---|---|
| `grad_norm_raw` / `grad_norm_clipped` | norma L2 gradientu przed i po clipie |
| `grad_cos_step` | cosinus kierunku gradientu vs poprzedni krok (oscylacje) |
| `adam_m1_norm` / `adam_m2_norm` | normy momentów AdamW (lub `momentum_norm` dla `--optim sgd`) |
| `weight_norm`, `weight_drift_rel`, `weight_cos_init` | dryf wag vs stan startowy |
| `update_norm` | faktyczna zmiana wag po kroku (w tym decay) |
| `train_loss`, `train_loss_ema`, `lr`, `epoch`, `samples_per_sec` | standard |
| co `--hist-every` (1000) | histogramy wag/gradientów wybranych warstw |
| co `--eval-every` (5000) | `val_loss`, `val_ema`, `val_near_perfect`, `val_cer`, przykłady GT vs pred |

## Użycie (serwer, tmux)

Build (raz, po zmianach w obrazie):

```bash
docker compose -f training/docker-compose.yml build parseq-training
```

Smoke (loss↓, checkpoint, wandb loguje grad_norm/cos/momentum):

```bash
docker compose -f training/docker-compose.yml run --rm parseq-training \
  python training/ocr/parseq/cli.py train \
  --base-model Felix92/doctr-torch-parseq-multilingual-v1 \
  --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
  --train-images-dir training/data/processed/surya800k/train/images \
  --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
  --val-images-dir training/data/processed/surya800k/val/images \
  --output-dir training/results/ocr/parseq/smoke \
  --limit-train 2048 --limit-val 512 \
  --max-steps 100 --eval-every 50 --save-every 0 --hist-every 10 --report-to wandb
```

Pełny trening (~40k kroków, bs 64, lr 3e-4, cosine + warmup 0.1; PARSeq mały
~24M, szybki):

```bash
docker compose -f training/docker-compose.yml run --rm parseq-training \
  python training/ocr/parseq/cli.py train \
  --base-model Felix92/doctr-torch-parseq-multilingual-v1 \
  --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
  --train-images-dir training/data/processed/surya800k/train/images \
  --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
  --val-images-dir training/data/processed/surya800k/val/images \
  --output-dir training/results/ocr/parseq/ocr800k-full \
  --max-steps 40000 --eval-every 5000 --report-to wandb --run-name parseq-ocr800k-full
```

Retrening z **szerszym wejściem** (32x512 — patrz tabela wyżej; osobny
`--output-dir`, żeby nie nadpisać modelu 32x128, bo oba idą do porównania w
benchmarku). `--batch-size` w dół, bo ViT ma 4x więcej tokenów:

```bash
docker compose -f training/docker-compose.yml run --rm parseq-training \
  python training/ocr/parseq/cli.py train \
  --base-model Felix92/doctr-torch-parseq-multilingual-v1 \
  --input-size 32x512 \
  --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
  --train-images-dir training/data/processed/surya800k/train/images \
  --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
  --val-images-dir training/data/processed/surya800k/val/images \
  --output-dir training/results/ocr/parseq/ocr800k-full-w512 \
  --batch-size 32 --max-steps 40000 --eval-every 5000 \
  --report-to wandb --run-name parseq-ocr800k-w512
```

W logu na starcie powinno pojawić się `Wagi pozycyjne ViT: interpolacja 8x16 -> 8x64`
oraz `input=(3, 32, 512)`.

## Checkpoint i wznowienie

Katalog wyjściowy (`--output-dir`):
- `model.pt` — czysty `state_dict` najlepszego modelu (nadpisywany co `--eval-every`,
  zapisany też przy końcu jako `last.pt`); jeśli ewaluacja wyłączona, `model.pt` = ostatni krok.
- `vocab.json` + `meta.json` (`input_shape`, `mean/std`, `max_label_length`, args, wersje) — pod
  benchmark Stage 4 i pod wznowienie/wariant lokalny `--base-model`.
- `trainer_state.pt` — pełny stan (model+optimizer+step) do `--resume-from`.

Wznowienie (np. po przerwie): ten sam `--output-dir`-based katalog z
`--resume-from training/results/ocr/parseq/ocr800k-full` i **większym** `--max-steps`
(liczba kroków to cel CAŁKOWITY, nie dodatkowy). W&B nazwa runu: podaj `--run-name`,
żeby nie mnożyć runów o tej samej nazwie po wznowieniu.

## Notatki serwerowe

- Serwer: partycja root `/var/lib/docker` ~99% pełna → **delta obrazu mała**, ale cache
  modeli HF i wandb idą do `/cache` (home hosta, duży dysk), a `HOME=/tmp`.
- Model bazowy (`Felix92/...`) pobiera się raz do `HF_HOME=/cache/hf` (vol
  `.../models/cache:/cache`) — dlatego `--cache-dir` domyślnie `None`, nie do repo.
- `--dataloader-num-workers 2`, `shm_size 8gb` w serwisie — zostało z TrOCR.
- Benchmark (Stage 4): wrapper `benchmark/modele/parseq_wrapper.py` będzie ładował lokalny
  katalog treningowy (ścieżka w kontenerze benchmarku) i czytał `vocab.json` + `meta.json`.

## Opcje godne uwagi

- `--optim sgd --momentum 0.9` — zamiast AdamW; wtedy W&B loguje `momentum_norm`.
- `--aug light` — drobny szum geometryczny/fotometryczny (obrót ±1.5°, jitter jasności).
- `--min-add-count` — próg częstości, od którego znak z danych dochodzi do vocabu
  (za niski = vocab puchnie od literówek w syntetykach).
- `--extra-vocab` — ręcznie dołóż znaki, np. `'„”–'`, których może brakować w danych.
