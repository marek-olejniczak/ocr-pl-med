"""Fine-tuning CLI dla PARSeq (docTR, wariant multilingual) na liniach ocr_800k.

Cel: dotrenować `Felix92/doctr-torch-parseq-multilingual-v1` (docTR PARSeq —
word-level, 32x128, vocab zawiera już polskie diakrytyki `ąćęłńóśźżĄĆĘŁŃŚŹŻ`) na
liniach ocr_800k, tych samych co Surya/TrOCR. Wymaga rozszerzenia vocabu o SPACJĘ
(linie, nie słowa) i podbicia `max_label_length` (~96 vs domyślne 32), bo model
docTR PARSeq sam nie wie, jak długie potrafi być wyjście.

Dlaczego własna pętla, a nie HF Trainer: docTR PARSeq to zwykły `torch.nn.Module`
z wewnętrznym celem (permutowana autoregresja — `forward(x, target=list[str])`
zwraca `{"loss": ...}`). HF Trainer nie da dostępu do gradientów / momentów
optymalizatora pod rozbudowane logowanie W&B (normy gradientu, cosinus kierunku,
momenty AdamW/SGD, dryf wag, normy update'u). Stąd własna pętla.

Co logujemy do W&B per krok (poza loss/lr):
  - grad_norm_raw / grad_norm_clipped — norma L2 gradientu przed i po clipie;
  - grad_cos_step — cosinus między wektorem gradientu a gradientem z poprzedniego
    logowanego kroku (wykrywa oscylacje kierunku);
  - adam_m1_norm / adam_m2_norm — normy pierwszego i drugiego momentu AdamW
    (przy --optim sgd zamiast tego: momentum_norm — norma bufora momentum);
  - weight_norm, weight_drift_rel (||w-w0||/||w0||), weight_cos_init (cosinus wag
    vs stan startowy);
  - update_norm — ||Δw|| faktycznej zmiany wag po kroku;
  - co --hist-every kroków histogramy gradientów i wag wybranych warstw.
Sterowanie: --log-every, --hist-every, --optim {adamw,sgd}, --momentum, --grad-clip.

Uwagi architektoniczne (dlaczego tak — patrz też README.md):
- INPUT SZTYWNO (3, 32, 128). Patch embedding backbone'u ViT-S ma patch (4, 8) i
  NIE interpoluje pozycji dla niekwadratowych patchy (doctr PatchEmbedding.forward
  używa self.interpolate tylko dla patchy 4x4), więc pos_embed 32x128 = 129 tokenów
  nie "rozciągnie się" na szerszy input bez przebudowy backbone'u. Obrazy są zatem
  letterboxowane (aspect-preserve, centrowane), nie squashowane — tak samo jak w
  oficjalnym treningu docTR (T.Resize preserve_aspect_ratio). Szeroki input =
  osobny wątek (przebudowa pos_embed), poza zakresem tego pipeline'u.
- max_label_length podnosimy (domyślnie 96 >= najdłuższa linia 84). Pierwsze
  33 pozycje pos_queries ładujemy z checkpointu (ciepły start), resztę inicjujemy.
- Vocab budowany z base + spacja + (opcjonalnie) znaki z danych o częstości >=
  --min-add-count. Dzięki temu żaden znak GT nie jest "nieznany" (chyba że zbyt
  rzadki — wtedy próbki z nim są odrzucane z raportem).

Format danych — ten sam co Surya/TrOCR:
    training/data/processed/surya800k/{train,val}/metadata.jsonl  # {"file_name": ..., "text": ...}
    training/data/processed/surya800k/{train,val}/images/         # płaskie obrazy linii

Użycie (pełny run, w kontenerze treningowym):
    python cli.py train \
        --base-model Felix92/doctr-torch-parseq-multilingual-v1 \
        --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
        --train-images-dir training/data/processed/surya800k/train/images \
        --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
        --val-images-dir training/data/processed/surya800k/val/images \
        --output-dir training/results/ocr/parseq/run01 \
        --batch-size 64 --learning-rate 3e-4 --max-steps 40000 --report-to wandb

Smoke test: dodać `--max-steps 100 --limit-train 2048 --limit-val 512 \
    --eval-every 50 --save-every 0 --log-every 5`.
"""

import argparse
import json
import logging
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("parseq.train")

# Parametry zależne od rozmiaru vocabu / max_length — ładowane z base ręcznie
# (po identyczności znaków + rzędy specjalne EOS/SOS/PAD), reszta 1:1.
_VOCAB_FAMILY = {
    "embed.embedding.weight",  # (vocab+3, d): 0..V-1 znaki, V=EOS, V+1=SOS, V+2=PAD
    "head.weight",             # (vocab+1, d): 0..V-1 znaki, V=EOS (logit EOS)
    "head.bias",               # jw.
    "pos_queries",             # (1, max_length+1, d) — pozycyjne kwerendy dekodera
}

_PL_DIACRITICS = "ąćęłńóśźżĄĆĘŁŃŚŹŻ"


def norm_text(text: str) -> str:
    """Normalizacja tekstu jak w benchmarku (src/metrics.py)."""
    return " ".join(str(text).strip().split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #
def count_chars(metadata_path: str) -> Counter:
    """Liczy częstość znaków w tekstach metadata.jsonl (jeden przebieg)."""
    counts: Counter = Counter()
    n_lines = 0
    max_len = 0
    total_chars = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = (json.loads(line).get("text") or "").strip()
            if not text:
                continue
            n_lines += 1
            total_chars += len(text)
            max_len = max(max_len, len(text))
            counts.update(text)
    logger.info("  %s: %d linii, śr. długość %.1f, max %d",
                metadata_path, n_lines, total_chars / max(n_lines, 1), max_len)
    return counts


def build_vocab(
    base_vocab: str,
    data_counts: Counter,
    extra_chars: str = "",
    min_add_count: int = 2,
) -> tuple[str, list[str]]:
    """Vocab = base + spacja + --extra-vocab + znaki danych >= min_add_count.

    Zachowuje porządek base (append na końcu) — dzięki temu rzędy istniejących
    znaków w embed/head nie zmieniają indeksów i można je przenieść 1:1.
    """
    seen = set(base_vocab)
    vocab = list(base_vocab)
    added: list[str] = []
    for ch in " " + extra_chars:
        if ch and ch not in seen:
            seen.add(ch)
            vocab.append(ch)
            added.append(ch)
    for ch, cnt in sorted(data_counts.items()):
        if ch not in seen and cnt >= min_add_count:
            seen.add(ch)
            vocab.append(ch)
            added.append(ch)
    return "".join(vocab), added


# --------------------------------------------------------------------------- #
# Model: budowa + przeniesienie wag z base
# --------------------------------------------------------------------------- #
def build_model(vocab: str, input_shape, max_label_length: int):
    """Nowa instancja docTR PARSeq z rozszerzonym vocabem (wagi losowe)."""
    from doctr.models import parseq

    return parseq(
        vocab=vocab,
        pretrained=False,
        input_shape=tuple(input_shape),
        max_length=max_label_length,
    )


def transfer_from_base(model, base_sd: dict, base_vocab: str):
    """Przenosi wagi z checkpointu base na model z rozszerzonym vocabem.

    - parametry o identycznych kształtach (backbone, dekoder, norms) kopiujemy 1:1;
    - embed.embedding / head: wiersze znaków po identyczności (znak -> indeks),
      a rzędy specjalne EOS/SOS/PAD przenosimy wg ROLI (przesuwają się wraz z
      len(vocab), bo append do vocabu przesuwa ich indeksy);
    - pos_queries: kopiujemy pierwsze min(nowy, stary) rzędów (ciepły start),
      dalsze pozycje zostają z inicjalizacji.
    Model i base_sd muszą być na CPU.
    """
    state = model.state_dict()  # referencje do .data — mutacja zmienia model w miejscu
    base_c2i = {ch: idx for idx, ch in enumerate(base_vocab)}

    # 1) kopiowanie 1:1 (pomijamy rodzinę zależną od vocabu / max_length)
    for key, v in base_sd.items():
        if key not in state or key in _VOCAB_FAMILY:
            continue
        if state[key].shape == v.shape:
            state[key].copy_(v)

    vo = len(base_vocab)
    vn = len(model.vocab)

    # 2) embed.embedding.weight: (vo+3, d) -> (vn+3, d)
    key = "embed.embedding.weight"
    if key in base_sd:
        old_e, new_e = base_sd[key], state[key]
        for i, ch in enumerate(model.vocab):
            src = base_c2i.get(ch)
            if src is not None:
                new_e[i].copy_(old_e[src])
        # rzędy specjalne wg roli: EOS@vn, SOS@vn+1, PAD@vn+2 <- EOS@vo, SOS@vo+1, PAD@vo+2
        for role in range(3):
            if vo + role < old_e.shape[0] and vn + role < new_e.shape[0]:
                new_e[vn + role].copy_(old_e[vo + role])

    # 3) head.weight / head.bias: (vo+1, out) -> (vn+1, out); wiersz EOS@vn <- EOS@vo
    for key in ("head.weight", "head.bias"):
        if key in base_sd:
            old_h, new_h = base_sd[key], state[key]
            for i, ch in enumerate(model.vocab):
                src = base_c2i.get(ch)
                if src is not None:
                    new_h[i].copy_(old_h[src])
            if vo < old_h.shape[0] and vn < new_h.shape[0]:
                new_h[vn].copy_(old_h[vo])

    # 4) pos_queries: ciepły start pierwszych rzędów
    key = "pos_queries"
    if key in base_sd:
        old_p, new_p = base_sd[key], state[key]
        k = min(old_p.shape[1], new_p.shape[1])
        new_p[:, :k].copy_(old_p[:, :k])


def load_base_state(base_model: str, cache_dir: Optional[str]):
    """Zwraca (state_dict, vocab_str, input_shape, max_length_base).

    base_model: id z HF hub (np. Felix92/...) ALBO lokalny katalog treningowy
    (nasz output: model.pt + vocab.json + meta.json). Dla hub używa docTR
    `from_hub` (config + pytorch_model.bin; cache w HF_HOME/cache_dir).
    """
    base = str(base_model)
    if Path(base).is_dir():
        model_pt = Path(base) / "model.pt"
        if not model_pt.exists():
            raise FileNotFoundError(f"Brak {model_pt} w katalogu bazowym {base}")
        vocab = json.loads((Path(base) / "vocab.json").read_text(encoding="utf-8"))
        meta = json.loads((Path(base) / "meta.json").read_text(encoding="utf-8"))
        if isinstance(vocab, list):
            vocab = "".join(vocab)
        sd = torch.load(model_pt, map_location="cpu")
        max_len = sd["pos_queries"].shape[1] - 1 if "pos_queries" in sd else int(meta.get("max_label_length", 32))
        return sd, vocab, tuple(meta.get("input_shape", (3, 32, 128))), max_len

    from doctr.models import from_hub

    base_model_obj = from_hub(base, cache_dir=cache_dir)
    sd = base_model_obj.state_dict()
    return sd, base_model_obj.vocab, tuple(base_model_obj.cfg["input_shape"]), base_model_obj.max_length


# --------------------------------------------------------------------------- #
# Dane: obrazy linii -> (3, 32, 128) letterbox
# --------------------------------------------------------------------------- #
def _letterbox(path: Path, height: int, width: int, aug: str, rng: random.Random) -> np.ndarray:
    """Otwiera obraz, skaluje aspect-preserve do boxa (width x height), paduje
    centrowanie białym (jak docTR T.Resize preserve_aspect_ratio, bez squasha).
    Zwraca float32 (H, W, 3) w [0, 1]."""
    with Image.open(path).convert("RGB") as img:
        iw, ih = img.size
        if aug == "light":
            angle = rng.uniform(-1.5, 1.5)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
            iw, ih = img.size
        scale = min(width / iw, height / ih)
        nw = max(int(round(iw * scale)), 1)
        nh = max(int(round(ih * scale)), 1)
        img = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    left = (width - nw) // 2
    top = (height - nh) // 2
    canvas.paste(img, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if aug == "light":
        arr = arr * rng.uniform(0.92, 1.08)
        mean = arr.mean()
        arr = (arr - mean) * rng.uniform(0.92, 1.08) + mean
        arr = arr.clip(0.0, 1.0)
    return arr


class ParSeqLineDataset(torch.utils.data.Dataset):
    """Linie z metadata.jsonl + images/, mapowane pod docTR PARSeq 32x128."""

    def __init__(self, metadata_path: str, images_dir: str, allowed_chars,
                 max_label_length: int, height: int, width: int,
                 aug: str = "none", limit: int = -1, seed: int = 0):
        self.images_dir = Path(images_dir)
        self.height, self.width = height, width
        self.max_label_length = max_label_length
        self.aug = aug
        self.rng = random.Random(seed)
        self.samples = []
        skipped_unk = Counter()
        skipped_long = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text = (entry.get("text") or "").strip()
                if not text:
                    continue
                if len(text) > max_label_length:
                    skipped_long += 1
                    continue
                bad = next((c for c in text if c not in allowed_chars), None)
                if bad is not None:
                    skipped_unk[bad] += 1
                    continue
                self.samples.append((entry["file_name"], text))
                if 0 < limit <= len(self.samples):
                    break
        if not self.samples:
            raise ValueError(f"Brak próbek w {metadata_path}")
        if skipped_unk:
            unk_str = ", ".join(f"{c!r}:{n}" for c, n in skipped_unk.most_common())
            logger.warning("%s: pominięto %d próbek z nieznanymi znakami [%s]; "
                           "zbyt długich (>%d): %d", metadata_path, sum(skipped_unk.values()),
                           unk_str, max_label_length, skipped_long)
        elif skipped_long:
            logger.warning("%s: pominięto %d próbek dłuższych niż max_label_length=%d",
                           metadata_path, skipped_long, max_label_length)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, text = self.samples[index]
        image_path = self.images_dir / file_name
        if not image_path.exists():
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                return self[random.randrange(len(self))]
        arr = _letterbox(image_path, self.height, self.width, self.aug, self.rng)
        return arr, text


def parseq_collate(batch):
    """Stack (H,W,3) float [0,1] -> (B,3,H,W); teksty jako lista str."""
    xs, texts = zip(*batch)
    images = torch.from_numpy(np.stack(xs, 0)).permute(0, 3, 1, 2).contiguous()
    return images, list(texts)


def _worker_init(_worker_id):
    """Determinizm pomocniczego RNG numpy w procesach DataLoadera.

    Dataset i tak ma wlasny random.Random z seedem; ta funkcja musi byc
    modulowa (nie lambda), bo DataLoader z num_workers>0 pickle'uje ja do
    procesow potomnych."""
    np.random.seed(0)


# --------------------------------------------------------------------------- #
# Narzędzia do logowania gradientów / momentów / dryfu wag
# --------------------------------------------------------------------------- #
def _grad_flat(params):
    pieces = [p.grad.detach().float().reshape(-1) for p in params if p.grad is not None]
    if not pieces:
        return None
    return torch.cat(pieces)


def _weight_flat(params):
    return torch.cat([p.detach().float().reshape(-1) for p in params if p.requires_grad])


def _moment_norm(params, optimizer, state_key):
    pieces = []
    for p in params:
        st = optimizer.state.get(p)
        if st and state_key in st:
            pieces.append(st[state_key].detach().float().reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces).norm().item()


def _cosine(a, b):
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return 0.0
    return float((a * b).sum() / denom)


class GradMoments:
    """Liczy metryki gradientowe do W&B na żądanie (co --log-every kroków)."""

    def __init__(self, model):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.init_weights = None   # płaski wektor wag na starcie treningu
        self.prev_grad = None      # płaski wektor gradientu z poprzedniego logu

    def snapshot_init(self, device):
        self.init_weights = _weight_flat(self.params).to(device)

    def metrics(self, optimizer, device):
        cur_w = _weight_flat(self.params).to(device)
        grad = _grad_flat(self.params)
        is_adam = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))

        out = {}
        if grad is not None:
            grad = grad.to(device)
            # cosinus kierunku vs gradient z poprzedniego logowania; p.grad w chwili
            # wywołania jest gradientem z tego kroku (po clipie L2 — kierunek
            # niezmieniony, bo clip skaluje wektor jednorodnie), więc sygnał
            # o oscylacjach kierunku jest uczciwy.
            if self.prev_grad is not None and self.prev_grad.numel() == grad.numel():
                out["grad_cos_step"] = _cosine(self.prev_grad, grad)
            self.prev_grad = grad.clone()
        out["weight_norm"] = cur_w.norm().item()
        if self.init_weights is not None:
            diff = cur_w - self.init_weights
            out["weight_drift_rel"] = (diff.norm() / (self.init_weights.norm() + 1e-12)).item()
            out["weight_cos_init"] = _cosine(self.init_weights, cur_w)

        if is_adam:
            m1 = _moment_norm(self.params, optimizer, "exp_avg")
            m2 = _moment_norm(self.params, optimizer, "exp_avg_sq")
            if m1 is not None:
                out["adam_m1_norm"] = m1
            if m2 is not None:
                out["adam_m2_norm"] = m2
        else:
            mb = _moment_norm(self.params, optimizer, "momentum_buffer")
            if mb is not None:
                out["momentum_norm"] = mb
        return out


# --------------------------------------------------------------------------- #
# Ewaluacja (val) — autoregresyjny decode + loss, jak docTR model.eval()
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, normalize, device, use_amp, val_examples=0):
    model.eval()
    total_loss, count = 0.0, 0
    exact = near = lev_sum = cer_num = cer_den = 0
    examples = []
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and device.type == "cuda") \
        else _noop_ctx()

    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        need_img = val_examples > 0 and len(examples) < val_examples
        raw_images = images.clone() if need_img else None
        images = normalize(images)
        with amp_ctx:
            out = model(images, texts, return_preds=True)
        loss = out["loss"].float().item()
        total_loss += loss * len(texts)
        count += len(texts)
        words = [w for w, _ in out["preds"]] if out.get("preds") else []
        for j, (gt, pr) in enumerate(zip(texts, words)):
            gt_n, pr_n = norm_text(gt), norm_text(pr)
            lev = levenshtein(gt_n, pr_n)
            exact += (lev == 0)
            near += (lev <= 1)
            lev_sum += lev
            cer_den += len(gt_n)
            cer_num += lev
            if need_img and len(examples) < val_examples:
                img = raw_images[j].permute(1, 2, 0).float().cpu().numpy()
                examples.append((img, gt, pr_n))
    model.train()
    return {
        "loss": total_loss / max(count, 1),
        "ema": exact / max(count, 1),
        "near_perfect": near / max(count, 1),
        "cer": cer_num / max(cer_den, 1),
        "mean_lev": lev_sum / max(count, 1),
        "count": count,
        "examples": examples,
    }


# --------------------------------------------------------------------------- #
# Zapis checkpointu / metadanych
# --------------------------------------------------------------------------- #
def save_model_and_state(model, vocab, args, out_dir: Path, tag: str,
                         trainer_state: Optional[dict]):
    """model.pt (lub last.pt) = CZYSTY state_dict (zgodny z docTR from_pretrained);
    trainer_state.pt = pełny stan do wznowienia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / tag)
    if trainer_state is not None:
        torch.save(trainer_state, out_dir / "trainer_state.pt")
    if tag == "model.pt":
        (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        meta = {
            "model": "docTR PARSeq (multilingual base, fine-tuned)",
            "base_model": args.base_model,
            "vocab_len": len(vocab),
            "input_shape": list(model.cfg["input_shape"]),
            "max_label_length": int(model.max_length),
            "mean": list(model.cfg["mean"]),
            "std": list(model.cfg["std"]),
            "args": {k: str(v) for k, v in vars(args).items()},
            "versions": {
                "torch": torch.__version__,
                "doctr": __import__("doctr").__version__,
                "python": platform.python_version(),
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


class _noop_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Trening
# --------------------------------------------------------------------------- #
def cmd_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not (args.train_metadata and args.train_images_dir):
        raise ValueError("Podaj --train-metadata i --train-images-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("device=%s", device)

    # -- base / vocab ------------------------------------------------------- #
    base_sd, base_vocab, input_shape, base_max_len = load_base_state(
        args.base_model, args.cache_dir)
    height, width = int(input_shape[1]), int(input_shape[2])

    logger.info("Zliczam znaki w danych (train)...")
    train_counts = count_chars(args.train_metadata)
    val_counts = count_chars(args.val_metadata) if args.val_metadata and Path(args.val_metadata).exists() else Counter()
    vocab, added = build_vocab(base_vocab, train_counts + val_counts,
                               extra_chars=args.extra_vocab, min_add_count=args.min_add_count)

    missing_pl = [c for c in _PL_DIACRITICS if c not in vocab]
    logger.info("Base vocab: %d znaków -> docelowy vocab: %d znaków (dodano %d)",
                len(base_vocab), len(vocab), len(vocab) - len(base_vocab))
    logger.info("Dodane znaki: %s", ", ".join(repr(c) for c in added) or "-")
    logger.info("Diakrytyki PL w vocabu: %s", "wszystkie ✓" if not missing_pl else f"BRAK: {missing_pl}")

    # -- model --------------------------------------------------------------- #
    model = build_model(vocab, input_shape, args.max_label_length)
    transfer_from_base(model, base_sd, base_vocab)
    del base_sd
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model PARSeq: %s parametrów | input=%s | max_label_length=%d (base=%d)",
                n_params, tuple(model.cfg["input_shape"]), model.max_length, base_max_len)
    model.to(device)

    allowed = set(vocab)
    train_ds = ParSeqLineDataset(
        args.train_metadata, args.train_images_dir, allowed, args.max_label_length,
        height, width, aug=args.aug, limit=args.limit_train, seed=args.seed)
    eval_ds = None
    if args.val_metadata and args.val_images_dir:
        full_eval_ds = ParSeqLineDataset(
            args.val_metadata, args.val_images_dir, allowed, args.max_label_length,
            height, width, aug="none", limit=args.limit_val, seed=args.seed)
        if args.val_size and args.val_size < len(full_eval_ds):
            rng = random.Random(args.seed)
            idx = sorted(rng.sample(range(len(full_eval_ds)), args.val_size))
            eval_ds = torch.utils.data.Subset(full_eval_ds, idx)
        else:
            eval_ds = full_eval_ds

    gen = torch.Generator().manual_seed(args.seed)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.dataloader_num_workers,
            collate_fn=parseq_collate, drop_last=shuffle,
            pin_memory=(device.type == "cuda"), generator=gen,
            worker_init_fn=_worker_init,
        )

    mean = torch.tensor(model.cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.cfg["std"], device=device).view(1, 3, 1, 1)

    def normalize(x):
        return (x - mean) / std

    # -- optymalizator / harmonogram ---------------------------------------- #
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999),
                                      eps=1e-6, weight_decay=args.weight_decay)

    eff_bs = args.batch_size * args.gradient_accumulation_steps
    if len(train_ds) < eff_bs:
        raise ValueError(f"train ma tylko {len(train_ds)} próbek < efektywny batch {eff_bs}; "
                         "zwiększ --limit-train lub zmniejsz batch/akumulację")
    steps_per_epoch = max(len(train_ds) // eff_bs, 1)
    total_target = args.max_steps if (args.max_steps and args.max_steps > 0) \
        else steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_target)

    def lr_at(step):
        """step: 1-based liczba wykonanych kroków optymalizatora."""
        if step <= warmup_steps:
            return args.learning_rate * (step / max(warmup_steps, 1))
        if step >= total_target:
            return 0.0
        p = (step - warmup_steps) / max(total_target - warmup_steps, 1)
        return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * p))

    # -- wandb ---------------------------------------------------------------- #
    report_to = [r for r in (args.report_to or "").split(",") if r]
    out_dir = Path(args.output_dir)
    if "wandb" in report_to:
        import wandb
        run = wandb.init(
            name=args.run_name or f"parseq-{out_dir.name}",
            config={
                "architecture": "docTR PARSeq (multilingual base)",
                "base_model": args.base_model,
                "vocab_len": len(vocab),
                "vocab_added": "".join(added),
                "diacritics_pl_present": "wszystkie" if not missing_pl else "BRAK:" + "".join(missing_pl),
                "input_shape": list(model.cfg["input_shape"]),
                "max_label_length": args.max_label_length,
                "n_params": n_params,
                "optimizer": args.optim,
                **{k: str(v) for k, v in vars(args).items()},
            },
        )

    def wlog(data, step=None):
        if "wandb" in report_to:
            import wandb
            wandb.log(data, step=step)

    # -- resume ---------------------------------------------------------------- #
    global_step = 0
    best_ema = -1.0
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        state_file = resume_path / "trainer_state.pt" if resume_path.is_dir() else resume_path
        if state_file.exists():
            st = torch.load(state_file, map_location=device)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            global_step = int(st.get("step", 0))
            best_ema = float(st.get("best_ema", -1.0))
            best_val_loss = float(st.get("best_val_loss", float("inf")))
            logger.info("Wznowiono z %s (step=%d)", state_file, global_step)

    # -- logowanie gradientów --------------------------------------------------- #
    mom = GradMoments(model)
    mom.snapshot_init(device)
    hist_keys = [k for k in ("head.weight", "embed.embedding.weight",
                              "feat_extractor.0.projection.weight")
                 if k in dict(model.named_parameters())]

    def histograms(tag):
        import wandb
        h = {}
        for name, p in model.named_parameters():
            if name not in hist_keys:
                continue
            src = p.grad if (tag == "grad" and p.grad is not None) else p
            h[f"{tag}/{name}"] = wandb.Histogram(src.detach().float().cpu().numpy().ravel())
        return h

    # -- strumień batchy (nieskończony; drop_last == pełne kroki) --------------- #
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if (args.bf16 and device.type == "cuda") else _noop_ctx()

    def batch_stream():
        epoch_no = 0
        while True:
            epoch_no += 1
            loader = make_loader(train_ds, shuffle=True)
            for batch in loader:
                yield epoch_no, batch

    stream = batch_stream()
    model.train()

    steps_done = global_step
    loss_ema = None
    log_t0 = time.time()
    micro_accum = args.gradient_accumulation_steps
    stop = False

    logger.info("Start treningu: kroków %d (%d/epoch, warmup %d), batch %d x%d",
                total_target, steps_per_epoch, warmup_steps, args.batch_size, micro_accum)
    while steps_done < total_target:
        optimizer.zero_grad(set_to_none=True)
        epoch_no = 0
        loss_val = None
        for _ in range(micro_accum):
            epoch_no, (images, texts) = next(stream)
            images = images.to(device, non_blocking=True)
            images = normalize(images)
            with amp_ctx:
                loss = model(images, texts)["loss"] / micro_accum
            loss.backward()
            if loss_val is None:
                loss_val = float(loss.item()) * micro_accum

        params = [p for p in model.parameters() if p.requires_grad]
        grad_raw = _grad_flat(params)
        raw_norm = grad_raw.norm().item() if grad_raw is not None else 0.0
        clip_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))

        next_step = steps_done + 1
        lr_now = lr_at(next_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # płaski wektor wag przed krokiem — tylko gdy ten krok będzie logowany
        weights_before = _weight_flat(params).to(device) if (steps_done + 1) % args.log_every == 0 else None
        optimizer.step()
        steps_done = next_step

        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = 0.99 * loss_ema + 0.01 * loss_val

        # -- logowanie co --log-every ------------------------------------------
        if steps_done % args.log_every == 0:
            log = {
                "train_loss": loss_val,
                "train_loss_ema": loss_ema,
                "lr": lr_now,
                "step": steps_done,
                "epoch": epoch_no,
                "grad_norm_raw": raw_norm,
                "grad_norm_clipped": clip_norm,
            }
            log.update(mom.metrics(optimizer, device))
            weights_after = _weight_flat(params).to(device)
            log["update_norm"] = (weights_after - weights_before).norm().item()
            log["samples_per_sec"] = eff_bs * args.log_every / max(time.time() - log_t0, 1e-9)
            wlog(log, step=steps_done)
            log_t0 = time.time()

        # -- histogramy ---------------------------------------------------------
        if args.hist_every and args.hist_every > 0 and steps_done % args.hist_every == 0:
            wlog({**histograms("grad"), **histograms("weight")}, step=steps_done)

        # -- ewaluacja ----------------------------------------------------------
        if eval_ds is not None and args.eval_every and args.eval_every > 0 \
                and steps_done % args.eval_every == 0:
            eval_loader = make_loader(eval_ds, shuffle=False)
            ev = evaluate(model, eval_loader, normalize, device, use_amp=args.bf16,
                          val_examples=args.wandb_val_examples if "wandb" in report_to else 0)
            log = {
                "val_loss": ev["loss"], "val_ema": ev["ema"],
                "val_near_perfect": ev["near_perfect"], "val_cer": ev["cer"],
                "val_mean_lev": ev["mean_lev"], "step": steps_done,
            }
            if "wandb" in report_to:
                import wandb
                for k, (img, gt, pr) in enumerate(ev["examples"]):
                    safe = "".join(c if c.isalnum() else "_" for c in gt)[:20]
                    log[f"val_example/{k:02d}_{safe}"] = wandb.Image(
                        (img * 255).clip(0, 255).astype(np.uint8),
                        caption=f"GT: {gt}\nPRED: {pr}")
            wlog(log, step=steps_done)
            logger.info("step=%d val_loss=%.4f ema=%.4f cer=%.4f (n=%d)",
                        steps_done, ev["loss"], ev["ema"], ev["cer"], ev["count"])

            better = ev["ema"] > best_ema or ev["loss"] < best_val_loss
            if better:
                best_ema = max(best_ema, ev["ema"])
                best_val_loss = min(best_val_loss, ev["loss"])
                save_model_and_state(model, vocab, args, out_dir, "model.pt",
                                     trainer_state={"model": model.state_dict(),
                                                    "optimizer": optimizer.state_dict(),
                                                    "step": steps_done,
                                                    "best_ema": best_ema,
                                                    "best_val_loss": best_val_loss})
                wlog({"best_val_ema": best_ema, "best_val_loss": best_val_loss}, step=steps_done)
                logger.info("  -> model.pt (best val_ema=%.4f val_loss=%.4f)", best_ema, best_val_loss)

        # -- okresowy checkpoint do wznowienia ------------------------------------
        if args.save_every and args.save_every > 0 and steps_done % args.save_every == 0:
            save_model_and_state(model, vocab, args, out_dir, "model.pt",
                                 trainer_state={"model": model.state_dict(),
                                                "optimizer": optimizer.state_dict(),
                                                "step": steps_done,
                                                "best_ema": best_ema,
                                                "best_val_loss": best_val_loss})
            logger.info("  -> trainer_state.pt @ step %d", steps_done)

        if steps_done % 50 == 0 or steps_done == 1:
            logger.info("step=%d loss=%.4f (ema %.4f) lr=%.2e grad_norm=%.3f",
                        steps_done, loss_val, loss_ema, lr_now, raw_norm)

    # -- finalizacja ------------------------------------------------------------ #
    save_model_and_state(model, vocab, args, out_dir, "last.pt",
                         trainer_state={"model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "step": steps_done,
                                        "best_ema": best_ema,
                                        "best_val_loss": best_val_loss})
    if not (out_dir / "model.pt").exists():
        # ewaluacja wyłączona — model.pt = ostatni krok
        save_model_and_state(model, vocab, args, out_dir, "model.pt")
    logger.info("Koniec: kroków %d | best val_ema=%.4f val_loss=%.4f",
                steps_done, best_ema, best_val_loss)

    if "wandb" in report_to:
        import wandb
        if wandb.run is not None:
            wandb.finish()


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="command", metavar="{train}", required=True)
    p = sub.add_parser("train", help="Fine-tuning PARSeq (docTR) na liniach ocr_800k")

    # model / vocab
    p.add_argument("--base-model", default="Felix92/doctr-torch-parseq-multilingual-v1",
                   help="Bazowy model docTR PARSeq: HF hub id albo lokalny katalog treningowy (model.pt+vocab.json)")
    p.add_argument("--cache-dir", default=None,
                   help="Cache modeli HF (domyślnie None = HF_HOME, w kontenerze /cache/hf)")
    p.add_argument("--max-label-length", type=int, default=96,
                   help="Maks. długość sekwencji (najdłuższa linia w danych ~84)")
    p.add_argument("--extra-vocab", default="",
                   help="Dodatkowe znaki do vocabu (oprócz base + spacji), np. '„”'")
    p.add_argument("--min-add-count", type=int, default=2,
                   help="Znaki z danych o częstości >= tego progu są dodawane do vocabu")
    # dane (lokalne, format Surya/TrOCR)
    p.add_argument("--train-metadata", default=None)
    p.add_argument("--train-images-dir", default=None)
    p.add_argument("--val-metadata", default=None)
    p.add_argument("--val-images-dir", default=None)
    p.add_argument("--limit-train", type=int, default=-1, help="Ograniczenie próbek train (-1 = wszystko)")
    p.add_argument("--limit-val", type=int, default=-1, help="Ograniczenie próbek val (-1 = wszystko)")
    p.add_argument("--val-size", type=int, default=4096,
                   help="Rozmiar podzbioru walidacyjnego używanego przy ewaluacji (0 = całość)")
    # trening
    p.add_argument("--output-dir", default="training/results/ocr/parseq/default")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=-1, help="Limit kroków optymalizatora (-1 = epochs)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--optim", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum (gdy --optim sgd)")
    p.add_argument("--grad-clip", type=float, default=5.0, help="Max norm gradientu (jak docTR ref)")
    p.add_argument("--aug", choices=["none", "light"], default="none",
                   help="light = drobny szum geometryczny/fotometryczny")
    p.add_argument("--bf16", action="store_true", default=True, help="Mixed precision bfloat16 (Ampere+)")
    p.add_argument("--no-bf16", action="store_true", help="Wyłącz bf16 (fp32)")
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # logowanie
    p.add_argument("--report-to", default="", help="Backend logowania, np. 'wandb' (pusty = brak)")
    p.add_argument("--run-name", default=None, help="Nazwa runu (dla W&B)")
    p.add_argument("--log-every", type=int, default=10, help="Co ile kroków log metryk gradientowych")
    p.add_argument("--hist-every", type=int, default=1000, help="Co ile kroków histogramy (0 = off)")
    p.add_argument("--eval-every", type=int, default=5000, help="Co ile kroków ewaluacja (0 = off)")
    p.add_argument("--save-every", type=int, default=5000, help="Co ile kroków checkpoint do wznowienia (0 = off)")
    p.add_argument("--wandb-val-examples", type=int, default=4,
                   help="Ile przykładów GT vs pred logować do W&B przy ewaluacji")
    p.add_argument("--resume-from", default=None, help="Katalog/plik trainer_state.pt do wznowienia")

    args = ap.parse_args(argv)
    if args.no_bf16:
        args.bf16 = False
    cmd_train(args)


if __name__ == "__main__":
    main()
