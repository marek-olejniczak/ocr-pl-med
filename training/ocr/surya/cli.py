"""Fine-tuning CLI dla Surya OCR (foundation model, surya-ocr==0.17.1).

Działa wyłącznie na lokalnych danych w formacie:
    processed/surya/{train,val}/metadata.jsonl   # {"file_name": ..., "text": ...}
    processed/surya/{train,val}/images/          # płaskie pliki obrazów

Konwersję z surowych danych (ocr_800k / handlabeled) robi `convert_data.py`.

Użycie:
    python cli.py train \
        --train-metadata data/processed/surya/train/metadata.jsonl \
        --train-images-dir data/processed/surya/train/images \
        --val-metadata data/processed/surya/val/metadata.jsonl \
        --val-images-dir data/processed/surya/val/images \
        --lora --report-to wandb

Ewaluacja wytrenowanego adaptera odbywa się w benchmarku (benchmark/docker/surya
ładuje LoRA przez `lora_adapter_path`), więc tu nie ma subkomendy predict.
"""

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Importy zgodne z surya-ocr==0.17.1 (model foundation)
from surya.common.surya.processor.schema import ImageInput, TextInput
from surya.common.surya.schema import TaskNames
from surya.common.util import SCRIPT_TOKEN_MAPPING, get_top_scripts

OCR_TASK_NAME = TaskNames.ocr_with_boxes
# Nie zmieniać — to domyślny rozmiar obrazu modelu foundation.
OCR_MAX_IMAGE_SIZE = (1024, 512)


# ---------------------------------------------------------------------------
# dataset lokalny (metadata.jsonl + obrazy)
# ---------------------------------------------------------------------------

class LocalOCRDataset(torch.utils.data.Dataset):
    """Dataset ładujący dane z lokalnego katalogu: metadata.jsonl + images/."""

    def __init__(self, metadata_path: str, images_dir: str, processor):
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text = (entry.get("text") or "").strip()
                if text:
                    self.samples.append({
                        "file_name": entry["file_name"],
                        "text": text,
                    })
        if not self.samples:
            raise ValueError(f"Brak próbek z tekstem w {metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.images_dir / sample["file_name"]
        if not image_path.exists():
            # fallback: inne rozszerzenia
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                logging.warning("Obraz nie znaleziony: %s, pomijanie", image_path)
                return self[(index + 1) % len(self)]

        image = Image.open(image_path).convert("RGB")
        # Procesor sam normalizuje (rescale 1/255) — nie dziel tu przez 255.
        image_np = np.asarray(image, dtype=np.float32)
        image_np = self.processor.scale_to_fit(image_np, max_size=OCR_MAX_IMAGE_SIZE)

        # Prefiks skryptowy — wymagany przez Surya w ground truth.
        scripts = get_top_scripts(sample["text"])
        script_prefix = "".join(
            SCRIPT_TOKEN_MAPPING[s] for s in scripts if s in SCRIPT_TOKEN_MAPPING
        )
        gt_text = script_prefix + sample["text"]

        return {
            "task": OCR_TASK_NAME,
            "inputs": [
                ImageInput(type="image", image=image_np, rotated=False),
                # Pusty TextInput MUSI być — tego wymaga procesor (pozycja inputu).
                TextInput(type="text", text=""),
                TextInput(type="text", text=gt_text),
            ],
        }


# ---------------------------------------------------------------------------
# data collator
# ---------------------------------------------------------------------------

class SuryaOCRDataCollator:
    """Collator dla foundation modelu (port z działającego forka).

    HF Trainer woła `model.forward()` bezpośrednio, a `SuryaModel.forward`
    wymaga przeliczonych wcześniej `cache_position` i `image_embeddings`.
    Dlatego collator trzyma referencję do modelu i musi działać w procesie
    głównym — stąd `dataloader_num_workers=0` (patrz argparse).
    """

    def __init__(self, model, processor,
                 max_sequence_length: Optional[int] = None,
                 encoder_chunk_size: int = 32768,
                 bf16: bool = True):
        self.model = model
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.encoder_chunk_size = encoder_chunk_size
        self.bf16 = bf16

    def __call__(self, inputs):
        processed = self.processor(inputs, padding_side="right")

        if self.max_sequence_length is not None:
            for key in ("input_ids", "attention_mask", "position_ids"):
                processed[key] = processed[key][:, : self.max_sequence_length]

        # cache_position — wymagany przez SuryaModel.forward.
        seq_len = processed["input_ids"].shape[1]
        processed["cache_position"] = torch.arange(0, seq_len, dtype=torch.long)

        # image_embeddings — Trainer nie robi tego za nas, więc precompute tu.
        image_tiles = processed.pop("image_tiles")
        grid_thw = processed.pop("grid_thw")
        with torch.no_grad():  # freeze vision encodera (przy LoRA i tak zamrożony)
            image_embeddings = self.model.get_image_embeddings(
                pixel_values=image_tiles,
                grid_thw=grid_thw,
                encoder_chunk_size=self.encoder_chunk_size,
            )
        image_embeddings = image_embeddings.cpu()

        # Podział na próbki + padding + stack (każda próbka ma 1 obraz).
        batch_size = grid_thw.shape[0]
        merge_size = self.processor.merge_size
        tokens_per_sample = []
        for i in range(batch_size):
            _, grid_h, grid_w = grid_thw[i]
            tokens_per_sample.append(((grid_h // merge_size) * (grid_w // merge_size)).item())

        split_embeddings = []
        start_idx = 0
        for num_tokens in tokens_per_sample:
            end_idx = start_idx + num_tokens
            split_embeddings.append(image_embeddings[start_idx:end_idx].clone())
            start_idx = end_idx

        # PŁASKO, bez padowania: SuryaModel.forward robi masked_scatter, które
        # wymaga numel(image_embeddings) == liczbie tokenów obrazu w batchu.
        # Padding/stackowanie rozjechałoby alignment między próbkami.
        processed["image_embeddings"] = torch.cat(split_embeddings, dim=0)

        # Sanity check: liczba tokenów obrazu w input_ids == liczba embeddingów.
        for i in range(batch_size):
            n_img_tokens = (processed["input_ids"][i] == self.processor.image_token_id).sum().item()
            assert n_img_tokens == tokens_per_sample[i], (
                f"Sample {i}: image tokens ({n_img_tokens}) != embeddings "
                f"({tokens_per_sample[i]})"
            )

        # Etykiety: maskuj tokeny specjalne jako -100.
        labels = processed["input_ids"].clone()
        skip_mask = (
            (labels == self.processor.pad_token_id)
            | (labels == self.processor.bos_token_id[OCR_TASK_NAME])
            | (labels == self.processor.eoi_token_id)
            | (labels == self.processor.image_token_id)
        )
        labels[skip_mask] = -100
        processed["labels"] = labels

        # 4D causal maska zamiast 2D: SuryaModel._prepare_4d_causal_attention_mask_with_cache_position
        # ma bug przy batch>1 (buduje causal_mask z batch=1, a masked_fill z attention_mask o batch=8).
        # Jeśli podamy maskę 4D, surya zwraca ją bez zmian (dim()==4). Dla flash_attention_2
        # surya zwraca 2D maskę bez zmian — tam NIE budujemy 4D.
        attn_impl = getattr(self.model.config, "_attn_implementation", None)
        if attn_impl != "flash_attention_2":
            batch, seq = processed["input_ids"].shape
            device = processed["input_ids"].device
            dtype = torch.bfloat16 if self.bf16 else torch.float32
            min_dtype = torch.finfo(dtype).min
            # 0 = można attendować (przeszłość + self), min_dtype = zablokowane.
            causal = torch.full((batch, 1, seq, seq), min_dtype, dtype=dtype, device=device)
            causal = causal.masked_fill(
                torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
                .unsqueeze(0).unsqueeze(0),
                0.0,
            )
            pad = processed["attention_mask"] == 0  # (batch, seq)
            causal = causal.masked_fill(pad.unsqueeze(1).unsqueeze(1), min_dtype)  # kv-padding
            causal = causal.masked_fill(pad.unsqueeze(1).unsqueeze(2), min_dtype)  # query-padding
            processed["attention_mask"] = causal

        return processed


# ---------------------------------------------------------------------------
# ładowanie modelu i procesora
# ---------------------------------------------------------------------------

def load_model_and_processor(checkpoint_path: Optional[str] = None):
    """Ładuje SuryaModel i SuryaOCRProcessor przez FoundationPredictor."""
    from surya.foundation import FoundationPredictor

    predictor = FoundationPredictor(checkpoint=checkpoint_path)
    return predictor.model, predictor.processor


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora(model, lora_config: dict):
    """Nakłada LoRA na model Surya (attention-only)."""
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=lora_config.get("rank", 32),
        lora_alpha=lora_config.get("alpha", 64),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "o_proj"]
        ),
        # task_type=None: SuryaModel to nie model CausalLM z prepare_inputs_for_generation,
        # więc PEFT musi użyć generycznego PeftModel, nie PeftModelForCausalLM.
        task_type=None,
    )
    return get_peft_model(model, peft_config)


# ---------------------------------------------------------------------------
# W&B — artefakty
# ---------------------------------------------------------------------------

def _log_artifact(run, name: str, dir_path: Path) -> None:
    if not dir_path.exists():
        return
    artifact = __import__("wandb").Artifact(name, type="model")
    artifact.add_dir(str(dir_path))
    run.log_artifact(artifact)


def _save_adapter(trainer, output_dir: Path, meta: dict) -> Path:
    """Zapisuje adapter LoRA (model po load_best_model_at_end) + meta.json."""
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Użyj trainer.model, bo load_best_model_at_end mogło podmienić self.model.
    trainer.model.save_pretrained(str(adapter_dir))
    (adapter_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return adapter_dir


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Fine-tuning Surya OCR — LoRA (lub pełny, jeśli --lora nie podane)."""
    from transformers import Trainer, TrainingArguments

    if not (args.train_metadata and args.train_images_dir):
        raise ValueError("Podaj --train-metadata i --train-images-dir (dane lokalne)")

    model, processor = load_model_and_processor(args.pretrained_checkpoint)

    train_dataset = LocalOCRDataset(args.train_metadata, args.train_images_dir, processor)
    eval_dataset = None
    if args.val_metadata and args.val_images_dir:
        eval_dataset = LocalOCRDataset(args.val_metadata, args.val_images_dir, processor)

    if args.lora:
        model = apply_lora(model, {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        })
        print(f"LoRA applied: rank={args.lora_rank}, alpha={args.lora_alpha}")

    collator = SuryaOCRDataCollator(
        model, processor, args.max_sequence_length, bf16=args.bf16)

    report_to = [r for r in (args.report_to or "").split(",") if r]

    # load_best_model_at_end wymaga zgodności save/eval strategii; przy
    # eval co kroki (--evaluation-strategy steps) checkpointy też idą co
    # kroki (w tych samych krokach), inaczej transformers rzuca ValueError.
    eval_strategy = args.evaluation_strategy if eval_dataset else "no"
    save_strategy = args.save_strategy
    if eval_strategy == "steps":
        save_strategy = "steps"
    save_steps = args.eval_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

    meta = {
        "model": "surya-ocr-foundation",
        "lora": args.lora,
        "config": {k: str(v) for k, v in vars(args).items()},
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
    }
    adapter_dir = _save_adapter(trainer, Path(args.output_dir), meta)
    print(f"Adapter LoRA zapisany w {adapter_dir}")

    if "wandb" in report_to:
        import wandb
        run = wandb.run
        if run is not None:
            _log_artifact(run, "surya-lora-best", adapter_dir)
            checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
            if checkpoints:
                _log_artifact(run, "surya-lora-last", checkpoints[-1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # model
    ap.add_argument("--pretrained-checkpoint", default=None,
                   help="Ścieżka do bazowego checkpointu (null = domyślny checkpoint surya)")
    ap.add_argument("--max-sequence-length", type=int, default=1024)
    # dane (lokalne)
    ap.add_argument("--train-metadata", default=None, help="metadata.jsonl (train)")
    ap.add_argument("--train-images-dir", default=None, help="Katalog obrazów (train)")
    ap.add_argument("--val-metadata", default=None, help="metadata.jsonl (val)")
    ap.add_argument("--val-images-dir", default=None, help="Katalog obrazów (val)")
    # LoRA
    ap.add_argument("--lora", action="store_true",
                   help="Użyj LoRA (attention-only) zamiast pełnego fine-tuningu")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "v_proj", "o_proj"])
    # trening
    ap.add_argument("--output-dir", default="training/results/ocr/surya/default")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1,
                   help="Limit kroków treningu (-1 = bez limitu; do smoke testu np. 2)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--lr-scheduler-type", default="cosine")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--evaluation-strategy", default="epoch")
    ap.add_argument("--eval-steps", type=int, default=None,
                   help="Co ile kroków ewaluacja (wymaga --evaluation-strategy steps)")
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--bf16", action="store_true", default=True,
                   help="Mixed precision bfloat16 (Ampere+; domyślnie włączone)")
    ap.add_argument("--no-bf16", action="store_true", help="Wyłącz bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Włącz gradient checkpointing (nieprzetestowane z custom forward)")
    ap.add_argument("--dataloader-num-workers", type=int, default=0,
                   help="Musi być 0 — collator trzyma referencję do modelu")
    ap.add_argument("--report-to", default="",
                   help="Backend logowania, np. 'wandb' (pusty = brak)")
    ap.add_argument("--run-name", default=None, help="Nazwa runu (dla W&B)")

    args = ap.parse_args(argv)

    if args.no_bf16:
        args.bf16 = False

    cmd_train(args)


if __name__ == "__main__":
    main()
