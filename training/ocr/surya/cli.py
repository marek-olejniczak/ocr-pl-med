"""Train / predict CLI for Surya OCR fine-tuning.

Wzorowane na line_benchmark/docker/ultralytics/cli.py — jeden entry point
z subkomendami train, predict, convert.

Subkomendy:
    train     — full fine-tuning lub LoRA (wskazuje --lora)
    predict   — ewaluacja wytrenowanego checkpointu na zbiorze testowym
    convert   — konwersja danych raw → format Surya

Przykłady:
    python cli.py train --data metadata.jsonl --images-dir images/ --lora --epochs 10
    python cli.py predict --checkpoint results/ocr/surya/run01/checkpoint \\
        --data val_metadata.jsonl --images-dir val_images/
    python cli.py convert --input data/raw --output data/processed/surya
"""

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


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
                entry = json.loads(line.strip())
                text = entry.get("text", "").strip()
                if text:
                    self.samples.append({
                        "file_name": entry["file_name"],
                        "text": text,
                    })
        if not self.samples:
            raise ValueError(f"Brak próbek z tekstem w {metadata_path}")

    @staticmethod
    def _get_script_text(text: str, processor) -> str:
        """Prefiks skryptowy wymagany przez Surya."""
        try:
            from surya.common.scripts import get_top_scripts, SCRIPT_TOKEN_MAPPING
            scripts = get_top_scripts(text)
            tokens = [SCRIPT_TOKEN_MAPPING[s] for s in scripts if s in SCRIPT_TOKEN_MAPPING]
            return "".join(tokens)
        except Exception:
            return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from surya.common.schema import ImageInput, TextInput, TaskNames

        sample = self.samples[index]
        image_path = self.images_dir / sample["file_name"]
        if not image_path.exists():
            # fallback: szukaj z innymi rozszerzeniami
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                logging.warning("Obraz nie znaleziony: %s, pomijanie", image_path)
                return self[(index + 1) % len(self)]

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Skaluj do OCR_MAX_IMAGE_SIZE
        scaled = self.processor.scale_image(image_np, (1024, 512))
        if scaled is not None:
            image_np = scaled

        script_prefix = self._get_script_text(sample["text"], self.processor)
        gt_text = script_prefix + sample["text"]

        return {
            "task": TaskNames.ocr_with_boxes,
            "inputs": [
                ImageInput(type="image", image=image_np, rotated=False),
                TextInput(type="text", text=""),
                TextInput(type="text", text=gt_text),
            ],
        }


# ---------------------------------------------------------------------------
# data collator
# ---------------------------------------------------------------------------

class SuryaOCRCollator:
    """Collator wzorowany na surya/scripts/finetune_ocr.py.

    Prawostronne paduje batch, maskuje tokeny specjalne (pad, bos, eoi,
    image_token) jako -100 w etykietach.
    """

    def __init__(self, processor, max_sequence_length: Optional[int] = None):
        self.processor = processor
        self.max_sequence_length = max_sequence_length

    def __call__(self, inputs: list[dict]) -> dict:
        processed = self.processor(inputs, padding_side="right")

        # Obcięcie do max_sequence_length
        if self.max_sequence_length is not None:
            for key in ("input_ids", "attention_mask", "position_ids"):
                if key in processed:
                    processed[key] = processed[key][:, : self.max_sequence_length]

        # Maskowanie etykiet
        input_ids = processed["input_ids"]
        labels = input_ids.clone()
        pad_id = self.processor.pad_token_id
        bos_id = self.processor.bos_token_id[self.processor.ocr_task_name]
        eoi_id = self.processor.eoi_token_id
        image_id = self.processor.image_token_id

        skip_mask = (
            (labels == pad_id)
            | (labels == bos_id)
            | (labels == eoi_id)
            | (labels == image_id)
        )
        labels[skip_mask] = -100
        processed["labels"] = labels

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
    """Nakłada LoRA na model Surya."""
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=lora_config.get("rank", 32),
        lora_alpha=lora_config.get("alpha", 64),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "o_proj"]
        ),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Fine-tuning Surya OCR — full lub LoRA."""
    from transformers import Trainer, TrainingArguments

    # --- dane ---
    model, processor = load_model_and_processor(args.pretrained_checkpoint)

    if args.dataset_name:
        # HuggingFace dataset (oficjalna ścieżka)
        from datasets import load_dataset

        from surya.scripts.finetune_ocr import SuryaOCRDataset as HfSuryaOCRDataset

        raw_dataset = load_dataset(
            args.dataset_name, num_proc=args.num_loading_proc, split="train"
        )
        train_dataset = HfSuryaOCRDataset(processor, raw_dataset)
    elif args.train_metadata and args.train_images_dir:
        train_dataset = LocalOCRDataset(
            args.train_metadata, args.train_images_dir, processor
        )
    else:
        raise ValueError(
            "Podaj --dataset-name (HuggingFace) lub "
            "--train-metadata + --train-images-dir (lokalne)"
        )

    collator = SuryaOCRCollator(processor, args.max_sequence_length)

    # --- walidacja (opcjonalnie) ---
    eval_dataset = None
    if args.val_metadata and args.val_images_dir:
        eval_dataset = LocalOCRDataset(
            args.val_metadata, args.val_images_dir, processor
        )

    # --- LoRA ---
    if args.lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        }
        model = apply_lora(model, lora_config)
        print(f"LoRA applied: rank={lora_config['rank']}, "
              f"alpha={lora_config['alpha']}")

    # --- training args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy if eval_dataset else "no",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to.split(",") if args.report_to else [],
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # --- trenuj ---
    trainer.train()

    # --- zapisz checkpoint ---
    checkpoint_dir = Path(args.output_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.lora:
        model.save_pretrained(str(checkpoint_dir))
    else:
        model.save_pretrained(str(checkpoint_dir))
    processor.save_pretrained(str(checkpoint_dir))

    # metadane
    meta = {
        "model": "surya-ocr",
        "checkpoint": str(checkpoint_dir),
        "lora": args.lora,
        "config": {k: str(v) for k, v in vars(args).items()},
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
    }
    (checkpoint_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Checkpoint saved to {checkpoint_dir}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args):
    """Predykcja wytrenowanym checkpointem na zbiorze obrazów."""
    from surya.common.schema import ImageInput, TaskNames

    model, processor = load_model_and_processor(args.pretrained_checkpoint)

    # Załaduj LoRA jeśli wskazano
    if args.lora_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)
        print(f"LoRA loaded from {args.lora_checkpoint}")

    # Alternatywnie: pełny checkpoint
    elif args.checkpoint:
        chkpt = Path(args.checkpoint)
        # spróbuj załadować pełny model lub LoRA adapter
        if (chkpt / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(chkpt))
            print(f"LoRA adapter loaded from {chkpt}")
        else:
            model, processor = load_model_and_processor(str(chkpt))
            print(f"Full checkpoint loaded from {chkpt}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Wczytaj metadane
    with open(args.metadata, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]

    images_dir = Path(args.images_dir)
    results = []
    for sample in samples:
        image_path = images_dir / sample["file_name"]
        if not image_path.exists():
            results.append({"file_name": sample["file_name"], "prediction": "", "error": "file not found"})
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255.0

        scaled = processor.scale_image(image_np, (1024, 512))
        if scaled is not None:
            image_np = scaled

        inputs = {
            "task": TaskNames.ocr_with_boxes,
            "inputs": [
                ImageInput(type="image", image=image_np, rotated=False),
                TextInput(type="text", text=""),
                TextInput(type="text", text=""),
            ],
        }
        processed = processor(inputs)

        with torch.no_grad():
            # Przenieś tensory na device
            processed = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in processed.items()}
            output = model.generate(
                **{k: v for k, v in processed.items() if k not in ("labels",)},
                max_new_tokens=args.max_new_tokens,
            )

        decoded = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({
            "file_name": sample["file_name"],
            "prediction": decoded.strip(),
            "ground_truth": sample.get("text", ""),
        })

    # Zapisz wyniki
    out_path = Path(args.output) if args.output else Path("predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {out_path} ({len(results)} samples)")


# ---------------------------------------------------------------------------
# convert — konwersja raw → format Surya
# ---------------------------------------------------------------------------

def cmd_convert(args):
    """Konwersja danych z formatu CSV/JSONL do formatu Surya.

    Wejście: katalog z obrazami + labels.csv (file_name,text) lub metadata.jsonl
    Wyjście: katalog processed/surya/{train,val}/ z metadata.jsonl i images/
    """
    import shutil

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    split_ratio = args.val_split

    # Znajdź plik metadanych
    metadata_candidates = [
        input_dir / "metadata.jsonl",
        input_dir / "labels.csv",
        input_dir / "labels.jsonl",
    ]
    metadata_path = None
    for cand in metadata_candidates:
        if cand.exists():
            metadata_path = cand
            break

    if metadata_path is None:
        raise FileNotFoundError(
            f"Nie znaleziono metadanych w {input_dir}. "
            f"Oczekiwano: metadata.jsonl, labels.csv, lub labels.jsonl"
        )

    # Wczytaj metadane
    samples = []
    if metadata_path.suffix == ".csv":
        import csv
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or row.get("transcription") or "").strip()
                file_name = row.get("file_name") or row.get("filename") or row.get("image") or ""
                if text and file_name:
                    samples.append({"file_name": file_name, "text": text})
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                text = entry.get("text", "").strip()
                file_name = entry.get("file_name", "")
                if text and file_name:
                    samples.append({"file_name": file_name, "text": text})

    if not samples:
        raise ValueError(f"Brak próbek w {metadata_path}")

    # Podział train/val
    split_idx = int(len(samples) * (1 - split_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Zapisz
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        metadata_lines = []
        for s in split_samples:
            src = input_dir / s["file_name"]
            if not src.exists():
                # Szukaj w podkatalogu images/
                src = input_dir / "images" / s["file_name"]
            if src.exists():
                dst = images_dir / s["file_name"]
                if not dst.exists():
                    shutil.copy2(src, dst)
                metadata_lines.append({"file_name": s["file_name"], "text": s["text"]})
            else:
                logging.warning("Obraz nie znaleziony: %s", s["file_name"])

        meta_path = split_dir / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for line in metadata_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"{split_name}: {len(metadata_lines)} próbek → {meta_path}")

    print(f"Konwersja zakończona. Dane w {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Fine-tuning Surya OCR")
    # model
    t.add_argument("--pretrained-checkpoint", default=None,
                   help="Ścieżka do bazowego checkpointu (null = domyślny vikp/surya_ocr2)")
    t.add_argument("--max-sequence-length", type=int, default=1024)
    # dane (HuggingFace)
    t.add_argument("--dataset-name", default=None,
                   help="Nazwa datasetu HuggingFace (np. datalab-to/ocr_finetune_example)")
    # dane (lokalne)
    t.add_argument("--train-metadata", default=None,
                   help="Ścieżka do metadata.jsonl (dane lokalne)")
    t.add_argument("--train-images-dir", default=None,
                   help="Ścieżka do katalogu z obrazami (dane lokalne)")
    t.add_argument("--val-metadata", default=None)
    t.add_argument("--val-images-dir", default=None)
    t.add_argument("--num-loading-proc", type=int, default=4)
    # LoRA
    t.add_argument("--lora", action="store_true",
                   help="Użyj LoRA zamiast pełnego fine-tuningu")
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--lora-alpha", type=int, default=64)
    t.add_argument("--lora-dropout", type=float, default=0.1)
    t.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "v_proj", "o_proj"])
    # trening
    t.add_argument("--output-dir", default="results/ocr/surya/default")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--gradient-accumulation-steps", type=int, default=1)
    t.add_argument("--learning-rate", type=float, default=5e-5)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup-ratio", type=float, default=0.1)
    t.add_argument("--lr-scheduler-type", default="cosine")
    t.add_argument("--logging-steps", type=int, default=10)
    t.add_argument("--save-strategy", default="epoch")
    t.add_argument("--evaluation-strategy", default="epoch")
    t.add_argument("--save-total-limit", type=int, default=3)
    t.add_argument("--bf16", action="store_true", default=True,
                   help="Mixed precision bfloat16 (domyślnie włączone)")
    t.add_argument("--no-bf16", action="store_true",
                   help="Wyłącz bf16")
    t.add_argument("--gradient-checkpointing", action="store_true", default=True)
    t.add_argument("--no-gradient-checkpointing", action="store_true")
    t.add_argument("--dataloader-num-workers", type=int, default=4)
    t.add_argument("--report-to", default="",
                   help="Backend logowania, np. 'wandb' (pusty = brak)")
    t.add_argument("--run-name", default=None,
                   help="Nazwa runu (dla W&B)")
    t.set_defaults(fn=cmd_train)

    # --- predict ---
    p = sub.add_parser("predict", help="Predykcja wytrenowanym modelem")
    p.add_argument("--metadata", required=True,
                   help="metadata.jsonl z polami file_name, text")
    p.add_argument("--images-dir", required=True,
                   help="Katalog z obrazami")
    p.add_argument("--output", default=None,
                   help="Ścieżka wyjściowa (domyślnie predictions.jsonl)")
    p.add_argument("--checkpoint", default=None,
                   help="Ścieżka do checkpointu (pełny model lub adapter LoRA)")
    p.add_argument("--lora-checkpoint", default=None,
                   help="Ścieżka do adaptera LoRA (osobno)")
    p.add_argument("--pretrained-checkpoint", default=None,
                   help="Bazowy checkpoint (jeśli inny niż domyślny)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(fn=cmd_predict)

    # --- convert ---
    c = sub.add_parser("convert", help="Konwersja danych raw → format Surya")
    c.add_argument("--input", required=True,
                   help="Katalog wejściowy (z metadata.jsonl/labels.csv + obrazami)")
    c.add_argument("--output", required=True,
                   help="Katalog wyjściowy (processed/surya/)")
    c.add_argument("--val-split", type=float, default=0.1,
                   help="Frakcja danych na walidację (domyślnie 0.1)")
    c.set_defaults(fn=cmd_convert)

    args = ap.parse_args(argv)

    # Obsługa --no-bf16 / --no-gradient-checkpointing
    if hasattr(args, "no_bf16") and args.no_bf16:
        args.bf16 = False
    if hasattr(args, "no_gradient_checkpointing") and args.no_gradient_checkpointing:
        args.gradient_checkpointing = False

    args.fn(args)


if __name__ == "__main__":
    main()
