"""Fine-tuning CLI dla TrOCR (VisionEncoderDecoderModel, polski wariant).

Cel: dotrenować `PET3R12/trocr-base-polish-handwriting` (standardowy TrOCR: encoder
ViT 384 + decoder BART, vocab 50265, tokenizer Roberta) na danych ocr_800k — tych
samych, na których trenowaliśmy Surya. W przeciwieństwie do Suryi robimy PEŁNY
fine-tuning (nie LoRA): angielski słownik dekodera TrOCR "kaleczy" polskie znaki,
więc pełny trening ma nadpisać wagi dekodera i nauczyć polskich transkrypcji.

Format danych — ten sam co Surya (bez nowej konwersji):
    training/data/processed/surya800k/{train,val}/metadata.jsonl  # {"file_name": ..., "text": ...}
    training/data/processed/surya800k/{train,val}/images/         # płaskie obrazy linii

Użycie:
    python cli.py train \
        --model-id PET3R12/trocr-base-polish-handwriting \
        --train-metadata training/data/processed/surya800k/train/metadata.jsonl \
        --train-images-dir training/data/processed/surya800k/train/images \
        --val-metadata training/data/processed/surya800k/val/metadata.jsonl \
        --val-images-dir training/data/processed/surya800k/val/images \
        --output-dir training/results/ocr/trocr/run01 \
        --max-steps 40000 --report-to wandb

UWAGA (transformers): `PET3R12/trocr-base-polish-handwriting` jest zapisany z
transformers 5.0.0 (plik `processor_config.json`, nowe nazewnictwo zamiast
`preprocessor_config.json`). Ten pipeline wymaga transformers >= 5.0 — obraz
treningowy `training-ocr-trocr` go ma; obraz `training-surya-training` (4.56.1)
NIE nadaje się do ładowania tego modelu.
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


class TrOCRDataCollator:
    """Collator dla TrOCR.

    W transformers 5.0.0 default_data_collator NIE paduje już labels (stary
    _torch_collate_batch z -100 zniknął) — batch z labels o różnych długościach
    wywala się na torch.stack. Ten collator stackuje pixel_values (wszystkie
    próbki są 384x384, więc równe) i paduje labels do długości najdłuższej
    próbki w batchu wartością label_pad_token_id (-100, maskowane w loss).
    """

    def __init__(self, label_pad_token_id: int = -100):
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = [f["labels"] for f in features]
        max_len = max(len(l) for l in labels)
        padded = torch.full(
            (len(labels), max_len), self.label_pad_token_id, dtype=torch.long
        )
        for i, l in enumerate(labels):
            n = len(l)
            padded[i, :n] = l
        return {
            "pixel_values": pixel_values,
            "labels": padded,
        }


class TrOCRDataset(torch.utils.data.Dataset):
    """Dataset ładujący dane z metadata.jsonl + images/ (jak LocalOCRDataset u Suryi)."""

    def __init__(self, metadata_path: str, images_dir: str, processor, max_length: int = 512):
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.max_length = max_length
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
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                alt = image_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                logging.warning("Obraz nie znaleziony: %s, pomijanie", image_path)
                return self[(index + 1) % len(self)]

        with Image.open(image_path).convert("RGB") as image:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)  # (3, 384, 384)

        # Etykiety = tokeny tekstu; pad_token -> -100 (ignorowane w loss).
        labels = self.processor.tokenizer(
            sample["text"], truncation=True, max_length=self.max_length
        ).input_ids
        labels = [tok if tok != self.processor.tokenizer.pad_token_id else -100 for tok in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def cmd_train(args):
    """Pełny fine-tuning TrOCR (encoder ViT + decoder TrOCR)."""
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if not (args.train_metadata and args.train_images_dir):
        raise ValueError("Podaj --train-metadata i --train-images-dir (dane lokalne)")

    model_id = args.model_id
    processor = TrOCRProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_id, cache_dir=args.cache_dir)

    # Dekoder TrOCR wymaga jawnych id tokenów specjalnych. UWAGA: NIE nadpisujemy
    # decoder_start_token_id — polski checkpoint ma własne (2 = eos </s>, wg
    # generation_config.json), nadpisanie bos_token_id (0) zepsułoby generowanie.
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # UWAGA: nie ustawiamy model.config.max_length — transformers 5.0.0 odmawia
    # zapisu parametrów generacji w model.config (ValueError przy save_pretrained).
    # Długość generowania i tak ustawia training_args.generation_max_length, a
    # benchmark przekazuje max_new_tokens jawnie.

    train_dataset = TrOCRDataset(
        args.train_metadata, args.train_images_dir, processor, args.max_length)
    eval_dataset = None
    if args.val_metadata and args.val_images_dir:
        eval_dataset = TrOCRDataset(
            args.val_metadata, args.val_images_dir, processor, args.max_length)

    report_to = [r for r in (args.report_to or "").split(",") if r]

    eval_strategy = args.eval_strategy if eval_dataset else "no"
    save_strategy = args.save_strategy
    if eval_strategy == "steps":
        save_strategy = "steps"

    training_args = Seq2SeqTrainingArguments(
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
        save_steps=args.eval_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        # Bez predict_with_generate: eval z generowaniem liczyłby tekst na całym
        # val (39934 obrazów) i kosztował ~1-2h na KAŻDY eval (×8 przy 40k
        # kroków = 8-16h nadgodzin). Eval = sam eval_loss (jeden forward, szybki);
        # jakość transkrypcji i tak zmierzy benchmark na handlabeled (CER/WER).
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrOCRDataCollator(),
    )

    trainer.train()

    meta = {
        "model": "trocr-polish-handwriting",
        "base_model": model_id,
        "config": {k: str(v) for k, v in vars(args).items()},
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
    }

    # Pełny checkpoint (model + processor) — bezpośrednio czytelny przez
    # benchmark (model_id = katalog), bez peft.
    save_dir = Path(args.output_dir)
    trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Model zapisany w {save_dir}")

    if "wandb" in report_to:
        import wandb
        # UWAGA: NIE logujemy modelu jako artefaktu W&B — pełny checkpoint TrOCR
        # to ~1.3 GB safetensors, a upload (szczególnie ×2: best + last) potrafi
        # zawiesić proces na długo po zakończeniu treningu. Checkpoint jest na
        # dysku (bind mount home), benchmark czyta go stamtąd. Jawnie zamykamy
        # run, żeby background-upload nie trzymał procesu przy życiu.
        if wandb.run is not None:
            wandb.finish()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", metavar="{train}", required=True)
    train_p = sub.add_parser("train", help="Pełny fine-tuning TrOCR (Seq2SeqTrainer)")

    # model
    train_p.add_argument("--model-id", default="PET3R12/trocr-base-polish-handwriting",
                        help="Bazowy model TrOCR (HF id lub lokalny katalog)")
    train_p.add_argument("--cache-dir", default="modele/cache/trocr",
                        help="Cache modelu (współdzielony z benchmarkiem)")
    train_p.add_argument("--max-length", type=int, default=512,
                        help="Maks. długość sekwencji dekodera (config modelu: 512)")
    # dane (lokalne, format Surya)
    train_p.add_argument("--train-metadata", default=None, help="metadata.jsonl (train)")
    train_p.add_argument("--train-images-dir", default=None, help="Katalog obrazów (train)")
    train_p.add_argument("--val-metadata", default=None, help="metadata.jsonl (val)")
    train_p.add_argument("--val-images-dir", default=None, help="Katalog obrazów (val)")
    # trening
    train_p.add_argument("--output-dir", default="training/results/ocr/trocr/default")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--max-steps", type=int, default=-1,
                        help="Limit kroków treningu (-1 = bez limitu; do smoke testu np. 100)")
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--learning-rate", type=float, default=2e-5)
    train_p.add_argument("--weight-decay", type=float, default=0.05)
    train_p.add_argument("--warmup-ratio", type=float, default=0.1)
    train_p.add_argument("--lr-scheduler-type", default="cosine")
    train_p.add_argument("--logging-steps", type=int, default=10)
    train_p.add_argument("--save-strategy", default="epoch")
    train_p.add_argument("--eval-strategy", default="steps")
    train_p.add_argument("--eval-steps", type=int, default=5000,
                        help="Co ile kroków ewaluacja (wymaga --eval-strategy steps)")
    train_p.add_argument("--save-total-limit", type=int, default=3)
    train_p.add_argument("--bf16", action="store_true", default=True,
                        help="Mixed precision bfloat16 (Ampere+; domyślnie włączone)")
    train_p.add_argument("--no-bf16", action="store_true", help="Wyłącz bf16")
    train_p.add_argument("--dataloader-num-workers", type=int, default=0)
    train_p.add_argument("--report-to", default="",
                        help="Backend logowania, np. 'wandb' (pusty = brak)")
    train_p.add_argument("--run-name", default=None, help="Nazwa runu (dla W&B)")

    # Uwaga: parsem jest ap, nie train_p — ap.parse_args obsługuje subkomendę
    # (dowolny argv zaczynający się od 'train') i merguje opcje train_p do
    # namespace. train_p.parse_args(argv) parsowałby argv ponownie i 'train'
    # wyskoczyłby jako nieznany argument.
    args = ap.parse_args(argv)

    if args.no_bf16:
        args.bf16 = False

    cmd_train(args)


if __name__ == "__main__":
    main()
