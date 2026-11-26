"""Ocena tekstu przez Suryę: log P(tekst | obraz linii), teacher forcing (ta sama ścieżka co strata w treningu LoRA).

Wymaga surya-ocr==0.17.1 i modelu z adapterem (tak jak `surya_worker.py` w aplikacji); moduł nie importuje kodu
treningowego. Używany przez `Corrector.correct_lines_rescored` do wyboru między kandydatami korekty.

    from surya.foundation import FoundationPredictor
    fp = FoundationPredictor(); fp.model = PeftModel.from_pretrained(fp.model, ADAPTER)
    scorer = SuryaScorer(fp)
    scorer.score(pil_image, ["tekst a", "tekst b"])   # -> [logP(a), logP(b)]
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MAX_IMAGE_SIZE = (1024, 512)  # jak w treningu (training/ocr/surya/cli.py)


class SuryaScorer:
    def __init__(self, foundation_predictor, max_batch: int = 16):
        import torch
        from surya.common.surya.schema import TaskNames

        self.torch = torch
        self.model = foundation_predictor.model
        self.base = getattr(self.model, "base_model", None)
        self.base = self.base.model if self.base is not None and hasattr(self.base, "model") else self.model
        self.proc = foundation_predictor.processor
        self.task = TaskNames.ocr_with_boxes  # zadanie, na którym uczono LoRA
        self.max_batch = max_batch
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

    def _item(self, img_np, text: str) -> dict:
        from surya.common.surya.processor.schema import ImageInput, TextInput
        from surya.common.util import SCRIPT_TOKEN_MAPPING, get_top_scripts

        pref = "".join(SCRIPT_TOKEN_MAPPING[s] for s in get_top_scripts(text) if s in SCRIPT_TOKEN_MAPPING)
        return {"task": self.task, "inputs": [ImageInput(type="image", image=img_np, rotated=False),
                                              TextInput(type="text", text=""), TextInput(type="text", text=pref + text)]}

    def _batch(self, items: list[dict]) -> dict:
        torch, p = self.torch, self.proc
        b = p(items, padding_side="right")
        seq = b["input_ids"].shape[1]
        b["cache_position"] = torch.arange(0, seq, dtype=torch.long)
        tiles, grid = b.pop("image_tiles"), b.pop("grid_thw")
        b["image_embeddings"] = self.base.get_image_embeddings(
            pixel_values=tiles.to(self.device), grid_thw=grid.to(self.device), encoder_chunk_size=32768)
        labels = b["input_ids"].clone()
        skip = ((labels == p.pad_token_id) | (labels == p.bos_token_id[self.task]) | (labels == p.eoi_token_id)
                | (labels == p.image_token_id))
        labels[skip] = -100
        b["labels"] = labels
        if getattr(self.base.config, "_attn_implementation", None) != "flash_attention_2":
            n, s = b["input_ids"].shape
            dt = self.dtype if self.dtype in (torch.bfloat16, torch.float16) else torch.float32
            lo = torch.finfo(dt).min
            causal = torch.full((n, 1, s, s), lo, dtype=dt)
            causal = causal.masked_fill(torch.tril(torch.ones(s, s, dtype=torch.bool))[None, None], 0.0)
            pad = b["attention_mask"] == 0
            causal = causal.masked_fill(pad[:, None, None, :], lo).masked_fill(pad[:, None, :, None], lo)
            b["attention_mask"] = causal
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def score(self, image, texts: Sequence[str]) -> list[float]:
        """Suma logprob tokenów tekstu (z prefiksem skryptu i EOS) dla każdego tekstu, przy tym samym obrazie."""
        torch = self.torch
        img = np.asarray(image.convert("RGB"), dtype=np.float32)
        img = self.proc.scale_to_fit(img, max_size=MAX_IMAGE_SIZE)
        out: list[float] = []
        with torch.no_grad():
            for s in range(0, len(texts), self.max_batch):
                b = self._batch([self._item(img, t) for t in texts[s:s + self.max_batch]])
                labels = b["labels"]
                lp = torch.log_softmax(self.model(**b).lm_logits.float(), dim=-1)
                tgt = torch.roll(labels, shifts=-1, dims=-1)
                mask = tgt != -100
                g = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
                out += g.sum(dim=1).tolist()
        return out
