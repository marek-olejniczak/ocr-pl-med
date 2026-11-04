"""Callback CER na zbiorze handlabeled dla fine-tuningu Surya LoRA.

Po co: `metric_for_best_model="eval_loss"` jest w tym zadaniu zawodny. Walidacja
w treningu jest z tego samego syntetycznego generatora co trening, więc niższy
`eval_loss` znaczy „lepiej wpasował się w artefakty augmentacji", a nie „lepiej
czyta pismo" — patrz `training/ocr/surya/hp_search/README.md`. Ten callback liczy
w trakcie treningu prawdziwy CER na ręcznie olabelowanych liniach
(`benchmark/dane/handlabeled`) i podaje go Trainerowi jako `eval_cer`, więc
`load_best_model_at_end` wybiera checkpoint po metryce, która ma znaczenie.

Implementacja: subprocess `hp_search/eval_cer.py` na świeżo zapisanym adapterze.
Świadomie NIE liczymy predykcji w procesie treningowym:
  * `eval_cer.py` jest zweryfikowany co do cyfry z benchmarkiem (smoke parity
    baseline -> CER 0.127243), więc nie ryzykujemy cichego rozjazdu metryki;
  * subprocess nie podwaja pamięci modelu w procesie treningu i można go
    skierować na drugą, wolną kartę (`--cer-eval-gpu 1`);
  * błąd ewaluacji nie może wywalić treningu — łapiemy go i wpisujemy sentinel.

Użycie (patrz `main()` w `cli.py`):
    python training/ocr/surya/cli.py ... --cer-callback \
        --cer-labels-csv benchmark/dane/handlabeled/labels.csv \
        --cer-images-dir benchmark/dane/handlabeled/images \
        --cer-eval-gpu 1 --cer-eval-batch-size 16
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# cer_callback.py leży w <root>/training/ocr/surya/ -> parents[3] to root checkoutu.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_SCRIPT = _REPO_ROOT / "training" / "ocr" / "surya" / "hp_search" / "eval_cer.py"

# Wartość wpisywana, gdy ewaluacja CER się nie powiedzie. Duża, więc przy
# greater_is_better=False `load_best_model_at_end` nigdy nie wybierze takiego
# checkpointu, a trening leci dalej.
_FAILED_CER = 1e9


class CERCallback(TrainerCallback):
    """Liczy CER na handlabeled po każdej ewaluacji (subprocess `eval_cer.py`)."""

    def __init__(self, *, labels_csv, images_dir, eval_gpu=1,
                 batch_size=16, max_samples=None, output_json=None):
        self.labels_csv = str(labels_csv)
        self.images_dir = str(images_dir)
        # Fizyczny indeks karty dla subprocessu (ustawiany jako
        # CUDA_VISIBLE_DEVICES, więc wewnątrz widzi ją jako cuda:0).
        # None/-1 = nie zmieniaj widoczności, użyj tej samej karty co trening.
        self.eval_gpu = eval_gpu
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.output_json = Path(output_json) if output_json else None
        # Ustawiane w cli.py po konstrukcji Trainera (callback nie dostaje
        # referencji do modelu w on_evaluate, więc musimy ją wstrzyknąć).
        self.trainer = None

    def _result_path(self, args) -> Path:
        if self.output_json is not None:
            return self.output_json
        return Path(args.output_dir) / "_cer_eval" / "result.json"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}
        if self.trainer is None:
            logger.warning("CERCallback: brak referencji do Trainera, pomijam eval CER")
            metrics["eval_cer"] = _FAILED_CER
            return control

        adapter_dir = Path(args.output_dir) / "_cer_eval" / "adapter"
        result_path = self._result_path(args)
        try:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.model.save_pretrained(str(adapter_dir))

            cmd = [
                sys.executable, str(_EVAL_SCRIPT),
                "--adapter", str(adapter_dir),
                "--labels-csv", self.labels_csv,
                "--images-dir", self.images_dir,
                "--device", "cuda",
                "--batch-size", str(self.batch_size),
                "--output-json", str(result_path),
            ]
            if self.max_samples:
                cmd += ["--max-samples", str(self.max_samples)]

            # Eval na innej karcie niż trening: subprocess widzi wyłącznie
            # wskazany fizyczny GPU (logicznie cuda:0), więc nie konkuruje
            # o pamięć z procesem treningowym.
            env = os.environ.copy()
            if self.eval_gpu is not None and int(self.eval_gpu) >= 0:
                env["CUDA_VISIBLE_DEVICES"] = str(int(self.eval_gpu))

            logger.info("CERCallback: step %s -> %s", state.global_step, " ".join(cmd))
            proc = subprocess.run(
                cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, env=env
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"eval_cer.py zwrócił kod {proc.returncode}\n"
                    f"--- stdout (ogon) ---\n{proc.stdout[-2000:]}\n"
                    f"--- stderr (ogon) ---\n{proc.stderr[-2000:]}"
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            cer = float(payload["metrics"]["cer"])
            wer = float(payload["metrics"]["wer"])
            metrics["eval_cer"] = cer
            metrics["eval_wer"] = wer
            # Trainer.evaluate() loguje output.metrics PRZED callbackiem, więc
            # samo wpisanie do `metrics` trafia tylko do selekcji best-model
            # (load_best_model_at_end). Żeby eval_cer było też w konsoli/W&B,
            # logujemy je jawnie.
            self.trainer.log({"eval_cer": cer, "eval_wer": wer})
            logger.info(
                "CERCallback: step %s -> CER=%.6f WER=%.6f (%s próbek)",
                state.global_step, cer, wer, payload.get("count"),
            )
        except Exception as exc:  # noqa: BLE001 — ewaluacja nie może przerwać treningu
            logger.warning(
                "CERCallback: eval CER NIE udał się (%s); wpisuję %s", exc, _FAILED_CER
            )
            metrics["eval_cer"] = _FAILED_CER
        return control
