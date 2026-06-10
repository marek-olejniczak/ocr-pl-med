from __future__ import annotations

import json
import time
from typing import List, Iterator

import pandas as pd

from modele.base_wrapper import HTRModelWrapper
from src.data_generator import HTRSample, load_htr_samples
from src.metrics import HTRMetricsEvaluator


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


class BenchmarkRunner:
    """Orchestrator benchmarku: ladowanie danych, inferencja i metryki."""

    def __init__(self, model: HTRModelWrapper) -> None:
        self.model = model

    @staticmethod
    def evaluate_samples(samples: List[HTRSample], model: HTRModelWrapper) -> pd.DataFrame:
        if not samples:
            return pd.DataFrame(
                columns=["file_name", "document_id", "line_id", "ground_truth", "prediction"]
            )

        image_paths = [str(sample.image_path) for sample in samples]
        options = getattr(model, "options", {})
        
        predictions = []
        for batch in BenchmarkRunner.load_batch(image_paths, options):
            predictions.extend(model.predict_batch(batch))
            
        if len(predictions) != len(samples):
            raise RuntimeError(
                f"Model zwrocil {len(predictions)} predykcji dla {len(samples)} probek."
            )

        rows = []
        for sample, pred in zip(samples, predictions):
            gt_norm = _normalize_text(sample.ground_truth)
            pred_norm = _normalize_text(pred)
            rows.append(
                {
                    "file_name": sample.file_name,
                    "document_id": sample.document_id,
                    "line_id": sample.line_id,
                    "ground_truth": gt_norm,
                    "prediction": pred_norm,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_samples_with_timing(
        samples: List[HTRSample],
        model: HTRModelWrapper,
    ) -> tuple[pd.DataFrame, float]:
        if not samples:
            return pd.DataFrame(
                columns=["file_name", "document_id", "line_id", "ground_truth", "prediction"]
            ), 0.0

        image_paths = [str(sample.image_path) for sample in samples]
        options = getattr(model, "options", {})
        
        predictions = []
        prediction_seconds = 0.0
        
        for batch in BenchmarkRunner.load_batch(image_paths, options):
            prediction_started_at = time.perf_counter()
            batch_preds = model.predict_batch(batch)
            prediction_seconds += time.perf_counter() - prediction_started_at
            predictions.extend(batch_preds)

        if len(predictions) != len(samples):
            raise RuntimeError(
                f"Model zwrocil {len(predictions)} predykcji dla {len(samples)} probek."
            )

        rows = []
        for sample, pred in zip(samples, predictions):
            gt_norm = _normalize_text(sample.ground_truth)
            pred_norm = _normalize_text(pred)
            rows.append(
                {
                    "file_name": sample.file_name,
                    "document_id": sample.document_id,
                    "line_id": sample.line_id,
                    "ground_truth": gt_norm,
                    "prediction": pred_norm,
                }
            )

        return pd.DataFrame(rows), prediction_seconds

    def run(self, labels_csv: str, images_dir: str, output_dir: str, limit: int | None) -> dict:
        samples = load_htr_samples(
            labels_csv_path=labels_csv,
            images_dir_path=images_dir,
            limit=limit,
        )
        results_df = self.evaluate_samples(samples=samples, model=self.model)

        metrics_evaluator = HTRMetricsEvaluator()
        report = metrics_evaluator.build_report(results_df, self.model.model_name)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        metrics_evaluator.save_outputs(results_df, report, output_dir)
        return report

    def run_with_timing(
        self,
        labels_csv: str,
        images_dir: str,
        limit: int | None,
    ) -> tuple[pd.DataFrame, float, int]:
        samples = load_htr_samples(
            labels_csv_path=labels_csv,
            images_dir_path=images_dir,
            limit=limit,
        )
        results_df, prediction_seconds = self.evaluate_samples_with_timing(
            samples=samples,
            model=self.model,
        )
        return results_df, prediction_seconds, len(samples)
    
    @staticmethod
    def load_batch(all_image_paths: List[str], options: dict | None = None) -> Iterator[List[str]]:
        batch_size = int((options or {}).get("batch_size", 8))
        for i in range(0, len(all_image_paths), batch_size):
            yield all_image_paths[i : i + batch_size]

