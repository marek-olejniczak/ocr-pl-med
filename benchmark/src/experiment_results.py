from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, Tuple

import pandas as pd
import yaml


def _normalize_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unnamed"

    cleaned = raw.replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unnamed"


def create_run_dir(
    output_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{_normalize_name(experiment_name)}_{timestamp}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.yaml"
    meta_yaml = yaml.safe_dump(config, sort_keys=False)
    meta_path.write_text(meta_yaml, encoding="utf-8")
    return run_dir, run_name


def write_dataset_results(
    run_dir: Path,
    model_id: str,
    dataset_id: str,
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> Path:
    model_dir = run_dir / _normalize_name(model_id)
    dataset_dir = model_dir / _normalize_name(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    results_path = dataset_dir / "raw_predictions.csv"
    summary_path = dataset_dir / "summary_metrics.json"

    results_df.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return dataset_dir
