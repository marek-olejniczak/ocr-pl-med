from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

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


def _latex_escape(value: str) -> str:
    if value is None:
        return ""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _format_number(value: Any, *, decimals: int = 4) -> str:
    if value is None:
        return "n/a"

    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(num):
        return "n/a"
    return f"{num:.{decimals}f}"


def _collect_metric_keys(metrics_list: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "ema",
        "near_perfect_match",
        "cer",
        "wer",
        "levenshtein_distance",
    ]
    excluded = {"count", "level"}
    extras = set()
    for metrics in metrics_list:
        for key in metrics.keys():
            if key in excluded or key in preferred:
                continue
            extras.add(key)

    ordered = [key for key in preferred if any(key in m for m in metrics_list)]
    ordered.extend(sorted(extras))
    return ordered


def _metric_label(metric_key: str) -> str:
    labels = {
        "ema": "EMA",
        "near_perfect_match": "NPM",
        "cer": "CER",
        "wer": "WER",
        "levenshtein_distance": "LevDist",
    }
    return labels.get(metric_key, metric_key)


def export_overleaf_tables(
    run_dir: Path,
    config: Dict[str, Any],
    output_name: str = "overleaf_tables.tex",
) -> Path:
    datasets = config.get("datasets", [])
    models = [
        model
        for model in config.get("models", [])
        if model.get("enabled", True)
    ]

    lines: List[str] = []

    for dataset in datasets:
        if not dataset.get("enabled", True):
            continue
        dataset_id = str(dataset.get("id", "unnamed"))
        dataset_dir_id = _normalize_name(dataset_id)
        metrics_level = "word" if dataset.get("single_words", False) else "line"

        summaries: Dict[str, Dict[str, Any]] = {}
        metrics_for_columns: List[Dict[str, Any]] = []

        for model in models:
            model_id = str(model.get("id", "unnamed"))
            model_dir_id = _normalize_name(model_id)
            summary_path = (
                run_dir
                / model_dir_id
                / dataset_dir_id
                / "summary_metrics.json"
            )
            summary: Dict[str, Any] = {}
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    summary = {}

            metrics = summary.get("metrics", {}).get(metrics_level, {})
            if not metrics and summary.get("metrics"):
                metrics = next(iter(summary.get("metrics", {}).values()), {})
            timing = summary.get("timing", {}) if summary else {}

            summaries[model_id] = {"metrics": metrics, "timing": timing}
            if metrics:
                metrics_for_columns.append(metrics)

        metric_keys = _collect_metric_keys(metrics_for_columns)
        headers = ["Model"]
        headers.extend([_latex_escape(_metric_label(key)) for key in metric_keys])
        headers.append("AvgTimeS")

        column_alignment = "l" + "r" * (len(headers) - 1)
        caption = _latex_escape(f"Wyniki dla datasetu {dataset_id}")
        label = _latex_escape(f"tab:{_normalize_name(dataset_id)}")

        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{{column_alignment}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\hline")

        for model in models:
            model_id = str(model.get("id", "unnamed"))
            summary_entry = summaries.get(model_id, {})
            metrics = summary_entry.get("metrics", {}) or {}
            timing = summary_entry.get("timing", {}) or {}

            row = [_latex_escape(model_id)]
            for key in metric_keys:
                row.append(_format_number(metrics.get(key)))
            row.append(_format_number(timing.get("prediction_seconds_per_sample")))

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    output_path = run_dir / output_name
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path
