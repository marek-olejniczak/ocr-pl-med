from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


MetricFn = Callable[[Sequence[str], Sequence[str]], float]


class HTRMetricsEvaluator:
	"""Silnik metryk i raportowania dla benchmarku HTR.

	Obsluguje metryki dla poziomu: `word`, `line`, `document` oraz pozwala
	rejestrowac dodatkowe metryki bez zmian w evaluatorze glownym.
	"""

	def __init__(self) -> None:
		self._metrics: Dict[str, MetricFn] = {}
		self.register_metric("ema", self.exact_match_accuracy)
		self.register_metric("cer", self.character_error_rate)
		self.register_metric("wer", self.word_error_rate)
		self.register_metric("levenshtein_distance", self.mean_levenshtein_distance)

	def register_metric(self, name: str, func: MetricFn) -> None:
		"""Dodaje nowa metryke obliczana na parach (label, prediction)."""
		self._metrics[name] = func

	@staticmethod
	def _normalize_text(text: str) -> str:
		return " ".join(str(text).strip().split())

	@classmethod
	def _levenshtein_distance(cls, a: str, b: str) -> int:
		if a == b:
			return 0
		if not a:
			return len(b)
		if not b:
			return len(a)

		prev = list(range(len(b) + 1))
		for i, ca in enumerate(a, start=1):
			curr = [i]
			for j, cb in enumerate(b, start=1):
				ins = curr[j - 1] + 1
				dele = prev[j] + 1
				sub = prev[j - 1] + (0 if ca == cb else 1)
				curr.append(min(ins, dele, sub))
			prev = curr
		return prev[-1]

	@classmethod
	def _token_levenshtein_distance(cls, ref_tokens: List[str], pred_tokens: List[str]) -> int:
		"""
		Oblicza odleglosc Levenshteina na poziomie tokenow (np. slow) miedzy dwoma listami tokenow.
		"""
		if ref_tokens == pred_tokens:
			return 0
		if not ref_tokens:
			return len(pred_tokens)
		if not pred_tokens:
			return len(ref_tokens)

		prev = list(range(len(pred_tokens) + 1))
		for i, r_tok in enumerate(ref_tokens, start=1):
			curr = [i]
			for j, p_tok in enumerate(pred_tokens, start=1):
				ins = curr[j - 1] + 1
				dele = prev[j] + 1
				sub = prev[j - 1] + (0 if r_tok == p_tok else 1)
				curr.append(min(ins, dele, sub))
			prev = curr
		return prev[-1]

	@classmethod
	def exact_match_accuracy(cls, labels: Sequence[str], predictions: Sequence[str]) -> float:
		"""
		Procent przypadkow, w ktorych znormalizowana predykcja jest identyczna z znormalizowana etykieta.
		"""
		if not labels:
			return 0.0
		matches = sum(
			1
			for gt, pred in zip(labels, predictions)
			if cls._normalize_text(gt) == cls._normalize_text(pred)
		)
		return matches / len(labels)

	@classmethod
	def mean_levenshtein_distance(cls, labels: Sequence[str], predictions: Sequence[str]) -> float:
		"""
		Srednia odleglosc Levenshteina miedzy znormalizowanymi etykietami a predykcjami.
        """
		if not labels:
			return 0.0
		distances = [
			cls._levenshtein_distance(cls._normalize_text(gt), cls._normalize_text(pred))
			for gt, pred in zip(labels, predictions)
		]
		return float(sum(distances) / len(distances))

	@classmethod
	def character_error_rate(cls, labels: Sequence[str], predictions: Sequence[str]) -> float:
		"""
		Stosunek liczby edycji (wstawien, usuniec, zamian) do liczby znakow w znormalizowanych etykietach."""
		total_chars = 0
		total_edits = 0
		for gt, pred in zip(labels, predictions):
			gt_n = cls._normalize_text(gt)
			pred_n = cls._normalize_text(pred)
			total_chars += len(gt_n)
			total_edits += cls._levenshtein_distance(gt_n, pred_n)

		if total_chars == 0:
			return 0.0
		return total_edits / total_chars

	@classmethod
	def word_error_rate(cls, labels: Sequence[str], predictions: Sequence[str]) -> float:
		"""
		Stosunek liczby edycji (wstawien, usuniec, zamian) do liczby slow w znormalizowanych etykietach.
		"""
		total_words = 0
		total_edits = 0
		for gt, pred in zip(labels, predictions):
			gt_tokens = cls._normalize_text(gt).split()
			pred_tokens = cls._normalize_text(pred).split()
			total_words += len(gt_tokens)
			total_edits += cls._token_levenshtein_distance(gt_tokens, pred_tokens)

		if total_words == 0:
			return 0.0
		return total_edits / total_words

	@classmethod
	def _aggregate_for_level(
		cls,
		results_df: pd.DataFrame,
		level: str,
	) -> Tuple[List[str], List[str], int]:
		required_columns = {"ground_truth", "prediction", "document_id", "line_id"}
		missing = required_columns - set(results_df.columns)
		if missing:
			missing_str = ", ".join(sorted(missing))
			raise ValueError(f"Brakuje kolumn w results_df: {missing_str}")

		if level == "word":
			grouped = results_df[["ground_truth", "prediction"]].copy()
		elif level == "line":
			grouped = (
				results_df.groupby(["document_id", "line_id"], as_index=False)
				.agg(
					{
						"ground_truth": lambda x: " ".join(x.astype(str)),
						"prediction": lambda x: " ".join(x.astype(str)),
					}
				)
				.loc[:, ["ground_truth", "prediction"]]
			)
		elif level == "document":
			grouped = (
				results_df.groupby("document_id", as_index=False)
				.agg(
					{
						"ground_truth": lambda x: " ".join(x.astype(str)),
						"prediction": lambda x: " ".join(x.astype(str)),
					}
				)
				.loc[:, ["ground_truth", "prediction"]]
			)
		else:
			raise ValueError("Nieznany poziom metryk. Uzyj: word, line, document")

		labels = grouped["ground_truth"].astype(str).tolist()
		predictions = grouped["prediction"].astype(str).tolist()
		return labels, predictions, len(grouped)

	def evaluate(
		self,
		labels: Sequence[str],
		predictions: Sequence[str],
		level: str,
	) -> Dict[str, float | int | str]:
		"""
		Oblicza zarejestrowane metryki dla podanych etykiet i predykcji na danym poziomie agregacji.
		
		:param labels: Lista znormalizowanych etykiet (ground truth).
        :param predictions: Lista znormalizowanych predykcji modelu.
		:param level: Poziom agregacji ("word", "line", "document") - uzywany do raportowania, ale nie zmienia sposobu obliczania metryk.
        :return: Slownik z wynikami metryk.
		"""
		if len(labels) != len(predictions):
			raise ValueError("Liczba etykiet i predykcji musi byc taka sama")

		metrics: Dict[str, float | int | str] = {"count": int(len(labels))}
		for name, metric_fn in self._metrics.items():
			metrics[name] = float(metric_fn(labels, predictions))
		metrics["level"] = level
		return metrics

	def evaluate_from_dataframe(self, results_df: pd.DataFrame, level: str) -> Dict[str, float | int | str]:
		"""
		Oblicza metryki bezposrednio z DataFrame zawierajacego kolumny: ground_truth, prediction, document_id, line_id.
        Agregacja do poziomu "line" lub "document" jest wykonywana wewnatrz tej metody, a nastepnie obliczane sa metryki na znormalizowanych tekstach.
		
		:param results_df: DataFrame z kolumnami: ground_truth, prediction, document_id, line_id.
        :param level: Poziom agregacji ("word", "line", "document")
		
		:return: Slownik z wynikami metryk dla danego poziomu.		
		"""
		labels, predictions, count = self._aggregate_for_level(results_df, level)
		result = self.evaluate(labels, predictions, level)
		result["count"] = count
		return result

	def build_report(
		self,
		results_df: pd.DataFrame,
		model_name: str,
		levels: Iterable[str] = ("word", "line", "document"),
	) -> Dict[str, object]:
		"""Buduje raport metryk dla podanego modelu i poziomow agregacji na podstawie DataFrame z wynikami.
		
		:param results_df: DataFrame z kolumnami: ground_truth, prediction, document_id, line_id.
        :param model_name: Nazwa modelu do umieszczenia w raporcie.
        :param levels: Poziomy agregacji do obliczenia metryk (domyslnie: word, line, document)
        :return: Slownik z raportem metryk.
		"""
		report_metrics: Dict[str, Dict[str, float | int | str]] = {}
		for level in levels:
			report_metrics[level] = self.evaluate_from_dataframe(results_df, level)

		return {
			"model": model_name,
			"created_at": datetime.now().isoformat(timespec="seconds"),
			"metrics": report_metrics,
		}

	@staticmethod
	def save_outputs(results_df: pd.DataFrame, report: Dict[str, object], output_dir: str) -> None:
		out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		results_path = out_dir / f"results_{timestamp}.csv"
		report_path = out_dir / f"summary_{timestamp}.json"

		results_df.to_csv(results_path, index=False)
		report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

		print(f"Zapisano szczegolowe wyniki: {results_path}")
		print(f"Zapisano podsumowanie: {report_path}")
