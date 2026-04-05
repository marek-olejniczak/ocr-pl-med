from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from modele.base_wrapper import HTRModelWrapper
from modele.tesseract_pol_wrapper import TesseractPolWrapper
from src.data_generator import HTRSample, load_htr_samples
from src.metrics import HTRMetricsEvaluator


def _normalize_text(text: str) -> str:
	return " ".join(text.strip().split())


def evaluate_samples(samples: List[HTRSample], model: HTRModelWrapper) -> pd.DataFrame:
	rows = []
	for sample in samples:
		pred = model.predict(str(sample.image_path))
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


def build_model(args: argparse.Namespace) -> HTRModelWrapper:
	if args.model == "tesseract_pol":
		return TesseractPolWrapper(
			language=args.lang,
			psm=args.psm,
			oem=args.oem,
			tesseract_cmd=args.tesseract_cmd,
		)

	if args.model == "rysocr":
		from modele.rysocr_wrapper import RysOCRWrapper

		return RysOCRWrapper(
			adapter_model_id=args.rysocr_adapter,
			base_model_id=args.rysocr_base,
			prompt=args.rysocr_prompt,
			max_new_tokens=args.rysocr_max_new_tokens,
			device=args.rysocr_device,
			local_files_only=args.rysocr_local_files_only,
		)

	raise ValueError(f"Nieobslugiwany model: {args.model}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark HTR")
	parser.add_argument(
		"--model",
		type=str,
		choices=["tesseract_pol", "rysocr"],
		default="tesseract_pol",
		help="Model OCR do uruchomienia",
	)
	parser.add_argument(
		"--labels-csv",
		type=str,
		default="dane/htr_lexicography-main/data/PL-20k-hand-labelled.csv", # nie zmieniac
		help="Sciezka do CSV z etykietami (kolumny: file_name,label)",
	)
	parser.add_argument(
		"--images-dir",
		type=str,
		default="dane/htr_lexicography-main/data/PL-20k-hand-labelled",
		help="Katalog z obrazami",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="wyniki",
		help="Katalog na raporty i wyniki CSV",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit liczby probek (przydatne do szybkich testow)",
	)
	parser.add_argument(
		"--lang",
		type=str,
		default="pol",
		help="Kod jezyka Tesseract",
	)
	parser.add_argument("--psm", type=int, default=7, help="Page segmentation mode")
	parser.add_argument("--oem", type=int, default=1, help="OCR engine mode")
	parser.add_argument(
		"--tesseract-cmd",
		type=str,
		default=None,
		help="Pelna sciezka do binarki tesseract (gdy nie ma jej w PATH)",
	)
	parser.add_argument(
		"--rysocr-adapter",
		type=str,
		default="kacperwikiel/RysOCR",
		help="Repozytorium adaptera LoRA RysOCR na Hugging Face",
	)
	parser.add_argument(
		"--rysocr-base",
		type=str,
		default="PaddlePaddle/PaddleOCR-VL",
		help="Repozytorium modelu bazowego dla RysOCR",
	)
	parser.add_argument(
		"--rysocr-prompt",
		type=str,
		default="Transcribe the text exactly.",
		help="Prompt dla modelu RysOCR",
	)
	parser.add_argument(
		"--rysocr-max-new-tokens",
		type=int,
		default=256,
		help="Maksymalna liczba tokenow generowanych przez RysOCR",
	)
	parser.add_argument(
		"--rysocr-device",
		type=str,
		default=None,
		help="Urzadzenie dla RysOCR (np. cpu, cuda). Domyslnie wybierane automatycznie.",
	)
	parser.add_argument(
		"--rysocr-local-files-only",
		action="store_true",
		help="Tryb offline: laduj tylko z lokalnego cache Hugging Face, bez pobierania.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	samples = load_htr_samples(
		labels_csv_path=args.labels_csv,
		images_dir_path=args.images_dir,
		limit=args.limit,
	)
	model = build_model(args)

	results_df = evaluate_samples(samples, model)
	metrics_evaluator = HTRMetricsEvaluator()
	report = metrics_evaluator.build_report(results_df, model.model_name)

	print(json.dumps(report, ensure_ascii=False, indent=2))
	metrics_evaluator.save_outputs(results_df, report, args.output_dir)


if __name__ == "__main__":
	main()
