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
	if not samples:
		return pd.DataFrame(
			columns=["file_name", "document_id", "line_id", "ground_truth", "prediction"]
		)

	image_paths = [str(sample.image_path) for sample in samples]
	predictions = model.predict_batch(image_paths)
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


def _resolve_trocr_model_id(args: argparse.Namespace) -> str:
	if args.trocr_model_id:
		return args.trocr_model_id

	if args.trocr_variant == "base":
		return "microsoft/trocr-base-handwritten"

	return "microsoft/trocr-small-handwritten"


def _resolve_paddleocr_rec_model_name(args: argparse.Namespace) -> str:
	if args.paddleocr_rec_model_name:
		return args.paddleocr_rec_model_name

	if args.paddleocr_variant == "server":
		return "PP-OCRv4_server_rec"

	return "PP-OCRv4_mobile_rec"


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
			batch_size=args.rysocr_batch_size,
			use_amp=args.rysocr_use_amp,
		)

	if args.model == "trocr":
		from modele.trocr_wrapper import TrOCRWrapper

		return TrOCRWrapper(
			model_id=_resolve_trocr_model_id(args),
			max_new_tokens=args.trocr_max_new_tokens,
			device=args.trocr_device,
			local_files_only=args.trocr_local_files_only,
			batch_size=args.trocr_batch_size,
			use_amp=args.trocr_use_amp,
		)

	if args.model == "paddleocr":
		from modele.paddleocr_wrapper import PaddleOCRWrapper

		return PaddleOCRWrapper(
			rec_model_name=_resolve_paddleocr_rec_model_name(args),
			lang=args.paddleocr_lang,
			device=args.paddleocr_device,
			use_angle_cls=args.paddleocr_use_angle_cls,
			rec_batch_size=args.paddleocr_rec_batch_size,
		)

	raise ValueError(f"Nieobslugiwany model: {args.model}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark HTR")
	parser.add_argument(
		"--model",
		type=str,
		choices=["tesseract_pol", "rysocr", "trocr", "paddleocr"],
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
	parser.add_argument(
		"--rysocr-batch-size",
		type=int,
		default=2,
		help="Rozmiar batcha inferencji dla RysOCR (domyslnie: 2).",
	)
	parser.add_argument(
		"--rysocr-use-amp",
		action="store_true",
		help="Wlacz mixed precision (AMP) dla RysOCR na CUDA.",
	)
	parser.add_argument(
		"--trocr-variant",
		type=str,
		choices=["small", "base"],
		default="small",
		help="Wariant TrOCR handwritten. small: lzejszy VRAM, base: ciezszy i zwykle dokladniejszy.",
	)
	parser.add_argument(
		"--trocr-model-id",
		type=str,
		default=None,
		help=(
			"Nadpisz domyslne repozytorium TrOCR na Hugging Face. "
			"Gdy pominiete, wybor wynika z --trocr-variant."
		),
	)
	parser.add_argument(
		"--trocr-max-new-tokens",
		type=int,
		default=128,
		help="Maksymalna liczba tokenow generowanych przez TrOCR.",
	)
	parser.add_argument(
		"--trocr-device",
		type=str,
		default=None,
		help="Urzadzenie dla TrOCR (np. cpu, cuda). Domyslnie wybierane automatycznie.",
	)
	parser.add_argument(
		"--trocr-local-files-only",
		action="store_true",
		help="Tryb offline: laduj TrOCR tylko z lokalnego cache Hugging Face, bez pobierania.",
	)
	parser.add_argument(
		"--trocr-batch-size",
		type=int,
		default=4,
		help="Rozmiar batcha inferencji dla TrOCR (domyslnie: 4).",
	)
	parser.add_argument(
		"--trocr-use-amp",
		action="store_true",
		help="Wlacz mixed precision (AMP) dla TrOCR na CUDA.",
	)
	parser.add_argument(
		"--paddleocr-variant",
		type=str,
		choices=["mobile", "server"],
		default="mobile",
		help="Wariant PP-OCRv4 dla rozpoznawania: mobile (lekki) lub server (dokladniejszy).",
	)
	parser.add_argument(
		"--paddleocr-rec-model-name",
		type=str,
		default=None,
		help=(
			"Nadpisz nazwe modelu rozpoznawania PaddleOCR. "
			"Gdy pominiete, wybor wynika z --paddleocr-variant."
		),
	)
	parser.add_argument(
		"--paddleocr-lang",
		type=str,
		default="pl",
		help="Jezyk PaddleOCR (uzywany glownie w fallbacku legacy OCR). Dla PL: pl.",
	)
	parser.add_argument(
		"--paddleocr-device",
		type=str,
		choices=["auto", "cpu", "gpu"],
		default="auto",
		help="Urzadzenie PaddleOCR: auto, cpu lub gpu.",
	)
	parser.add_argument(
		"--paddleocr-use-angle-cls",
		action="store_true",
		help="Wlacz klasyfikator kata (CLS) w PaddleOCR.",
	)
	parser.add_argument(
		"--paddleocr-rec-batch-size",
		type=int,
		default=8,
		help="Batch size dla rozpoznawania PaddleOCR (domyslnie: 8).",
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
