from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from modele.base_wrapper import HTRModelWrapper
from modele.tesseract_pol_wrapper import TesseractPolWrapper
from orchestrator.benchmark import BenchmarkRunner
from orchestrator.client import HTTPModelWrapper


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


def _parse_easyocr_langs(raw_langs: str) -> list[str]:
	langs = [lang.strip() for lang in raw_langs.split(",") if lang.strip()]
	if not langs:
		return ["pl"]
	return langs


def _resolve_http_base_url(args: argparse.Namespace) -> str:
	if args.http_base_url:
		return args.http_base_url.rstrip("/")

	defaults = {
		"easyocr": "http://localhost:8001",
		"trocr": "http://localhost:8002",
		"paddleocr": "http://localhost:8003",
		"parseq": "http://localhost:8004",
		"calamari": "http://localhost:8005",
		"rysocr": "http://localhost:8006",
		"tesseract_pol": "http://localhost:8007",
		"surya": "http://localhost:8008",
	}
	return defaults[args.model]


def _build_http_options(args: argparse.Namespace) -> dict:
	if args.model == "easyocr":
		return {
			"langs": _parse_easyocr_langs(args.easyocr_langs),
			"device": args.easyocr_device,
			"batch_size": args.easyocr_batch_size,
		}

	if args.model == "trocr":
		return {
			"model_id": _resolve_trocr_model_id(args),
			"max_new_tokens": args.trocr_max_new_tokens,
			"device": args.trocr_device,
			"local_files_only": args.trocr_local_files_only,
			"batch_size": args.trocr_batch_size,
			"use_amp": args.trocr_use_amp,
			"cache_dir": args.trocr_cache_dir,
		}

	if args.model == "paddleocr":
		return {
			"rec_model_name": _resolve_paddleocr_rec_model_name(args),
			"lang": args.paddleocr_lang,
			"device": args.paddleocr_device,
			"use_angle_cls": args.paddleocr_use_angle_cls,
			"rec_batch_size": args.paddleocr_rec_batch_size,
			"cache_dir": args.paddleocr_cache_dir,
		}

	if args.model == "parseq":
		return {
			"device": args.parseq_device,
			"batch_size": args.parseq_batch_size,
			"cache_dir": args.parseq_cache_dir,
			"input_size": args.parseq_input_size,
			"use_amp": args.parseq_use_amp,
			"language": args.parseq_lang,
			"model_id": args.parseq_model_id,
			"local_files_only": args.parseq_local_files_only,
		}

	if args.model == "calamari":
		return {
			"model": args.calamari_model,
			"batch_size": args.calamari_batch_size,
			"cache_dir": args.calamari_cache_dir,
			"local_files_only": args.calamari_local_files_only,
			"checkpoints": args.calamari_checkpoints,
			"device": args.calamari_device,
		}

	if args.model == "rysocr":
		return {
			"adapter_model_id": args.rysocr_adapter,
			"base_model_id": args.rysocr_base,
			"prompt": args.rysocr_prompt,
			"max_new_tokens": args.rysocr_max_new_tokens,
			"device": args.rysocr_device,
			"local_files_only": args.rysocr_local_files_only,
			"batch_size": args.rysocr_batch_size,
			"use_amp": args.rysocr_use_amp,
			"cache_dir": args.rysocr_cache_dir,
		}

	if args.model == "surya":
		return {
			"device": args.surya_device,
			"batch_size": args.surya_batch_size,
			"task_name": args.surya_task_name,
			"disable_math": args.surya_disable_math,
			"cache_dir": args.surya_cache_dir,
		}

	return {
		"language": args.lang,
		"psm": args.psm,
		"oem": args.oem,
		"tesseract_cmd": args.tesseract_cmd,
	}


def build_model(args: argparse.Namespace) -> HTRModelWrapper:
	if args.inference_mode == "http":
		return HTTPModelWrapper(
			model_name=f"{args.model}_http",
			base_url=_resolve_http_base_url(args),
			timeout_seconds=args.http_timeout,
			options=_build_http_options(args),
		)

	if args.model == "surya":
		raise RuntimeError(
			"Model 'surya' jest wspierany tylko w trybie HTTP. "
			"Uzyj: --model surya --inference-mode http"
		)

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
			cache_dir=args.rysocr_cache_dir,
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
			cache_dir=args.trocr_cache_dir,
		)

	if args.model == "paddleocr":
		from modele.paddleocr_wrapper import PaddleOCRWrapper

		return PaddleOCRWrapper(
			rec_model_name=_resolve_paddleocr_rec_model_name(args),
			lang=args.paddleocr_lang,
			device=args.paddleocr_device,
			use_angle_cls=args.paddleocr_use_angle_cls,
			rec_batch_size=args.paddleocr_rec_batch_size,
			cache_dir=args.paddleocr_cache_dir,
		)

	if args.model == "easyocr":
		from modele.easyocr_wrapper import EasyOCRWrapper

		return EasyOCRWrapper(
			langs=_parse_easyocr_langs(args.easyocr_langs),
			device=args.easyocr_device,
			batch_size=args.easyocr_batch_size,
			model_storage_dir=args.easyocr_model_storage_dir,
		)

	if args.model == "parseq":
		from modele.parseq_wrapper import PARSeqWrapper

		return PARSeqWrapper(
			device=args.parseq_device,
			batch_size=args.parseq_batch_size,
			cache_dir=args.parseq_cache_dir,
			input_size=args.parseq_input_size,
			use_amp=args.parseq_use_amp,
			language=args.parseq_lang,
			model_id=args.parseq_model_id,
			local_files_only=args.parseq_local_files_only,
		)

	if args.model == "calamari":
		from modele.calamari_wrapper import CalamariWrapper

		return CalamariWrapper(
			model=args.calamari_model,
			batch_size=args.calamari_batch_size,
			cache_dir=args.calamari_cache_dir,
			local_files_only=args.calamari_local_files_only,
			checkpoints=args.calamari_checkpoints,
			device=args.calamari_device,
		)

	raise ValueError(f"Nieobslugiwany model: {args.model}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark HTR")
	parser.add_argument(
		"--model",
		type=str,
		choices=["tesseract_pol", "rysocr", "trocr", "paddleocr", "easyocr", "parseq", "calamari", "surya"],
		default="tesseract_pol",
		help="Model OCR do uruchomienia",
	)
	parser.add_argument(
		"--inference-mode",
		type=str,
		choices=["local", "http"],
		default="local",
		help="Tryb inferencji: lokalny wrapper lub zdalny serwis HTTP.",
	)
	parser.add_argument(
		"--http-base-url",
		type=str,
		default=None,
		help="Opcjonalny URL serwisu HTTP dla wybranego modelu (np. http://localhost:8002).",
	)
	parser.add_argument(
		"--http-timeout",
		type=float,
		default=60.0,
		help="Timeout (sekundy) dla wywolan HTTP do serwisu modelu.",
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
		"--rysocr-cache-dir",
		type=str,
		default="modele/cache/rysocr",
		help="Katalog cache (Hugging Face) dla RysOCR.",
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
		"--trocr-cache-dir",
		type=str,
		default="modele/cache/trocr",
		help="Katalog cache (Hugging Face) dla TrOCR.",
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
	parser.add_argument(
		"--paddleocr-cache-dir",
		type=str,
		default="modele/cache/paddlex",
		help="Katalog cache modeli PaddleX/PaddleOCR.",
	)
	parser.add_argument(
		"--easyocr-langs",
		type=str,
		default="pl,en",
		help="Lista jezykow EasyOCR rozdzielona przecinkami (np. pl,en).",
	)
	parser.add_argument(
		"--easyocr-device",
		type=str,
		choices=["auto", "cpu", "cuda"],
		default="auto",
		help="Urzadzenie EasyOCR: auto, cpu lub cuda.",
	)
	parser.add_argument(
		"--easyocr-batch-size",
		type=int,
		default=8,
		help="Batch size inferencji EasyOCR (domyslnie: 8).",
	)
	parser.add_argument(
		"--easyocr-model-storage-dir",
		type=str,
		default="modele/cache/easyocr",
		help="Katalog na lokalny cache wag EasyOCR.",
	)
	parser.add_argument(
		"--parseq-device",
		type=str,
		choices=["auto", "cpu", "cuda"],
		default="auto",
		help="Urzadzenie PARSeq/docTR: auto, cpu lub cuda.",
	)
	parser.add_argument(
		"--parseq-batch-size",
		type=int,
		default=8,
		help="Batch size inferencji PARSeq (domyslnie: 8).",
	)
	parser.add_argument(
		"--parseq-cache-dir",
		type=str,
		default="modele/cache/parseq",
		help="Katalog na lokalny cache wag PARSeq/docTR.",
	)
	parser.add_argument(
		"--parseq-input-size",
		type=str,
		choices=["32x128", "128x128"],
		default="32x128",
		help="Preset resize preprocessingu PARSeq: 32x128 (domyslny) albo 128x128.",
	)
	parser.add_argument(
		"--parseq-use-amp",
		action="store_true",
		help="Wlacz mixed precision (AMP) dla PARSeq na CUDA.",
	)
	parser.add_argument(
		"--parseq-lang",
		type=str,
		default="pl",
		help="Preferowany jezyk dla PARSeq (informacyjnie; pretrained PARSeq nie ma twardego przelacznika jezyka).",
	)
	parser.add_argument(
		"--parseq-model-id",
		type=str,
		default=None,
		help="Opcjonalny repo_id Hugging Face dla checkpointu PARSeq/docTR (gdy dostepny).",
	)
	parser.add_argument(
		"--parseq-local-files-only",
		action="store_true",
		help="Tryb offline dla PARSeq: korzystaj tylko z lokalnego cache.",
	)
	parser.add_argument(
		"--calamari-model",
		type=str,
		default="idiotikon",
		help=(
			"Nazwa modelu Calamari z oficjalnego release (np. idiotikon, uw3-modern-english). "
			"Domyslnie idiotikon (szeroki zestaw diakrytykow)."
		),
	)
	parser.add_argument(
		"--calamari-batch-size",
		type=int,
		default=8,
		help="Batch size inferencji Calamari (domyslnie: 8).",
	)
	parser.add_argument(
		"--calamari-cache-dir",
		type=str,
		default="modele/cache/calamari",
		help="Katalog na lokalny cache modeli Calamari.",
	)
	parser.add_argument(
		"--calamari-local-files-only",
		action="store_true",
		help="Tryb offline Calamari: korzystaj tylko z lokalnego cache modeli.",
	)
	parser.add_argument(
		"--calamari-checkpoints",
		type=str,
		default=None,
		help=(
			"Opcjonalna lista sciezek do checkpointow .ckpt rozdzielona przecinkami. "
			"Gdy podane, nadpisuja --calamari-model."
		),
	)
	parser.add_argument(
		"--calamari-device",
		type=str,
		choices=["auto", "cpu", "gpu"],
		default="auto",
		help="Preferowane urzadzenie Calamari (informacyjnie; zalezy od backendu TensorFlow).",
	)
	parser.add_argument(
		"--surya-device",
		type=str,
		choices=["cpu", "cuda", "auto"],
		default="cpu",
		help="Urzadzenie dla Surya OCR (v1 line-only, HTTP-only).",
	)
	parser.add_argument(
		"--surya-batch-size",
		type=int,
		default=32,
		help="Batch size inferencji recognition dla Surya OCR.",
	)
	parser.add_argument(
		"--surya-task-name",
		type=str,
		choices=["ocr_with_boxes", "ocr_without_boxes", "block_without_boxes"],
		default="ocr_without_boxes",
		help="Task recognition Surya dla gotowych wycinkow linii.",
	)
	parser.add_argument(
		"--surya-disable-math",
		action="store_true",
		help="Wylacz rozpoznawanie matematyki w Surya (zalecane dla linii tekstowych).",
	)
	parser.add_argument(
		"--surya-cache-dir",
		type=str,
		default="modele/cache/surya",
		help="Katalog cache modeli Surya OCR.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model = build_model(args)
	runner = BenchmarkRunner(model=model)
	runner.run(
		labels_csv=args.labels_csv,
		images_dir=args.images_dir,
		output_dir=args.output_dir,
		limit=args.limit,
	)


if __name__ == "__main__":
	main()
