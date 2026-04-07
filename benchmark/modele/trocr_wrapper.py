from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class TrOCRWrapper(HTRModelWrapper):
	"""Wrapper dla TrOCR handwritten (VisionEncoderDecoder)."""

	def __init__(
		self,
		model_id: str = "microsoft/trocr-small-handwritten",
		max_new_tokens: int = 128,
		device: str | None = None,
		local_files_only: bool = False,
		batch_size: int = 4,
		use_amp: bool = False,
		cache_dir: str = "modele/cache/trocr",
	) -> None:
		model_slug = model_id.split("/")[-1].replace("-", "_")
		super().__init__(model_name=f"TrOCR_{model_slug}")
		self.model_id = model_id
		self.max_new_tokens = max(1, int(max_new_tokens))
		self.local_files_only = local_files_only
		self.batch_size = max(1, int(batch_size))
		self.use_amp = bool(use_amp)
		self.cache_dir = str(Path(cache_dir))

		try:
			import torch
			from transformers import TrOCRProcessor, VisionEncoderDecoderModel
		except Exception as exc:
			raise RuntimeError(
				"Brakuje zaleznosci dla TrOCR. Zainstaluj: torch, torchvision, transformers, sentencepiece"
			) from exc

		self._torch = torch
		self._processor_cls = TrOCRProcessor
		self._model_cls = VisionEncoderDecoderModel
		Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

		resolved_device = device
		if resolved_device is None:
			resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = resolved_device

		if self.use_amp and self.device != "cuda":
			print("[TrOCR] --trocr-use-amp zignorowane: AMP dziala tylko na CUDA.")

		torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

		print(f"[TrOCR] Inicjalizacja modelu. model={self.model_id}, device={self.device}")
		if self.local_files_only:
			print("[TrOCR] Tryb offline: local_files_only=True (bez pobierania z Hugging Face).")

		try:
			self.processor = self._processor_cls.from_pretrained(
				self.model_id,
				local_files_only=self.local_files_only,
				cache_dir=self.cache_dir,
			)
			try:
				self.model = self._model_cls.from_pretrained(
					self.model_id,
					dtype=torch_dtype,
					local_files_only=self.local_files_only,
					cache_dir=self.cache_dir,
				)
			except TypeError:
				self.model = self._model_cls.from_pretrained(
					self.model_id,
					torch_dtype=torch_dtype,
					local_files_only=self.local_files_only,
					cache_dir=self.cache_dir,
				)
		except Exception as exc:
			if self.local_files_only:
				raise RuntimeError(
					"Nie udalo sie zaladowac TrOCR w trybie offline. "
					"Uruchom raz bez --trocr-local-files-only, aby pobrac model do cache."
				) from exc
			raise

		self.model = self.model.to(self.device).eval()
		print("[TrOCR] Model i processor gotowe do inferencji.")

	def predict(self, image_path: str) -> str:
		predictions = self.predict_batch([image_path])
		return predictions[0] if predictions else ""

	def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
		image_paths_list = list(image_paths)
		if not image_paths_list:
			return []

		predictions: list[str] = []
		amp_enabled = self.use_amp and self.device == "cuda"

		for start in range(0, len(image_paths_list), self.batch_size):
			batch_paths = image_paths_list[start:start + self.batch_size]
			images = []
			for image_path in batch_paths:
				path = Path(image_path)
				if not path.exists():
					raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
				with Image.open(path) as image:
					images.append(image.convert("RGB"))

			inputs = self.processor(images=images, return_tensors="pt", padding=True)
			pixel_values = inputs["pixel_values"].to(self.device)

			autocast_ctx = (
				self._torch.autocast(device_type="cuda", dtype=self._torch.float16, enabled=amp_enabled)
				if self.device == "cuda"
				else nullcontext()
			)

			with self._torch.no_grad():
				with autocast_ctx:
					generated_ids = self.model.generate(
						pixel_values=pixel_values,
						max_new_tokens=self.max_new_tokens,
					)

			batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
			predictions.extend(text.strip() for text in batch_predictions)

		return predictions