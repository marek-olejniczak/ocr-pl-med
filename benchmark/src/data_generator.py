from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List
import unicodedata

import pandas as pd


@dataclass(frozen=True)
class HTRSample:
	file_name: str
	image_path: Path
	ground_truth: str
	document_id: str
	line_id: str


def _extract_ids(file_name: str) -> tuple[str, str]:
	stem = Path(file_name).stem
	match = re.match(r"^(?P<doc>.+)_(?P<line>\d+)$", stem)
	if match:
		return match.group("doc"), match.group("line")
	return stem, "0"


def _build_image_index(images_dir: Path) -> dict[str, Path]:
	index: dict[str, Path] = {}
	for pattern in ("*.jpg", "*.jpeg", "*.png"):
		for path in images_dir.glob(f"**/{pattern}"):
			name = path.name
			keys = {
				name,
				unicodedata.normalize("NFC", name),
				unicodedata.normalize("NFD", name),
			}
			for key in keys:
				index.setdefault(key, path)
	return index


def load_htr_samples(
	labels_csv_path: str,
	images_dir_path: str,
	limit: int | None = None,
) -> List[HTRSample]:
	"""Laduje probki HTR z CSV i mapuje je na fizyczne pliki obrazow."""
	labels_path = Path(labels_csv_path)
	images_dir = Path(images_dir_path)

	if not labels_path.exists():
		raise FileNotFoundError(f"Nie znaleziono pliku etykiet: {labels_csv_path}")
	if not images_dir.exists():
		raise FileNotFoundError(f"Nie znaleziono katalogu obrazow: {images_dir_path}")

	df = pd.read_csv(labels_path)
	if "file_name" not in df.columns or "label" not in df.columns:
		raise ValueError("CSV musi zawierac kolumny: file_name, label")

	image_index = _build_image_index(images_dir)

	samples: List[HTRSample] = []
	for row in df.itertuples(index=False):
		file_name = str(row.file_name)
		ground_truth = str(row.label)
		image_path = image_index.get(file_name)
		if image_path is None:
			image_path = image_index.get(unicodedata.normalize("NFC", file_name))
		if image_path is None:
			image_path = image_index.get(unicodedata.normalize("NFD", file_name))
		if image_path is None or not image_path.exists():
			continue

		document_id, line_id = _extract_ids(file_name)
		samples.append(
			HTRSample(
				file_name=file_name,
				image_path=image_path,
				ground_truth=ground_truth,
				document_id=document_id,
				line_id=line_id,
			)
		)

	if limit is not None and limit > 0:
		samples = samples[:limit]

	if not samples:
		raise ValueError("Nie zaladowano zadnych probek. Sprawdz sciezki i format plikow.")

	return samples


def samples_to_dataframe(samples: Iterable[HTRSample]) -> pd.DataFrame:
	return pd.DataFrame(
		{
			"file_name": [sample.file_name for sample in samples],
			"image_path": [str(sample.image_path) for sample in samples],
			"ground_truth": [sample.ground_truth for sample in samples],
			"document_id": [sample.document_id for sample in samples],
			"line_id": [sample.line_id for sample in samples],
		}
	)
