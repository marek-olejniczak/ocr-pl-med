from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


def decode_base64_image_to_temp_file(image_base64: str, suffix: str = ".png") -> str:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Niepoprawny image_base64 w request.") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        return str(Path(tmp.name))
