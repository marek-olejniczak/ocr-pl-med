from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from docker.logging_utils import emit_event


class PredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class LoadRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


def decode_base64_image_to_temp_file(
    image_base64: str,
    suffix: str = ".png",
    *,
    logger: logging.Logger | None = None,
    request_id: str | None = None,
) -> str:
    if logger is not None:
        emit_event(
            logger,
            logging.INFO,
            "decode_base64_started",
            request_id=request_id,
            image_base64_len=len(image_base64),
            suffix=suffix,
        )

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        if logger is not None:
            emit_event(
                logger,
                logging.ERROR,
                "decode_base64_failed",
                request_id=request_id,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
        raise ValueError("Niepoprawny image_base64 w request.") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        temp_path = str(Path(tmp.name))

    if logger is not None:
        emit_event(
            logger,
            logging.INFO,
            "decode_base64_succeeded",
            request_id=request_id,
            bytes_len=len(image_bytes),
            temp_path=temp_path,
        )

    return temp_path


def detect_cache_presence(cache_dir: str | None) -> bool | None:
    if not cache_dir:
        return None

    path = Path(cache_dir)
    if not path.exists():
        return False
    if path.is_file():
        return True
    if path.is_dir():
        try:
            return any(path.iterdir())
        except OSError:
            return False
    return False
