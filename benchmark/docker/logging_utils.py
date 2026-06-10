from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    _CONFIGURED = True


def get_service_logger(service_name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(service_name)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def timer_start() -> float:
    return time.perf_counter()


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 2)


def _normalize_field(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "[" + ",".join(_normalize_field(item) for item in value) + "]"

    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys(), key=lambda candidate: str(candidate)):
            items.append(f"{key}:{_normalize_field(value[key])}")
        return "{" + ",".join(items) + "}"

    return str(value).replace("\n", " ").strip()


def emit_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    request_id: str | None = None,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    parts = [f"event={event}"]

    if request_id:
        parts.append(f"request_id={request_id}")

    for key in sorted(fields.keys()):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{key}={_normalize_field(value)}")

    logger.log(level, " ".join(parts), exc_info=exc_info)