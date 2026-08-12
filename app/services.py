"""OCR service registry (name -> base URL) read from the benchmark
compose file."""

from pathlib import Path

import requests
import yaml

COMPOSE_PATH = (Path(__file__).resolve().parent.parent
                / "benchmark" / "docker-compose.yml")


def load_registry(compose_path=COMPOSE_PATH):
    with open(compose_path) as f:
        compose = yaml.safe_load(f)
    registry = {}
    for name, svc in (compose.get("services") or {}).items():
        ports = svc.get("ports") or []
        if ports:
            host_port = str(ports[0]).split(":")[0]
            registry[name] = f"http://localhost:{host_port}"
    return registry


def probe(url, timeout=2):
    try:
        return requests.get(f"{url}/health", timeout=timeout).ok
    except requests.RequestException:
        return False
