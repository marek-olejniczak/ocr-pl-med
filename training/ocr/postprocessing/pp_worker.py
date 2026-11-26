"""pp_worker.py — lokalny serwis korekty po OCR (opcjonalny krok za Suryą), tylko biblioteka standardowa + zależności modułu.

    PP_BUNDLE=deploy/v4_200k_25k PP_WORKER_PORT=5056 .venv/bin/python pp_worker.py

API (JSON):
    GET  /health          -> {"status": "ok", "bundle": ..., "loaded": bool}
    POST /correct         {"lines": ["...", ...], "detailed": false}
                          -> {"lines": [...]}  albo  {"results": [{"text", "changes": [{"token", "from", "to"}]}]}

Korektor ładuje się przy pierwszym żądaniu (ok. 1.5 s ze skompilowanym indeksem `python -m pp.compiled`,
ok. 1.6 GB RAM) albo od razu z PP_PRELOAD=1.
Korygować całą stronę jednym żądaniem: prefiltr jest szybki tylko w partii.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
BUNDLE = Path(os.environ.get("PP_BUNDLE", HERE / "deploy" / "v4_200k_25k"))
PORT = int(os.environ.get("PP_WORKER_PORT", "5056"))
_corr = None
_lock = threading.Lock()


def corrector():
    global _corr
    with _lock:
        if _corr is None:
            from pp.deploy import load
            _corr = load(BUNDLE)
    return _corr


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "bundle": str(BUNDLE), "loaded": _corr is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/correct":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            lines = req.get("lines")
            if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
                return self._send(400, {"error": "lines: lista napisów"})
            c = corrector()
            if req.get("detailed"):
                return self._send(200, {"results": c.correct_lines_detailed(lines)})
            return self._send(200, {"lines": c.correct_lines(lines)})
        except Exception as exc:  # serwis pomocniczy: błąd korekty nie może zatrzymać aplikacji
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    if os.environ.get("PP_PRELOAD") == "1":
        corrector()
    print(f"pp_worker: http://127.0.0.1:{PORT} bundle={BUNDLE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
