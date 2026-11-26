"""Lokalny serwis HTTP korekty po OCR, TYLKO tryb statyczny (nie ma obrazu linii ani Suryi). Biblioteka standardowa.

    ocr-postprocess serve --port 5056 --preload

API (JSON):
    GET  /health          -> {"status": "ok", "bundle": ..., "loaded": bool}
    POST /correct         {"lines": ["...", ...], "detailed": false}
                          -> {"lines": [...]}  albo  {"results": [{"text", "changes": [{"token", "from", "to"}]}]}

Nasłuchuje na 127.0.0.1. Korektor ładuje się przy pierwszym żądaniu (z --preload od razu).
Korygować całą stronę jednym żądaniem: wyszukiwanie kandydatów jest szybkie tylko w partii.
Błąd korekty to HTTP 500 z opisem; aplikacja powinna wtedy pokazać tekst z OCR.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ocr_postprocess import bundle as B

_bundle: str | None = None
_corr = None
_lock = threading.Lock()


def corrector():
    global _corr
    with _lock:
        if _corr is None:
            _corr = B.load(_bundle)
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
            self._send(200, {"status": "ok", "bundle": str(B.resolve(_bundle)), "loaded": _corr is not None})
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


def serve(bundle: str | None = None, port: int = 5056, preload: bool = False) -> None:
    global _bundle
    _bundle = bundle
    path = B.resolve(bundle)
    if preload:
        corrector()
    print(f"ocr-postprocess: http://127.0.0.1:{port} paczka={path}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
