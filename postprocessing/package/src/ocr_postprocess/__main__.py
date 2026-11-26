"""Polecenia pakietu (`ocr-postprocess ...` albo `python -m ocr_postprocess ...`):

    setup    pobierz słowniki (SHA-256), skompiluj indeks i sprawdź, że korektor działa; raz po instalacji
    check    sprawdź zainstalowany korektor i zmierz czas startu
    check-surya  test zgodności trybu z oceną Suryi na tej maszynie (wymaga extras [surya])
    correct  popraw linie ze stdin (tryb statyczny)
    serve    serwis HTTP (tryb statyczny), patrz server.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ocr_postprocess import bundle as B

SMOKE_IN, SMOKE_OUT = "WZGORZE - impulsy z", "WZGÓRZE - impulsy z"


def check(name: str | None) -> int:
    t0 = time.time()
    corr = B.load(name)
    t1 = time.time()
    got = corr.correct(SMOKE_IN)
    print(f"paczka: {B.resolve(name)}")
    print(f"start {t1 - t0:.1f} s, korekta linii kontrolnej {time.time() - t1:.1f} s")
    if got != SMOKE_OUT:
        print(f"BŁĄD: {SMOKE_IN!r} -> {got!r}, oczekiwano {SMOKE_OUT!r}", file=sys.stderr)
        return 1
    print("tryb statyczny: OK" + ("; paczka ma progi do oceny przez Suryę" if corr.p_rescored else ""))
    return 0


def setup(name: str | None, all_resources: bool) -> int:
    from ocr_postprocess import compiled, resources

    meta = json.loads((B.resolve(name) / "bundle.json").read_text(encoding="utf-8"))
    if not resources.fetch_all(all_resources):
        return 1
    out = compiled.default_dir(meta["domain_langs"])
    if (out / "meta.json").exists():
        print(f"indeks aktualny: {out}")
    else:
        print("kompiluję indeks słownika (ok. 40 s, ok. 250 MB)...", flush=True)
        t0 = time.time()
        compiled.build(out, meta["domain_langs"])
        print(f"indeks: {out} ({time.time() - t0:.0f} s)")
    return check(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ocr-postprocess", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "check", "correct", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", help=f"nazwa albo ścieżka paczki (domyślnie {B.DEFAULT_BUNDLE})")
        if name == "setup":
            s.add_argument("--all", action="store_true", help="także zasoby potrzebne tylko eksperymentom")
        if name == "correct":
            s.add_argument("--detailed", action="store_true", help="jsonl z listą zmian zamiast samego tekstu")
        if name == "serve":
            s.add_argument("--port", type=int, default=5056)
            s.add_argument("--preload", action="store_true", help="wczytaj korektor przed pierwszym żądaniem")
    cs = sub.add_parser("check-surya")
    cs.add_argument("--adapter", help="katalog adaptera LoRA (bez: model bazowy)")
    a = ap.parse_args(argv)
    if a.cmd == "check-surya":
        from ocr_postprocess import surya_check
        return surya_check.run(a.adapter)
    if a.cmd == "setup":
        return setup(a.bundle, a.all)
    if a.cmd == "check":
        return check(a.bundle)
    if a.cmd == "serve":
        from ocr_postprocess import server
        server.serve(a.bundle, a.port, a.preload)
        return 0
    corr = B.load(a.bundle)
    lines = [line.rstrip("\n") for line in sys.stdin]
    if a.detailed:
        for r in corr.correct_lines_detailed(lines):
            print(json.dumps(r, ensure_ascii=False))
    else:
        for line in corr.correct_lines(lines):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
