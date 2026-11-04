#!/usr/bin/env python
"""Pobiera zasoby leksykalne do $PP_CACHE (domyślnie ~/.cache/ocr-pl-med/postprocessing)
i weryfikuje SHA-256. Pliki NIE trafiają do repo (licencje BSD-2 / CC-BY wymagają
zachowania noty: patrz README, sekcja Atrybucja).

Wersje są pinowane (ticket 05). sjp.pl zmienia nazwę pliku z datą wydania; jeśli stare
wydanie zniknie, trzeba zaktualizować URL i sumę tutaj (i zapisać to w raporcie).

    python fetch_resources.py            # domyślny zestaw (PoliMorf 2026, NKJP 1grams, sjp)
    python fetch_resources.py --all      # + SGJP 2026 i PoliMorf 0.6.7 (porównania)
    python fetch_resources.py --verify   # tylko sprawdź sumy
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RESOURCES = {
    "polimorf-20261018.tab.gz": (
        "https://download.sgjp.pl/morfeusz/20261018/polimorf-20261018.tab.gz",
        47_390_278, "50ca3f3b569115f31556380477ff0bccea858e30aff5a73764f747307790966c", True),
    "nkjp-1grams.gz": (
        "https://zil.ipipan.waw.pl/NKJPNGrams?action=AttachFile&do=get&target=1grams.gz",
        22_692_778, "73af49094621b2f86b594beba783c14aee8c8212f1aff3d61f281ab5d99e3dc9", True),
    "sjp-odm-20261101.zip": (
        "https://sjp.pl/sl/odmiany/sjp-odm-20261101.zip",
        12_701_206, "d1837a8b9a44107694f57c04e4564e3baa0135f86f74c00786d71902e4ae99ad", True),
    "sgjp-20261018.tab.gz": (
        "https://download.sgjp.pl/morfeusz/20261018/sgjp-20261018.tab.gz",
        42_691_269, "346bd7bc2695cb69c323de16c4c6d93afc514055fab96b458b8ead2523dae5be", False),
    "PoliMorf-0.6.7.tab.gz": (
        "https://zil.ipipan.waw.pl/PoliMorf?action=AttachFile&do=get&target=PoliMorf-0.6.7.tab.gz",
        37_391_897, "e9b2633bf384b7a778bdb4d17dd4c8055b1d040da9f5113dae513718854216b2", False),
}


def cache_dir() -> Path:
    return Path(os.environ.get("PP_CACHE", Path.home() / ".cache" / "ocr-pl-med" / "postprocessing"))


def resource_path(name: str) -> Path:
    if name not in RESOURCES:
        raise KeyError(name)
    p = cache_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run fetch_resources.py")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, size: int, digest: str) -> bool:
    dst = cache_dir() / name
    if dst.exists() and sha256(dst) == digest:
        print(f"ok      {name}")
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"get     {name} ({size / 1e6:.1f} MB)")
    req = urllib.request.Request(url, headers={"User-Agent": "ocr-pl-med-postprocessing/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        print(f"FAIL    {name}: sha256 {got} != {digest}", file=sys.stderr)
        return False
    tmp.replace(dst)
    print(f"ok      {name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="także zasoby tylko do porównań")
    ap.add_argument("--verify", action="store_true", help="nie pobieraj, tylko sprawdź sumy")
    a = ap.parse_args()
    cache_dir().mkdir(parents=True, exist_ok=True)
    print(f"PP_CACHE={cache_dir()}")
    ok = True
    for name, (url, size, digest, default) in RESOURCES.items():
        if not (default or a.all):
            continue
        if a.verify:
            p = cache_dir() / name
            good = p.exists() and sha256(p) == digest
            print(f"{'ok' if good else 'BAD':7} {name}")
            ok &= good
        else:
            ok &= fetch(name, url, size, digest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
