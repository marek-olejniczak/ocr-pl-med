"""Build resources/anatomy_phrases.txt with a local LLM (Ollama).

The OCR test material is student anatomy notes. The hand-written seed list
(anatomy_seed.txt) covers the nouns; this script asks a local model for the
kind of short phrases a student actually writes in a notebook — "przyczep
początkowy na guzku większym", "unerwienie: n. udowy", "zgina i odwraca stopę"
— and cleans the result into one phrase per line.

Nothing from the real test set is shown to the model; the pool is a stylistic
imitation, not a copy. Output is deterministic per run only up to the model's
own sampling.

Usage:
    python src/build_anatomy_pool.py --batches 60 --model gemma4:e4b
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

TOPICS = [
    "kości kończyny dolnej", "kości kończyny górnej", "kręgosłup i klatka piersiowa",
    "czaszka", "stawy kończyny dolnej i ich więzadła", "stawy kończyny górnej",
    "mięśnie uda i podudzia", "mięśnie obręczy barkowej i ramienia", "mięśnie grzbietu",
    "mięśnie brzucha i przepona", "mięśnie głowy i szyi", "unaczynienie kończyny dolnej",
    "unaczynienie miednicy", "unaczynienie kończyny górnej", "tętnice głowy i szyi",
    "układ żylny", "układ chłonny", "splot lędźwiowy i krzyżowy", "splot ramienny",
    "nerwy czaszkowe", "serce: budowa, zastawki, unaczynienie", "układ oddechowy",
    "układ pokarmowy", "wątroba, trzustka, śledziona", "układ moczowy",
    "układ płciowy męski i żeński", "gruczoły dokrewne", "mózgowie i opony",
    "rdzeń kręgowy i drogi nerwowe", "narząd słuchu i równowagi", "narząd wzroku",
    "skóra i narządy zmysłów", "histologia tkanek", "topografia dołu podkolanowego i trójkąta udowego",
    "kanał pachwinowy i przepukliny", "śródpiersie", "otrzewna i jama brzuszna",
    "ruchy w stawach i mięśnie odpowiedzialne", "przyczepy mięśni", "otwory czaszki i co przez nie przechodzi",
]

PROMPT = """Jesteś studentem medycyny i robisz odręczne notatki z anatomii po polsku na temat: {topic}.
Wypisz {n} krótkich linijek notatek, każda w osobnym wierszu, bez numeracji i bez myślników na początku.
Mieszaj formy: pojedyncze terminy ("kość łódkowata"), krótkie hasła z dwukropkiem ("unerwienie: n. udowy"),
skróty ("m. czworogłowy uda", "t. udowa", "n. kulszowy", "w. krzyżowe"), krótkie zdania o czynności
("zgina udo w stawie biodrowym"), przyczepy ("pp: kresa chropawa", "pk: guzowatość piszczeli"),
położenie ("leży bocznie od t. udowej"). Od 1 do 8 słów w linijce. Tylko polskie litery, bez angielskiego,
bez łaciny w nawiasach, bez cudzysłowów, bez markdown."""

# Anything outside this set marks a line as garbage (mojibake, Cyrillic, emoji).
_ALLOWED = re.compile(r"^[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9 .,:;()/\-–+%°]+$")


def ask(model: str, prompt: str, host: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.9, "num_predict": 900}}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


def clean(raw: str) -> list[str]:
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
        line = line.strip('"„”\'`*_ ')
        line = " ".join(line.split())
        if not (3 <= len(line) <= 70):
            continue
        if not _ALLOWED.match(line):
            continue
        if len(line.split()) > 8:
            continue
        if line.isupper() and len(line) > 12:   # model shouting a heading
            line = line.lower()
        out.append(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--per-batch", type=int, default=40)
    ap.add_argument("--output", default="resources/anatomy_phrases.txt")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    out_path = Path(args.output)
    phrases: set[str] = set()
    if out_path.exists():
        phrases.update(l.strip() for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip())
        print(f"resuming with {len(phrases)} phrases")

    for i in range(args.batches):
        topic = TOPICS[i % len(TOPICS)]
        try:
            raw = ask(args.model, PROMPT.format(topic=topic, n=args.per_batch), args.host)
        except Exception as exc:
            print(f"  batch {i} failed: {exc}", file=sys.stderr)
            continue
        new = [p for p in clean(raw) if p not in phrases]
        phrases.update(new)
        print(f"  [{i + 1}/{args.batches}] {topic}: +{len(new)} (total {len(phrases)})")
        out_path.write_text("\n".join(sorted(phrases)) + "\n", encoding="utf-8")

    print(f"Done: {len(phrases)} phrases -> {out_path}")


if __name__ == "__main__":
    main()
