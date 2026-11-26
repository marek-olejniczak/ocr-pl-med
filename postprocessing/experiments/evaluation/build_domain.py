"""Lista terminów anatomicznych spoza PoliMorf (ticket 12, wersja mini): ocr_postprocess/data/ta_terms.tsv (w pakiecie).

Źródło jest zewnętrzne (Wikidata, CC0), więc nie ma przecieku z GT. Słowa trafiają tylko do
`known`: chronią angielskie i łacińskie terminy przed „poprawieniem”, nie są kandydatami korekty.

    curl -G https://query.wikidata.org/sparql -H 'Accept: text/csv' \
        --data-urlencode query@data/wikidata_ta.rq -o $PP_CACHE/domain/wikidata_ta.csv
    python -m evaluation.build_domain
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from ocr_postprocess.resources import cache_dir
from ocr_postprocess.lexicon import DOMAIN_TERMS, MIN_DOMAIN_LEN, Lexicon

WORD = re.compile(r"[^\W\d_]+")


def main(src: Path = cache_dir() / "domain" / "wikidata_ta.csv", dst: Path = DOMAIN_TERMS) -> None:
    lex = Lexicon(counts={})
    langs: dict[str, set[str]] = defaultdict(set)
    with open(src, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for w in WORD.findall(r["label"].lower()):
                if len(w) >= MIN_DOMAIN_LEN and not lex.known(w):
                    langs[w].add(r["lang"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for w in sorted(langs):
            f.write(f"{w}\t{','.join(sorted(langs[w]))}\n")
    print(f"{len(langs)} słów -> {dst}")


if __name__ == "__main__":
    main()
