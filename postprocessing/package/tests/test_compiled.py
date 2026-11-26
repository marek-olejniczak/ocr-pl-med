"""Skompilowany indeks daje te same odpowiedzi co Lexicon/UnigramLM (na małym słowniku)."""
import math
from types import SimpleNamespace

from ocr_postprocess.candidates import Prefilter
from ocr_postprocess.compiled import CompiledLexicon, CompiledLM, write_index
from ocr_postprocess.lexicon import fold


class TinyLex:
    common = {"kość", "kosz", "kosa", "żyła", "zyła", "noga", "nogi", "mięsień"}
    proper_only = {"Kraków".lower()}
    abbrev = {"wewn", "m"}
    fragments = {"przed"}
    sjp = frozenset()
    extra_known = {"musculus"}
    proper_target = set()

    def targets(self):
        yield from self.common

    def known(self, w):
        return w in self.common | self.proper_only | self.abbrev | self.fragments | self.extra_known

    def by_length(self):
        d = {}
        for w in self.targets():
            d.setdefault(len(w), []).append(w)
        return d

    def by_fold(self):
        d = {}
        for w in self.targets():
            d.setdefault(fold(w), []).append(w)
        return d


def test_compiled_matches_source(tmp_path):
    lex = TinyLex()
    counts = {"kość": 50, "kosz": 3, "noga": 7}
    rstar = {3: 2.5, 7: 6.1, 50: 49.0}
    lm = SimpleNamespace(count=lambda w: counts.get(w, 0), rstar=rstar, unseen=0.1, logz=math.log(200.0))
    write_index(tmp_path, lex, lm, ["pl"])
    cl = CompiledLexicon(tmp_path)
    clm = CompiledLM(tmp_path, cl)
    for w in ["kość", "kosz", "kraków", "wewn", "przed", "musculus", "xyz", "nog", "mięsień"]:
        assert cl.known(w) == lex.known(w), w
        assert cl.target(w) == (w in lex.common), w
    for w in lex.common:
        c = counts.get(w, 0)
        assert clm.count(w) == c
        assert math.isclose(clm.logp(w), math.log(rstar.get(c, c) if c else 0.1) - math.log(200.0))
    assert cl.abbrev == lex.abbrev
    assert sorted(cl.fold_group("zyla")) == ["zyła", "żyła"]
    assert {k: sorted(v) for k, v in cl.by_length().items()} == {k: sorted(v) for k, v in lex.by_length().items()}
    keys = ["kosc", "zyla", "nogq"]
    assert Prefilter(cl, d_max=1).batch(keys) == Prefilter(lex, d_max=1).batch(keys)
