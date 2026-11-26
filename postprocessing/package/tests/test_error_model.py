import gzip
import json
import math

from rapidfuzz.distance import Levenshtein

from ocr_postprocess.error_model import ErrorModel, Params, capped
from collections import Counter


def toy_pairs():
    base = ["czas", "oczy", "beczka", "kość", "most", "rana", "noga", "ręka", "kot", "las"]
    out = []
    for g in ("a", "b", "c", "d"):
        for w in base:
            out.append((w, w, g))
        out += [("czas", "was", g), ("oczy", "owy", g), ("beczka", "bewka", g)]  # cz -> w
        out += [("kość", "kosc", g), ("ręka", "reka", g)]                         # diakrytyki
    return out


def test_unit_cost_alignment_is_levenshtein():
    m = ErrorModel.unit_cost()
    for a, b in [("kość", "kosc"), ("czas", "was"), ("", "ab"), ("abc", "")]:
        ops = m.align(a, b)
        assert sum(s != t for s, t in ops) == Levenshtein.distance(a, b)
        assert "".join(s for s, _ in ops) == a and "".join(t for _, t in ops) == b


def test_multi_unit_learned_and_helps():
    pairs = toy_pairs()
    bm = ErrorModel.fit(pairs)
    assert ("cz", "w") in bm.lmulti
    no_multi = ErrorModel.fit(pairs, Params(multi=False))
    assert bm.logp("wy", "czy") > no_multi.logp("wy", "czy")


def test_learned_confusion_beats_uniform_and_ranks_candidates():
    pairs = toy_pairs()
    bm, uni = ErrorModel.fit(pairs), ErrorModel.uniform(pairs)
    assert bm.logp("kosc", "kość") > uni.logp("kosc", "kość")
    # diakrytyk (ś→s) tańszy niż dowolna substytucja (ś→m)
    assert bm.logp("kosc", "kość") > bm.logp("kosc", "kosm")
    # identyczność jest najbardziej prawdopodobna
    assert bm.logp("noga", "noga") > bm.logp("noga", "nogi")


def test_forward_ge_viterbi_and_probabilities_bounded():
    bm = ErrorModel.fit(toy_pairs())
    for obs, cand in [("was", "czas"), ("kosc", "kość"), ("noga", "noga")]:
        v, f = bm.logp(obs, cand), bm.logp(obs, cand, forward=True)
        assert f >= v - 1e-9
        assert f <= 0.0
    assert math.isfinite(bm.logp("xyz", "ąę"))


def test_group_share_cap():
    assert capped(Counter({"a": 10}), 0.5) == 10
    assert capped(Counter({"a": 90, "b": 10}), 0.5) == 20
    assert capped(Counter({"a": 5, "b": 5}), 0.5) == 10


def test_json_roundtrip_for_deploy_bundle(tmp_path):
    from ocr_postprocess import bundle
    bm = ErrorModel.fit(toy_pairs())
    bundle.save(tmp_path, bm, {})
    bm2 = ErrorModel.from_dict(json.loads(gzip.decompress((tmp_path / "error_model.json.gz").read_bytes())))
    for obs, cand in (("kosc", "kość"), ("zyla", "żyła"), ("abc", "abd"), ("was", "czas")):
        assert bm2.logp(obs, cand) == bm.logp(obs, cand)
        assert bm2.align(cand, obs) == bm.align(cand, obs)
