import random
from math import comb

def _agg(ea, eb, n, groups):
    d = {}
    for a, b, m, g in zip(ea, eb, n, groups):
        x = d.setdefault(g, [0, 0, 0]); x[0] += a; x[1] += b; x[2] += m
    return list(d.values())

def paired_bootstrap(ea, eb, n, groups, B=10000, seed=0):
    """A=baseline, B=po korekcie. Resampling CALYCH grup (zeszyt/autor), a nie linii.
    Zwraca (CER_B-CER_A, 95% CI percentylowy, P(delta*>=0))."""
    U = _agg(ea, eb, n, groups); k = len(U); rnd = random.Random(seed)
    obs = (sum(u[1] for u in U) - sum(u[0] for u in U)) / sum(u[2] for u in U)
    ds = []
    for _ in range(B):
        s = [U[rnd.randrange(k)] for _ in range(k)]
        ds.append((sum(u[1] for u in s) - sum(u[0] for u in s)) / sum(u[2] for u in s))
    ds.sort()
    return obs, (ds[int(.025 * B)], ds[int(.975 * B)]), sum(d >= 0 for d in ds) / B

def paired_permutation(ea, eb, n, groups, B=20000, seed=0):
    """Test permutacyjny ze zamiana znaku roznicy na jednostke (grupa lub linia); dwustronny p."""
    diff = [u[1] - u[0] for u in _agg(ea, eb, n, groups)]; rnd = random.Random(seed)
    obs = abs(sum(diff)); c = 0
    for _ in range(B):
        c += abs(sum(x if rnd.random() < .5 else -x for x in diff)) >= obs
    return (c + 1) / (B + 1)

def sign_test(ea, eb):
    w = sum(b < a for a, b in zip(ea, eb)); l = sum(b > a for a, b in zip(ea, eb)); m = w + l
    return w, l, min(1.0, 2 * sum(comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m)

if __name__ == "__main__":
    rnd = random.Random(1); L, G = 2160, 6
    groups = [rnd.randrange(G) for _ in range(L)]; n = [rnd.randint(20, 60) for _ in range(L)]
    sh = [rnd.gauss(0, .03) for _ in range(G)]
    ea = [min(m, max(0, round(rnd.gauss(m * (.128 + sh[g]), (m * .13) ** .5)))) for m, g in zip(n, groups)]
    eb = [a - sum(rnd.random() < .06 + (g % 2) * .04 for _ in range(a)) for a, g in zip(ea, groups)]
    print("CER_A %.4f CER_B %.4f" % (sum(ea) / sum(n), sum(eb) / sum(n)))
    print("bootstrap po zeszytach:", paired_bootstrap(ea, eb, n, groups, B=5000))
    print("bootstrap po liniach  :", paired_bootstrap(ea, eb, n, list(range(L)), B=1000))
    print("permutacja linie / zeszyty:", paired_permutation(ea, eb, n, list(range(L)), B=5000), paired_permutation(ea, eb, n, groups))
    print("sign test linie (wygrane, przegrane, p):", sign_test(ea, eb))
