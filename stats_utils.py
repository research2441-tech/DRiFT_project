"""Statistics used in the paper: tail metrics, paired tests, multiplicity correction, effect sizes,
and a two-level (seed x environment) bootstrap."""
import numpy as np
from scipy import stats


def cvar(x, alpha=0.1):
    x = np.sort(np.asarray(x))
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(np.mean(x[-k:]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for x - y (negative => x smaller)."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((np.sum(r[d > 0]) - np.sum(r[d < 0])) / np.sum(r))


def paired_tests(per_env, ref="DRiFT"):
    """per_env: dict method -> (n_env,) seed-averaged coherent infidelity on common environments."""
    names = [m for m in per_env if m != ref]
    fr = stats.friedmanchisquare(*[per_env[m] for m in per_env])
    raw, eff = [], []
    for m in names:
        w = stats.wilcoxon(per_env[ref], per_env[m], alternative="two-sided")
        raw.append(w.pvalue)
        eff.append(rank_biserial(per_env[ref], per_env[m]))
    return {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
            "pairs": {m: {"p_raw": float(p), "p_holm": float(ph), "rank_biserial": e}
                      for m, p, ph, e in zip(names, raw, holm(raw), eff)}}


def bootstrap_ci(M, stat=np.mean, n=4000, seed=0, level=0.95):
    """M: (n_seed, n_env). Resample seeds and environments independently."""
    rng = np.random.default_rng(seed)
    M = np.asarray(M)
    s, e = M.shape
    vals = np.empty(n)
    for b in range(n):
        rs = rng.integers(0, s, s)
        re = rng.integers(0, e, e)
        vals[b] = stat(M[np.ix_(rs, re)])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
