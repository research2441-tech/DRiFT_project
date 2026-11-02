"""Noise spectra, spectral ambiguity sets and the closed-form worst case (spectral knapsack).

A noise environment for one source is a list of components:
    ('ou',  rate r, variance v)   Ornstein-Uhlenbeck,  S(w) = 2 v r / (r^2 + w^2)
    ('rtn', rate lam, amplitude b) random telegraph,    S(w) = 2 b^2 (2 lam) / ((2 lam)^2 + w^2)
    ('qs',  variance v)            quasi-static Gaussian offset (DC)
PSDs are two-sided; variance = int S dw / 2pi.
"""
import numpy as np
import jax.numpy as jnp
from jax import lax
import config as C


# ------------------------------------------------------------------ grid
def frequency_grid(w_lo=C.W_LO, w_hi=C.W_HI, n=C.N_W):
    """Return (w, weights) with w[0] = 0 the DC bin (weight 1, carries variance below w_lo)."""
    wg = np.geomspace(w_lo, w_hi, n)
    tw = np.zeros(n)
    tw[1:-1] = 0.5 * (wg[2:] - wg[:-2])
    tw[0] = 0.5 * (wg[1] - wg[0])
    tw[-1] = 0.5 * (wg[-1] - wg[-2])
    w = np.concatenate([[0.0], wg])
    weights = np.concatenate([[1.0], tw / np.pi])     # symmetric two-sided integral over w >= 0
    return w, weights


# ------------------------------------------------------------------ components
def one_over_f(rms, exponent=1.0, rate_range=C.OU_RATE_RANGE, n=C.N_OU):
    rates = np.geomspace(*rate_range, n)
    v = rates ** (exponent - 1.0)
    v = v / v.sum() * rms ** 2
    return [("ou", r, vi) for r, vi in zip(rates, v)]


def lorentz(r, v, w):
    return 2.0 * v * r / (r ** 2 + w ** 2)


def dc_part(r, v, w_lo):
    """Variance of a Lorentzian (corner r, variance v) contained in |w| < w_lo."""
    return v * (2.0 / np.pi) * np.arctan(w_lo / r)


def on_grid(comps, w, w_lo=C.W_LO):
    """Spectrum vector on the grid: entry 0 is the DC variance, entries 1.. are S(w_m)."""
    S = np.zeros_like(w)
    for c in comps:
        if c[0] == "qs":
            S[0] += c[1]
            continue
        if c[0] == "ou":
            r, v = c[1], c[2]
        else:
            r, v = 2.0 * c[1], c[2] ** 2
        S[1:] += lorentz(r, v, w[1:])
        S[0] += dc_part(r, v, w_lo)
    return S


def variance(comps):
    tot = 0.0
    for c in comps:
        tot += c[2] ** 2 if c[0] == "rtn" else (c[1] if c[0] == "qs" else c[2])
    return tot


# ------------------------------------------------------------------ nominal environments
def _freq_env(rms_1f, rms_qs, tls, exponent=1.0):
    comps = one_over_f(rms_1f, exponent) + [("qs", rms_qs ** 2)]
    comps += [("rtn", lam, b) for lam, b in tls]
    return comps


def _amp_env(rms_1f, rms_qs, exponent=1.0):
    return one_over_f(rms_1f, exponent) + [("qs", rms_qs ** 2)]


def nominal_env(system):
    env = {}
    for s in system.sources:
        nm = s["name"]
        if nm.startswith("freq"):
            tls = [(C.TLS_RATE_NOM, C.TLS_B)] if nm in ("freq", "freq1") else []
            env[nm] = _freq_env(C.FREQ_1F_RMS, C.FREQ_QS_RMS, tls)
        elif nm == "amp":
            env[nm] = _amp_env(C.AMP_1F_RMS, C.AMP_QS_RMS)
        elif nm == "coupler":
            env[nm] = _amp_env(2 * C.AMP_1F_RMS, C.AMP_QS_RMS)
    return env


def sample_env(system, rng, mode="in"):
    """Draw an environment from the drift prior. mode = 'in' (in-distribution) or 'ood'."""
    k = C.KAPPA if mode == "in" else 2 * C.KAPPA
    ex_lo, ex_hi = (0.8, 1.2) if mode == "in" else (0.6, 1.4)
    b_hi = 1.0 if mode == "in" else 1.6
    rate_hi = C.TLS_RATE_RANGE[1] if mode == "in" else 3.0
    ntls_p = [0.25, 0.5, 0.25, 0.0] if mode == "in" else [0.1, 0.4, 0.3, 0.2]
    lu = lambda: np.exp(rng.uniform(-np.log(k), np.log(k)))
    env = {}
    for s in system.sources:
        nm = s["name"]
        ex = rng.uniform(ex_lo, ex_hi)
        if nm.startswith("freq"):
            ntls = rng.choice(4, p=ntls_p)
            tls = [(np.exp(rng.uniform(np.log(C.TLS_RATE_RANGE[0]), np.log(rate_hi))),
                    rng.uniform(0.5, b_hi) * C.TLS_B) for _ in range(ntls)]
            env[nm] = _freq_env(C.FREQ_1F_RMS * lu(), C.FREQ_QS_RMS * lu(), tls, ex)
        elif nm == "amp":
            env[nm] = _amp_env(C.AMP_1F_RMS * lu(), C.AMP_QS_RMS * lu(), ex)
        elif nm == "coupler":
            env[nm] = _amp_env(2 * C.AMP_1F_RMS * lu(), C.AMP_QS_RMS * lu(), ex)
    return env


def scale_env(env, factor):
    """Scale every amplitude by `factor` (variance by factor^2)."""
    out = {}
    for nm, comps in env.items():
        new = []
        for c in comps:
            if c[0] == "qs":
                new.append(("qs", c[1] * factor ** 2))
            elif c[0] == "ou":
                new.append(("ou", c[1], c[2] * factor ** 2))
            else:
                new.append(("rtn", c[1], c[2] * factor))
        out[nm] = new
    return out


# ------------------------------------------------------------------ ambiguity set
def ambiguity_set(system, w, weights, kappa=C.KAPPA, slack=C.BUDGET_SLACK, tls_envelope=True):
    """Box-and-budget spectral ambiguity set built from the nominal (measured) spectrum.

    lo(w) = S_nom,bg / kappa;  hi(w) = kappa S_nom,bg + E_TLS(w);  sum_m weights_m S_m <= budget,
    where S_nom,bg excludes telegraph components (a TLS may vanish or move) and E_TLS is the upper
    envelope of all telegraph spectra with amplitude TLS_B and rates in TLS_RATE_RANGE.
    """
    env = nominal_env(system)
    sets = {}
    for s in system.sources:
        nm = s["name"]
        bg = [c for c in env[nm] if c[0] != "rtn"]
        Sbg = on_grid(bg, w)
        lo = Sbg / kappa
        hi = kappa * Sbg
        meas_var = variance(env[nm])
        if nm.startswith("freq") and tls_envelope:
            b = C.TLS_B
            rmin, rmax = 2 * C.TLS_RATE_RANGE[0], 2 * C.TLS_RATE_RANGE[1]
            wpos = w[1:]
            r_star = np.clip(wpos, rmin, rmax)
            hi[1:] += lorentz(r_star, b ** 2, wpos)
            hi[0] += dc_part(rmin, b ** 2, C.W_LO)
            meas_var = max(meas_var, variance(bg) + b ** 2)
        budget = slack * kappa * meas_var
        budget = max(budget, float(np.dot(weights, lo)) * 1.0001)
        sets[nm] = dict(lo=lo, hi=hi, budget=budget)
    return sets


def in_set(S, aset, weights, tol=1e-9):
    return bool(np.all(S >= aset["lo"] * (1 - 1e-6)) and np.all(S <= aset["hi"] * (1 + 1e-6))
                and np.dot(weights, S) <= aset["budget"] * (1 + tol))


# ------------------------------------------------------------------ worst case (knapsack)
def knapsack_np(phi, weights, lo, hi, budget):
    """Exact maximiser of sum_m weights_m S_m phi_m over the box-and-budget set (numpy)."""
    base = np.dot(weights, lo * phi)
    cap = budget - np.dot(weights, lo)
    mass = weights * (hi - lo)
    order = np.argsort(-phi)
    S = lo.copy()
    for m in order:
        if cap <= 0:
            break
        take = min(1.0, cap / mass[m]) if mass[m] > 0 else 0.0
        S[m] += take * (hi[m] - lo[m])
        cap -= take * mass[m]
    return float(np.dot(weights, S * phi)), S


def knapsack_jax(phi, weights, lo, hi, budget):
    """Differentiable worst case: allocation is held fixed (Danskin), gradient flows through phi."""
    phi_ng = lax.stop_gradient(phi)
    mass = weights * (hi - lo)
    cap = budget - jnp.dot(weights, lo)
    order = jnp.argsort(-phi_ng)
    m_sorted = mass[order]
    before = jnp.cumsum(m_sorted) - m_sorted
    frac_sorted = jnp.clip((cap - before) / jnp.where(m_sorted > 0, m_sorted, 1.0), 0.0, 1.0)
    frac = jnp.zeros_like(frac_sorted).at[order].set(frac_sorted)
    S_star = lax.stop_gradient(lo + frac * (hi - lo))
    return jnp.dot(weights, S_star * phi), S_star


# ------------------------------------------------------------------ time-domain synthesis
def synthesize(comps, ntraj, nsteps, dt, rng):
    """Sample noise trajectories (ntraj x nsteps) on a uniform grid with spacing dt."""
    x = np.zeros((ntraj, nsteps))
    for c in comps:
        if c[0] == "qs":
            x += np.sqrt(c[1]) * rng.standard_normal((ntraj, 1))
        elif c[0] == "ou":
            r, v = c[1], c[2]
            a = np.exp(-r * dt)
            sig = np.sqrt(v * (1 - a ** 2))
            y = np.sqrt(v) * rng.standard_normal(ntraj)
            xi = rng.standard_normal((ntraj, nsteps))
            for k in range(nsteps):
                x[:, k] += y
                y = a * y + sig * xi[:, k]
        else:
            lam, b = c[1], c[2]
            pflip = 0.5 * (1 - np.exp(-2 * lam * dt))
            st = np.where(rng.random(ntraj) < 0.5, -1.0, 1.0)
            u = rng.random((ntraj, nsteps))
            for k in range(nsteps):
                x[:, k] += b * st
                st = np.where(u[:, k] < pflip, -st, st)
    return x
