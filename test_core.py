"""Correctness checks: run with `python -m pytest -q test_core.py` (or `python test_core.py`)."""
import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm
from scipy.optimize import linprog
import config as C
import core
import models
import methods as MT
import spectra as SP
import noise_sim as NS


def test_intrastep_quadrature_matches_van_loan():
    """Gauss-Legendre intra-step integral equals the exact Van Loan block-exponential integral."""
    rng = np.random.default_rng(0)
    d, dt, w = 3, 0.5, 0.8
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = 0.6 * (A + A.conj().T)
    E = np.diag([0.0, 1.0, 2.0]).astype(complex)
    M = np.zeros((2 * d, 2 * d), complex)
    M[:d, :d] = -1j * H
    M[:d, d:] = E
    M[d:, d:] = -1j * H + 1j * w * np.eye(d)
    eM = expm(M * dt)
    J_exact = eM[:d, :d].conj().T @ eM[:d, d:]
    s, wq = core.gl_nodes(dt, C.GL_NODES)
    J_gl = sum(wi * np.exp(1j * w * si) * expm(1j * H * si) @ E @ expm(-1j * H * si) for si, wi in zip(s, wq))
    assert np.max(np.abs(J_gl - J_exact)) < 1e-9


def test_knapsack_matches_linear_program():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 30
        phi, wts = rng.random(n), rng.random(n)
        lo = rng.random(n)
        hi = lo + rng.random(n)
        budget = float(wts @ lo + 0.4 * wts @ (hi - lo))
        val, S = SP.knapsack_np(phi, wts, lo, hi, budget)
        res = linprog(-(wts * phi), A_ub=[wts], b_ub=[budget], bounds=list(zip(lo, hi)))
        assert abs(val - (-res.fun)) < 1e-8
        vj, _ = SP.knapsack_jax(jnp.asarray(phi), jnp.asarray(wts), jnp.asarray(lo), jnp.asarray(hi), budget)
        assert abs(float(vj) - val) < 1e-8


def test_first_order_prediction_matches_monte_carlo_small_noise():
    """With weak Gaussian (OU) noise the filter-function prediction reproduces exact Monte Carlo."""
    sysm = models.get_system("X")
    prob = MT.Problem(sysm, C.GATE_1Q)
    p = prob.init_params(0, rel_noise=0.0)
    env = {"freq": [("ou", 0.2, (C.TWO_PI * 1e-3) ** 2)], "amp": [("ou", 0.05, 0.01 ** 2)]}
    I0 = float(prob.nominal_infid(p))
    pred = prob.predicted(p, [{nm: SP.on_grid(c, prob.w) for nm, c in env.items()}])[0] - I0
    inf, _ = NS.coherent_mc(prob, p, env, 4000, seed=3, substeps=4)
    mc = float(np.mean(inf)) - I0
    assert abs(pred - mc) / mc < 0.10, (pred, mc)


if __name__ == "__main__":
    test_intrastep_quadrature_matches_van_loan()
    test_knapsack_matches_linear_program()
    test_first_order_prediction_matches_monte_carlo_small_noise()
    print("all tests passed")
