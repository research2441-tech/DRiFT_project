"""Ground-truth evaluation: exact time-domain Monte Carlo of non-Gaussian, temporally correlated
noise (OU + random-telegraph + quasi-static) plus Markovian T1/T_phi decay from a Lindblad model.

The coherent part makes no perturbative (first-order) approximation; it is used to validate the
filter-function predictions and every method on equal footing (common random numbers).
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm as sp_expm
import config as C
import core
import spectra as SP

jax.config.update("jax_enable_x64", True)


@jax.jit
def _evolve(Hs, dt):
    """Hs: (ntraj, Ns, d, d) Hermitian -> final unitaries (ntraj, d, d)."""
    w, V = jnp.linalg.eigh(Hs)
    Us = jnp.einsum("tsij,tsj,tskj->tsik", V, jnp.exp(-1j * w * dt), V.conj())
    d = Hs.shape[-1]
    U0 = jnp.broadcast_to(jnp.eye(d, dtype=complex), (Hs.shape[0], d, d))

    def body(U, Uk):
        return jnp.einsum("tij,tjk->tik", Uk, U), None

    U, _ = jax.lax.scan(body, U0, jnp.swapaxes(Us, 0, 1))
    return U


def coherent_mc(prob, params, env, ntraj, seed, substeps=C.SUBSTEPS, alpha_scale=1.0):
    sysm = prob.sys
    u = np.asarray(core.controls(params["coef"], prob.B))
    N, nc = u.shape
    Ns = N * substeps
    dts = prob.dt / substeps
    rng = np.random.default_rng(seed)
    us = np.repeat(u, substeps, axis=0)                       # (Ns, nc)
    uu = np.broadcast_to(us, (ntraj, Ns, nc)).copy()
    Hadd = np.zeros((ntraj, Ns) + sysm.H0.shape, complex)
    for s in sysm.sources:
        x = SP.synthesize(env[s["name"]], ntraj, Ns, dts, rng)
        if s["kind"] == "add":
            Hadd += x[:, :, None, None] * s["op"][None, None]
        else:
            idx = np.array(s["ctrl"])
            uu[:, :, idx] *= (1.0 + x[:, :, None])
    Hs = alpha_scale * sysm.H0[None, None] + np.einsum("tsc,cij->tsij", uu, sysm.Hc) + Hadd
    U = np.asarray(_evolve(jnp.asarray(Hs), dts))
    Z = core.phase_correction(params.get("phis", jnp.zeros(0)), sysm.phase_ops)
    if Z is not None:
        U = np.einsum("ij,tjk->tik", np.asarray(Z), U)
    G = sysm.target
    idx = sysm.comp_idx
    d = sysm.dsub
    M = np.einsum("ji,tjk->tik", G.conj(), U)[:, idx][:, :, idx]
    F = (np.real(np.einsum("tij,tij->t", M, M.conj())) + np.abs(np.einsum("tii->t", M)) ** 2) / (d * (d + 1))
    leak = 1.0 - np.sum(np.abs(U[:, idx][:, :, idx]) ** 2, axis=(1, 2)) / d
    return 1.0 - F, leak


def _liouvillian(H, Ls):
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for Lk in Ls:
        LdL = Lk.conj().T @ Lk
        L += np.kron(Lk.conj(), Lk) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    return L


def markovian_infid(prob, params, alpha_scale=1.0):
    """Average infidelity with T1 and Markovian T_phi only (no coherent noise)."""
    sysm = prob.sys
    u = np.asarray(core.controls(params["coef"], prob.B))
    Ls = [np.sqrt(1.0 / C.T1) * a for a in sysm.a_ops] + [np.sqrt(2.0 / C.TPHI) * n for n in sysm.n_ops]
    d = sysm.dfull
    S = np.eye(d * d, dtype=complex)
    for k in range(u.shape[0]):
        H = alpha_scale * sysm.H0 + np.einsum("c,cij->ij", u[k], sysm.Hc)
        S = sp_expm(_liouvillian(H, Ls) * prob.dt) @ S
    Z = core.phase_correction(params.get("phis", jnp.zeros(0)), sysm.phase_ops)
    Z = np.eye(d) if Z is None else np.asarray(Z)
    G = sysm.target
    idx = sysm.comp_idx
    ds = sysm.dsub

    def chan(i, j):
        rho = np.zeros((d, d), complex)
        rho[i, j] = 1.0
        out = (S @ rho.reshape(-1, order="F")).reshape(d, d, order="F")
        return G.conj().T @ (Z @ out @ Z.conj().T) @ G

    tot = 0.0
    for i in idx:
        Y = chan(i, i)
        tot += np.real(np.sum(np.diag(Y)[idx]))
        for j in idx:
            tot += np.real(chan(i, j)[i, j])
    return 1.0 - tot / (ds * (ds + 1))


def evaluate(prob, params, envs, ntraj, seed0=0, with_markov=True, alpha_scale=1.0):
    """Ground-truth average infidelity for every environment (common random numbers via seed0+e).

    Returns total infidelity (coherent Monte Carlo + Markovian excess), leakage, the nominal
    infidelity and the Markovian infidelity of the noiseless pulse.
    """
    if alpha_scale == 1.0:
        I_nom = float(prob.nominal_infid(params))
    else:
        inf0, _ = coherent_mc(prob, params, {nm: [] for nm in prob.S_nom}, 1, 0, alpha_scale=alpha_scale)
        I_nom = float(inf0[0])
    I_mk = markovian_infid(prob, params, alpha_scale) if with_markov else I_nom
    out_I, out_L = [], []
    for e, env in enumerate(envs):
        inf, lk = coherent_mc(prob, params, env, ntraj, seed0 + e, alpha_scale=alpha_scale)
        out_I.append(float(np.mean(inf)) + (I_mk - I_nom))
        out_L.append(float(np.mean(lk)))
    return np.array(out_I), np.array(out_L), I_nom, I_mk
