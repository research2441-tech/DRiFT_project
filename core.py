"""Differentiable propagation, nominal fidelity and leakage-resolved filter functions (JAX)."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import config as C

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ pulse basis
def sine_basis(T, N, K):
    """B[k, l] = sin(l pi t_k / T) at PWC step midpoints t_k, l = 1..K (zero at both ends)."""
    t = (np.arange(N) + 0.5) * T / N
    l = np.arange(1, K + 1)
    return np.sin(np.pi * np.outer(t, l) / T)


def controls(coef, B):
    """coef: (n_ctrl, K) -> u: (N, n_ctrl)."""
    return B @ coef.T


# ------------------------------------------------------------------ propagation
def step_hamiltonians(u, H0, Hc):
    return H0[None] + jnp.einsum("kc,cij->kij", u, Hc)


def propagate(Hk, dt):
    """Step propagators V_k and cumulative propagators U_k (state *before* step k), and U_N."""
    V = jax.vmap(lambda h: expm(-1j * h * dt))(Hk)
    d = Hk.shape[-1]

    def body(U, Vk):
        return Vk @ U, U

    UN, Ubefore = jax.lax.scan(body, jnp.eye(d, dtype=complex), V)
    return V, Ubefore, UN


def phase_correction(phis, phase_ops):
    """Virtual-Z correction exp(+i sum_j phi_j n_j) (diagonal operators)."""
    if len(phase_ops) == 0:
        return None
    diag = sum(p * jnp.diag(op).real for p, op in zip(phis, phase_ops))
    return jnp.diag(jnp.exp(1j * diag))


def avg_gate_fidelity(U, G, P_idx, d_sub):
    """Average gate fidelity of the (possibly leaky) restriction M = P G^dag U P (Pedersen et al.)."""
    M = (G.conj().T @ U)[jnp.ix_(P_idx, P_idx)]
    return (jnp.real(jnp.trace(M @ M.conj().T)) + jnp.abs(jnp.trace(M)) ** 2) / (d_sub * (d_sub + 1))


def leakage(U, P_idx, d_sub):
    Ms = U[:, P_idx]
    return 1.0 - jnp.real(jnp.sum(jnp.abs(Ms[P_idx, :]) ** 2)) / d_sub


# ------------------------------------------------------------------ filter functions
def gl_nodes(dt, q):
    if q == 0:  # zeroth-order toggling frame: one node at the step start, weight dt
        return np.array([0.0]), np.array([dt])
    x, wq = np.polynomial.legendre.leggauss(q)
    return 0.5 * dt * (x + 1), 0.5 * dt * wq


def toggling_operators(Hk, Ubefore, Ek, dt, q):
    """X_{k,q} = W^dag E_k W with W = exp(-i H_k s_q) U_k; returns (N*q, d, d) and node times."""
    s, wq = gl_nodes(dt, q)
    W = jax.vmap(lambda h: jax.vmap(lambda sq: expm(-1j * h * sq))(jnp.asarray(s)))(Hk)  # (N,q,d,d)
    WU = jnp.einsum("kqij,kjl->kqil", W, Ubefore)
    X = jnp.einsum("kqji,kjl,kqlm->kqim", WU.conj(), Ek, WU)
    N = Hk.shape[0]
    tk = np.arange(N)[:, None] * dt + s[None, :]
    wts = np.broadcast_to(wq[None, :], (N, len(s)))
    return X.reshape(N * len(s), *X.shape[2:]), tk.reshape(-1), wts.reshape(-1)


def filter_function(X, tnodes, wnodes, w, P_idx, d_sub, leak_resolved=True):
    """Symmetrised first-order filter function Phi_bar(w_m) for one noise source.

    R(w) = sum_n wnodes_n e^{i w t_n} X_n;
    Phi(w) = Tr(P R R^dag P)/d - |Tr(P R P)|^2/d^2 (leak-resolved: full-space product),
    Phi_bar = [Phi(w) + Phi(-w)] / 2.
    """
    ph = jnp.exp(1j * jnp.outer(w, tnodes)) * wnodes[None, :]       # (M, Nn)
    R = jnp.einsum("mn,nij->mij", ph, X)                              # (M, d, d)
    Rs = R[:, P_idx][:, :, P_idx]
    tr = jnp.einsum("mii->m", Rs)
    if leak_resolved:
        RRd = jnp.einsum("mij,mkj->mik", R[:, P_idx, :], R[:, P_idx, :].conj())
        RdR = jnp.einsum("mji,mjk->mik", R[:, :, P_idx].conj(), R[:, :, P_idx])
    else:
        RRd = jnp.einsum("mij,mkj->mik", Rs, Rs.conj())
        RdR = jnp.einsum("mji,mjk->mik", Rs.conj(), Rs)
    a = jnp.real(jnp.einsum("mii->m", RRd)) / d_sub
    b = jnp.real(jnp.einsum("mii->m", RdR)) / d_sub
    c = jnp.abs(tr) ** 2 / d_sub ** 2
    return 0.5 * (a + b) - c


def source_operators(system, u):
    """E_k for every noise source: (N, d, d) arrays."""
    N = u.shape[0]
    out = []
    for s in system.sources:
        if s["kind"] == "add":
            out.append(jnp.broadcast_to(jnp.asarray(s["op"]), (N,) + s["op"].shape))
        else:
            idx = np.array(s["ctrl"])
            out.append(jnp.einsum("kc,cij->kij", u[:, idx], jnp.asarray(system.Hc[idx])))
    return out
