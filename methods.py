"""Proposed method (DRiFT) and baselines: objectives, initial guesses and the optimiser."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import config as C
import core
import spectra as SP

METHODS = ["DRAG", "GRAPE", "Ensemble", "QS-FO", "FF-nominal", "CVaR", "DRiFT"]
LABELS = {
    "DRAG": "Calibrated DRAG",
    "GRAPE": "GRAPE (nominal)",
    "Ensemble": "Sampling-based robust",
    "QS-FO": "Quasi-static first-order",
    "FF-nominal": "Filter function (nominal PSD)",
    "CVaR": "Risk-aware CVaR",
    "DRiFT": "DRiFT (proposed)",
}


class Problem:
    """Pre-computes everything a given system needs for every method."""

    def __init__(self, system, timing, kappa=C.KAPPA, slack=C.BUDGET_SLACK, seed_aux=1234):
        self.sys = system
        self.T, self.N, self.K = timing["T"], timing["N"], timing["K"]
        self.dt = self.T / self.N
        self.B = jnp.asarray(core.sine_basis(self.T, self.N, self.K))
        self.H0 = jnp.asarray(system.H0)
        self.Hc = jnp.asarray(system.Hc)
        self.G = jnp.asarray(system.target)
        self.P_idx = jnp.asarray(system.comp_idx)
        self.dsub = system.dsub
        self.bounds = jnp.asarray(system.ctrl_bounds)
        self.w, self.weights = SP.frequency_grid()
        self.wj, self.weightsj = jnp.asarray(self.w), jnp.asarray(self.weights)
        env = SP.nominal_env(system)
        self.S_nom = {nm: SP.on_grid(c, self.w) for nm, c in env.items()}
        self.var_nom = {nm: SP.variance(c) for nm, c in env.items()}
        self.aset = SP.ambiguity_set(system, self.w, self.weights, kappa=kappa, slack=slack)
        rng = np.random.default_rng(seed_aux)
        # CVaR scenarios drawn from the same prior that defines the ambiguity set
        scen = [SP.sample_env(system, rng, "in") for _ in range(C.CVAR_K)]
        self.S_scen = {nm: jnp.asarray(np.stack([SP.on_grid(e[nm], self.w) for e in scen]))
                       for nm in self.S_nom}
        # quasi-static ensemble for the sampling-based baseline
        self.ens = {nm: rng.standard_normal(C.ENSEMBLE_M) * np.sqrt(self.var_nom[nm]) for nm in self.S_nom}
        self.scale = self.dsub / (self.dsub + 1.0)   # entanglement -> average infidelity

    # ---------------------------------------------------------- building blocks
    def unpack(self, params):
        u = core.controls(params["coef"], self.B)
        return u

    def final_U(self, params, u=None, extra_H=None):
        u = self.unpack(params) if u is None else u
        Hk = core.step_hamiltonians(u, self.H0, self.Hc)
        if extra_H is not None:
            Hk = Hk + extra_H[None]
        V, Ub, UN = core.propagate(Hk, self.dt)
        Z = core.phase_correction(params.get("phis", jnp.zeros(0)), self.sys.phase_ops)
        if Z is not None:
            UN = Z @ UN
        return Hk, Ub, UN

    def nominal_infid(self, params):
        _, _, UN = self.final_U(params)
        return 1.0 - core.avg_gate_fidelity(UN, self.G, self.P_idx, self.dsub)

    def penalty(self, u):
        return C.BOUND_PENALTY * jnp.sum(jax.nn.relu(jnp.abs(u) / self.bounds[None] - 1.0) ** 2)

    def filter_functions(self, params, q=C.GL_NODES, leak_resolved=True):
        u = self.unpack(params)
        Hk, Ub, UN = self.final_U(params, u)      # UN already includes the virtual-Z correction
        phis = {}
        for s, Ek in zip(self.sys.sources, core.source_operators(self.sys, u)):
            X, tn, wn = core.toggling_operators(Hk, Ub, Ek, self.dt, q)
            phis[s["name"]] = core.filter_function(X, jnp.asarray(tn), jnp.asarray(wn), self.wj,
                                                   self.P_idx, self.dsub, leak_resolved)
        return phis, u, UN

    # ---------------------------------------------------------- objectives
    def loss_fn(self, method, dro=True, q=C.GL_NODES, leak_resolved=True):
        G, P_idx, dsub = self.G, self.P_idx, self.dsub

        def base(params):
            phis, u, UN = self.filter_functions(params, q, leak_resolved)
            I0 = 1.0 - core.avg_gate_fidelity(UN, G, P_idx, dsub)
            return phis, u, I0

        if method == "GRAPE":
            def f(params):
                u = self.unpack(params)
                return self.nominal_infid(params) + self.penalty(u)
            return f

        if method == "Ensemble":
            names = [s["name"] for s in self.sys.sources]

            def f(params):
                u = self.unpack(params)
                tot = 0.0
                for m in range(C.ENSEMBLE_M):
                    uu = u
                    extra = jnp.zeros_like(self.H0)
                    for s in self.sys.sources:
                        dlt = self.ens[s["name"]][m]
                        if s["kind"] == "add":
                            extra = extra + dlt * jnp.asarray(s["op"])
                        else:
                            idx = np.array(s["ctrl"])
                            uu = uu.at[:, idx].multiply(1.0 + dlt)
                    _, _, UN = self.final_U(params, uu, extra)
                    tot = tot + 1.0 - core.avg_gate_fidelity(UN, G, P_idx, dsub)
                return tot / C.ENSEMBLE_M + self.penalty(u)
            return f

        if method == "QS-FO":
            def f(params):
                phis, u, I0 = base(params)
                noise = sum(self.var_nom[nm] * ph[0] for nm, ph in phis.items())
                return I0 + self.scale * noise + self.penalty(u)
            return f

        if method == "FF-nominal" or (method == "DRiFT" and not dro):
            def f(params):
                phis, u, I0 = base(params)
                noise = sum(jnp.dot(self.weightsj, jnp.asarray(self.S_nom[nm]) * ph) for nm, ph in phis.items())
                return I0 + self.scale * noise + self.penalty(u)
            return f

        if method == "CVaR":
            kt = int(np.ceil(C.CVAR_ALPHA * C.CVAR_K))

            def f(params):
                phis, u, I0 = base(params)
                per = sum(self.S_scen[nm] @ (self.weightsj * ph) for nm, ph in phis.items())
                tail = jnp.sort(per)[-kt:]
                return I0 + self.scale * jnp.mean(tail) + self.penalty(u)
            return f

        if method == "DRiFT":
            def f(params):
                phis, u, I0 = base(params)
                noise = 0.0
                for nm, ph in phis.items():
                    a = self.aset[nm]
                    val, _ = SP.knapsack_jax(ph, self.weightsj, jnp.asarray(a["lo"]), jnp.asarray(a["hi"]),
                                             a["budget"])
                    noise = noise + val
                return I0 + self.scale * noise + self.penalty(u)
            return f
        raise ValueError(method)

    # ---------------------------------------------------------- certificate
    def certificate(self, params):
        """Certified (first-order) worst-case average infidelity over the ambiguity set."""
        phis, u, UN = self.filter_functions(params)
        I0 = float(1.0 - core.avg_gate_fidelity(UN, self.G, self.P_idx, self.dsub))
        wc = 0.0
        for nm, ph in phis.items():
            a = self.aset[nm]
            v, _ = SP.knapsack_np(np.asarray(ph), self.weights, a["lo"], a["hi"], a["budget"])
            wc += v
        return I0 + self.scale * wc, I0

    def predicted(self, params, S_list):
        """First-order predictions of the average infidelity for a list of spectra (dicts)."""
        phis, u, UN = self.filter_functions(params)
        I0 = float(1.0 - core.avg_gate_fidelity(UN, self.G, self.P_idx, self.dsub))
        ph = {nm: np.asarray(v) for nm, v in phis.items()}
        return [I0 + self.scale * sum(float(np.dot(self.weights, Sd[nm] * ph[nm])) for nm in ph) for Sd in S_list]

    # ---------------------------------------------------------- initial guesses
    def drag_like(self, amp=1.0, beta=1.0):
        """Raised-cosine (+DRAG quadrature) envelopes sampled at the step midpoints."""
        t = (np.arange(self.N) + 0.5) * self.dt
        env = (1 - np.cos(2 * np.pi * t / self.T)) / self.T
        denv = 2 * np.pi * np.sin(2 * np.pi * t / self.T) / self.T ** 2
        nc = len(self.sys.Hc)
        u = np.zeros((self.N, nc))
        if self.sys.name.startswith("1Q"):
            theta = {"1Q-X": np.pi, "1Q-SX": np.pi / 2, "1Q-H": np.pi}[self.sys.name]
            u[:, 0] = amp * theta * env
            u[:, 1] = -beta * amp * theta * denv / C.ALPHA
        else:
            u[:, 4] = amp * (np.pi / 4) * env
        return u

    def project(self, u):
        Bn = np.asarray(self.B)
        coef, *_ = np.linalg.lstsq(Bn, u, rcond=None)
        return coef.T

    def init_params(self, seed, rel_noise=0.15):
        rng = np.random.default_rng(seed)
        coef = self.project(self.drag_like())
        scale = np.max(np.abs(coef)) if np.max(np.abs(coef)) > 0 else 1.0
        coef = coef + rel_noise * scale * rng.standard_normal(coef.shape)
        p = {"coef": jnp.asarray(coef)}
        if self.sys.phase_ops:
            p["phis"] = jnp.zeros(len(self.sys.phase_ops))
        return p


# ------------------------------------------------------------------ optimiser
def adam(loss, params, iters, lr=C.LR, log_every=0):
    vg = jax.jit(jax.value_and_grad(loss))
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-12
    hist = []
    best, best_p = np.inf, params
    for it in range(1, iters + 1):
        val, g = vg(params)
        val = float(val)
        hist.append(val)
        if val < best:
            best, best_p = val, params
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * (it - 1) / iters)) + lr * 0.02
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * jnp.abs(b) ** 2, v, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** it), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** it), v)
        params = jax.tree_util.tree_map(lambda p, a, b: p - lr_t * a / (jnp.sqrt(b) + eps), params, mh, vh)
        if log_every and it % log_every == 0:
            print(f"  it {it:4d}  loss {val:.3e}", flush=True)
    return best_p, np.array(hist)


def calibrate_drag(prob, seed=0, iters=300):
    """Analytic baseline: only the amplitude, DRAG coefficient (1Q) or coupling area (2Q) and the
    virtual-Z phases are calibrated on the nominal model, as done in routine device calibration."""
    c0 = jnp.asarray(prob.project(prob.drag_like(1.0, 1.0)))
    cb = jnp.asarray(prob.project(prob.drag_like(1.0, 2.0))) - c0
    nphi = len(prob.sys.phase_ops)

    def to_params(th):
        p = {"coef": th["a"] * (c0 + (th["b"] - 1.0) * cb)}
        if nphi:
            p["phis"] = th["phis"]
        return p

    def loss(th):
        return prob.nominal_infid(to_params(th))

    th = {"a": jnp.asarray(1.0), "b": jnp.asarray(1.0)}
    if nphi:
        th["phis"] = jnp.zeros(nphi)
    th, _ = adam(loss, th, iters, lr=0.01)
    return to_params(th)


def train(prob, method, seed, iters, **kw):
    t0 = time.time()
    if method == "DRAG":
        p = calibrate_drag(prob, seed)
        return p, np.array([]), time.time() - t0
    p0 = prob.init_params(seed)
    loss = prob.loss_fn(method, **kw)
    lr = C.LR_2Q if prob.sys.name.startswith("2Q") else C.LR
    p, hist = adam(loss, p0, iters, lr=lr)
    return p, hist, time.time() - t0
