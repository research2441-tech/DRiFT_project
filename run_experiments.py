"""All experiments (E1-E10). Every number in the manuscript is produced by this script.

Usage:  python run_experiments.py e1 --gate X        (see the `main` function for every stage)
Outputs are written next to this file (flat layout): results_*.npz / results_*.json / pulses_*.pkl
"""
import argparse
import json
import pickle
import time
import numpy as np
import jax
import config as C
import models
import methods as MT
import spectra as SP
import noise_sim as NS
import core

GATES_1Q = ["X", "SX"]
SEEDS = {"X": 10, "SX": 10, "SQiSW": 5}


def timing(gate):
    return C.GATE_2Q if gate == "SQiSW" else C.GATE_1Q


def iters(gate):
    return C.ITERS_2Q if gate == "SQiSW" else C.ITERS_1Q


def ntraj(gate):
    return C.EVAL_TRAJ_2Q if gate == "SQiSW" else C.EVAL_TRAJ_1Q


def n_envs(gate):
    return (60, 40) if gate == "SQiSW" else (C.N_TEST_IN, C.N_TEST_OOD)


def test_envs(sysm, gate, seed=2026):
    n_in, n_ood = n_envs(gate)
    rng = np.random.default_rng(seed)
    e_in = [SP.sample_env(sysm, rng, "in") for _ in range(n_in)]
    e_ood = [SP.sample_env(sysm, rng, "ood") for _ in range(n_ood)]
    return e_in, e_ood


def env_descriptors(prob, envs):
    """Spectral descriptors used for set membership and failure analysis."""
    out = []
    for env in envs:
        d = {"in_set": True, "var": {}, "dc_frac": {}, "n_tls": 0, "tls_rates": []}
        for nm, comps in env.items():
            S = SP.on_grid(comps, prob.w)
            d["in_set"] &= SP.in_set(S, prob.aset[nm], prob.weights)
            tot = float(np.dot(prob.weights, S))
            d["var"][nm] = tot
            d["dc_frac"][nm] = float(S[0] / tot) if tot > 0 else 0.0
            for c in comps:
                if c[0] == "rtn":
                    d["n_tls"] += 1
                    d["tls_rates"].append(float(c[1]))
        out.append(d)
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ====================================================================== E1 main comparison
def e1(gate):
    sysm = models.get_system(gate)
    prob = MT.Problem(sysm, timing(gate))
    e_in, e_ood = test_envs(sysm, gate)
    desc_in, desc_ood = env_descriptors(prob, e_in), env_descriptors(prob, e_ood)
    nseed = SEEDS[gate]
    R = {k: {} for k in ["I_in", "I_ood", "L_in", "I0", "Imk", "cert", "pred_in", "train_time", "hist"]}
    pulses = {}
    for m in MT.METHODS:
        for k in R:
            R[k][m] = []
        pulses[m] = []
        seeds = [0] if m == "DRAG" else range(nseed)
        for sd in seeds:
            t0 = time.time()
            p, hist, tt = MT.train(prob, m, sd, iters(gate))
            I_in, L_in, I0, Imk = NS.evaluate(prob, p, e_in, ntraj(gate), seed0=10_000)
            I_ood, _, _, _ = NS.evaluate(prob, p, e_ood, ntraj(gate), seed0=20_000)
            cert, _ = prob.certificate(p)
            pred = prob.predicted(p, [{nm: SP.on_grid(c, prob.w) for nm, c in env.items()} for env in e_in])
            for k, v in zip(["I_in", "I_ood", "L_in", "I0", "Imk", "cert", "pred_in", "train_time", "hist"],
                            [I_in, I_ood, L_in, I0, Imk, cert, pred, tt, hist]):
                R[k][m].append(v)
            pulses[m].append({k: np.asarray(v) for k, v in p.items()})
            print(f"[E1 {gate}] {m:10s} seed {sd}  coh-in {np.mean(I_in - (Imk - I0)):.3e}  "
                  f"tot-in {np.mean(I_in):.3e}  ood {np.mean(I_ood):.3e}  cert {cert:.3e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # DRAG has no random initialisation: replicate for array shape compatibility
        if m == "DRAG":
            for k in R:
                R[k][m] = R[k][m] * nseed
            pulses[m] = pulses[m] * nseed
    arrays = {f"{k}__{m}": np.array(R[k][m]) for k in R if k != "hist" for m in MT.METHODS}
    np.savez(f"results_E1_{gate}.npz", **arrays)
    with open(f"pulses_{gate}.pkl", "wb") as f:
        pickle.dump(pulses, f)
    with open(f"hist_{gate}.pkl", "wb") as f:
        pickle.dump(R["hist"], f)
    save_json({"in": desc_in, "ood": desc_ood}, f"envs_{gate}.json")


# ====================================================================== E2 ablation
ABLATION = {  # (dro, exact intra-step quadrature, leakage-resolved)
    "Base": (False, False, False), "+A": (True, False, False), "+B": (False, True, False),
    "+C": (False, False, True), "+A+B": (True, True, False), "+A+C": (True, False, True),
    "+B+C": (False, True, True), "Full": (True, True, True),
}


def e2(gate, nseed=None):
    sysm = models.get_system(gate)
    prob = MT.Problem(sysm, timing(gate))
    e_in, _ = test_envs(sysm, gate)
    variants = ABLATION if gate != "SQiSW" else {k: ABLATION[k] for k in ["Full", "+B+C", "+A+C", "+A+B"]}
    nseed = nseed or (10 if gate != "SQiSW" else 3)
    out = {}
    for name, (dro, exact, leak) in variants.items():
        out[name] = {"I_in": [], "cert": [], "I0": [], "Imk": []}
        for sd in range(nseed):
            loss = prob.loss_fn("DRiFT", dro=dro, q=C.GL_NODES if exact else 0, leak_resolved=leak)
            lr = C.LR_2Q if gate == "SQiSW" else C.LR
            p, _ = MT.adam(loss, prob.init_params(sd), iters(gate), lr=lr)
            I_in, _, I0, Imk = NS.evaluate(prob, p, e_in, ntraj(gate), seed0=10_000)
            cert, _ = prob.certificate(p)
            for k, v in zip(["I_in", "cert", "I0", "Imk"], [I_in, cert, I0, Imk]):
                out[name][k].append(v)
            print(f"[E2 {gate}] {name:6s} seed {sd} coh {np.mean(I_in - (Imk - I0)):.3e} cert {cert:.3e}", flush=True)
    np.savez(f"results_E2_{gate}.npz", **{f"{k}__{n}": np.array(v[k]) for n, v in out.items() for k in v})


# ====================================================================== E3 sensitivity
def e3(nseed=5):
    gate = "X"
    sysm = models.get_system(gate)
    out = {}
    e_in, _ = test_envs(sysm, gate)
    e_in = e_in[:60]
    base_prob = MT.Problem(sysm, timing(gate))

    def run(prob, tag, method):
        res = []
        for sd in range(nseed):
            p, _, _ = MT.train(prob, method, sd, iters(gate))
            I_in, _, I0, Imk = NS.evaluate(prob, p, e_in, 128, seed0=10_000)
            res.append(np.mean(I_in - (Imk - I0)))
        out[tag] = res
        print(f"[E3] {tag:28s} coh {np.mean(res):.3e} +- {np.std(res):.1e}", flush=True)

    for kappa in [1.25, 1.5, 2.0, 3.0, 4.0, 6.0]:
        run(MT.Problem(sysm, timing(gate), kappa=kappa), f"kappa={kappa}", "DRiFT")
    for slack in [1.0, 1.5, 2.0, 3.0]:
        run(MT.Problem(sysm, timing(gate), slack=slack), f"slack={slack}", "DRiFT")
    for T in [12.0, 16.0, 24.0, 28.0]:
        N = int(2 * T)
        for m in ["GRAPE", "FF-nominal", "CVaR", "DRiFT"]:
            run(MT.Problem(sysm, dict(T=T, N=N, K=12)), f"T={T}|{m}", m)
    for K in [6, 8, 16, 20]:
        for m in ["GRAPE", "DRiFT"]:
            run(MT.Problem(sysm, dict(T=20.0, N=40, K=K)), f"K={K}|{m}", m)
    for m in ["GRAPE", "FF-nominal", "CVaR", "DRiFT"]:  # T = 20 ns, K = 12 reference row
        run(base_prob, f"T=20.0|{m}", m)
    save_json(out, "results_E3_X.json")


# ====================================================================== E4/E5 robustness + generalisation
def e45(gate):
    sysm = models.get_system(gate)
    prob = MT.Problem(sysm, timing(gate))
    with open(f"pulses_{gate}.pkl", "rb") as f:
        pulses = pickle.load(f)
    e_in, _ = test_envs(sysm, gate)
    e_in = e_in[:40]
    nseed = 5 if gate != "SQiSW" else 3
    scales = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    alphas = [0.98, 0.99, 1.0, 1.01, 1.02]
    out = {}
    for m in MT.METHODS:
        for sd in range(nseed if m != "DRAG" else 1):
            p = {k: jax.numpy.asarray(v) for k, v in pulses[m][sd].items()}
            for s in scales:
                envs = [SP.scale_env(e, s) for e in e_in]
                I, _, I0, Imk = NS.evaluate(prob, p, envs, ntraj(gate) // 2, seed0=30_000, with_markov=(s == 1.0))
                out.setdefault(f"scale={s}|{m}", []).append((I - (Imk - I0)).tolist())
            for a in alphas:
                I, _, I0, Imk = NS.evaluate(prob, p, e_in, ntraj(gate) // 2, seed0=40_000, alpha_scale=a)
                out.setdefault(f"alpha={a}|{m}", []).append((I - (Imk - I0)).tolist())
            print(f"[E4/5 {gate}] {m} seed {sd} done", flush=True)
    save_json(out, f"results_E45_{gate}.json")


# ====================================================================== E6 computational cost
def e6():
    out = {}
    for gate in ["X", "SQiSW"]:
        sysm = models.get_system(gate)
        for N_scale in [0.5, 1.0, 2.0]:
            tm = dict(timing(gate))
            tm["N"] = int(tm["N"] * N_scale)
            prob = MT.Problem(sysm, tm)
            p = prob.init_params(0)
            for m in MT.METHODS[1:]:
                f = jax.jit(jax.value_and_grad(prob.loss_fn(m)))
                v, g = f(p)
                jax.block_until_ready(g)
                t = time.time()
                for _ in range(10):
                    v, g = f(p)
                    jax.block_until_ready(g)
                key = f"{gate}|N={tm['N']}|{m}"
                out[key] = (time.time() - t) / 10
                print(f"[E6] {key:28s} {out[key] * 1e3:.2f} ms/iter", flush=True)
    save_json(out, "results_E6.json")


# ====================================================================== E10 drift
def drift_trace(sysm, rng, hours=72):
    """Hourly spectral environments: random-walk 1/f and quasi-static levels, TLS birth/death."""
    env_t = []
    lv = {s["name"]: [0.0, 0.0] for s in sysm.sources}
    tls = {s["name"]: [] for s in sysm.sources if s["name"].startswith("freq")}
    for h in range(hours):
        env = {}
        for s in sysm.sources:
            nm = s["name"]
            lv[nm] = [float(np.clip(x + 0.12 * rng.standard_normal(), -np.log(2.5), np.log(2.5))) for x in lv[nm]]
            f1, fq = np.exp(lv[nm][0]), np.exp(lv[nm][1])
            if nm.startswith("freq"):
                tls[nm] = [t for t in tls[nm] if rng.random() > 0.15]
                if rng.random() < 0.20 and len(tls[nm]) < 2:
                    tls[nm].append((float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0)))),
                                    float(rng.uniform(0.4, 1.0) * C.TLS_B)))
                env[nm] = SP._freq_env(C.FREQ_1F_RMS * f1, C.FREQ_QS_RMS * fq, list(tls[nm]))
            elif nm == "amp":
                env[nm] = SP._amp_env(C.AMP_1F_RMS * f1, C.AMP_QS_RMS * fq)
            else:
                env[nm] = SP._amp_env(2 * C.AMP_1F_RMS * f1, C.AMP_QS_RMS * fq)
        env_t.append(env)
    return env_t


def e10(gate, n_real=6, hours=72):
    sysm = models.get_system(gate)
    prob = MT.Problem(sysm, timing(gate))
    with open(f"pulses_{gate}.pkl", "rb") as f:
        pulses = pickle.load(f)
    rng = np.random.default_rng(99)
    traces = [drift_trace(sysm, rng, hours) for _ in range(n_real)]
    member = [[all(SP.in_set(SP.on_grid(env[nm], prob.w), prob.aset[nm], prob.weights) for nm in env)
               for env in tr] for tr in traces]
    out = {"member": member}
    for m in MT.METHODS:
        p = {k: jax.numpy.asarray(v) for k, v in pulses[m][0].items()}
        Imk = NS.markovian_infid(prob, p)
        I0 = float(prob.nominal_infid(p))
        series = []
        for r, tr in enumerate(traces):
            I, _, _, _ = NS.evaluate(prob, p, tr, 64 if gate != "SQiSW" else 48, seed0=50_000 + 1000 * r,
                                     with_markov=False)
            series.append((I + (Imk - I0)).tolist())
        out[m] = series
        out[m + "|cert"] = prob.certificate(p)[0] + (Imk - I0)
        print(f"[E10 {gate}] {m} mean {np.mean(series):.3e}", flush=True)
    save_json(out, f"results_E10_{gate}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["e1", "e2", "e3", "e45", "e6", "e10"])
    ap.add_argument("--gate", default="X")
    a = ap.parse_args()
    t = time.time()
    if a.stage == "e1":
        e1(a.gate)
    elif a.stage == "e2":
        e2(a.gate)
    elif a.stage == "e3":
        e3()
    elif a.stage == "e45":
        e45(a.gate)
    elif a.stage == "e6":
        e6()
    elif a.stage == "e10":
        e10(a.gate)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
