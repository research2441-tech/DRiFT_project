"""Figures 2-9 from the saved results (Figure 1 is drawn by fig_architecture.py).
Legends sit outside the plotting area and every panel uses constrained layout so that no
label, legend or marker overlaps another element."""
import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import jax.numpy as jnp
import config as C
import models
import methods as MT
import spectra as SP
import stats_utils as ST

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.8, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d9d9", "grid.linewidth": 0.5,
    "savefig.dpi": 600, "mathtext.fontset": "dejavusans",
})
COLORS = {"DRiFT": "#2a78d6", "CVaR": "#eb6834", "FF-nominal": "#1baf7a", "QS-FO": "#eda100",
          "Ensemble": "#e87ba4", "GRAPE": "#008300", "DRAG": "#6250d6"}
MARK = {"DRiFT": "o", "CVaR": "s", "FF-nominal": "^", "QS-FO": "D", "Ensemble": "v", "GRAPE": "P", "DRAG": "X"}
LS = {"DRiFT": "-", "CVaR": "--", "FF-nominal": "-.", "QS-FO": ":", "Ensemble": (0, (5, 1.5)),
      "GRAPE": (0, (3, 1, 1, 1)), "DRAG": (0, (1, 1))}
ORDER = ["DRAG", "GRAPE", "Ensemble", "QS-FO", "FF-nominal", "CVaR", "DRiFT"]
SHORT = {"DRAG": "DRAG", "GRAPE": "GRAPE", "Ensemble": "Ensemble", "QS-FO": "QS-FO",
         "FF-nominal": "FF-nom", "CVaR": "CVaR", "DRiFT": "DRiFT"}
GSHORT = {"X": "X", "SX": r"$\sqrt{X}$", "SQiSW": r"$\sqrt{\mathrm{iSWAP}}$"}
GATE_TITLE = {"X": "X gate (20 ns)", "SX": r"$\sqrt{X}$ gate (20 ns)", "SQiSW": r"$\sqrt{\mathrm{iSWAP}}$ (40 ns)"}


def save(fig, name):
    fig.savefig(name + ".png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(name + ".pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)


def load_e1(gate):
    f = f"results_E1_{gate}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    out = {}
    for m in ORDER:
        I_in, I_ood = z[f"I_in__{m}"], z[f"I_ood__{m}"]
        exc = (z[f"Imk__{m}"] - z[f"I0__{m}"])[:, None]
        out[m] = dict(coh_in=I_in - exc, coh_ood=I_ood - exc, tot_in=I_in, tot_ood=I_ood,
                      cert=z[f"cert__{m}"], I0=z[f"I0__{m}"], Imk=z[f"Imk__{m}"], pred=z[f"pred_in__{m}"],
                      L=z[f"L_in__{m}"], tt=z[f"train_time__{m}"])
    return out


def method_legend(fig, methods, ncol=7, y=0.0):
    hs = [plt.Line2D([], [], color=COLORS[m], marker=MARK[m], ls=LS[m], lw=1.6, ms=5.5) for m in methods]
    fig.legend(hs, [MT.LABELS[m] for m in methods], loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=2.6, columnspacing=1.2)


# ---------------------------------------------------------------- Fig 2: ambiguity set + filter functions
def fig2():
    gate = "X"
    if not os.path.exists(f"pulses_{gate}.pkl"):
        return
    sysm = models.get_system(gate)
    prob = MT.Problem(sysm, C.GATE_1Q)
    pulses = pickle.load(open(f"pulses_{gate}.pkl", "rb"))
    w = prob.w[1:]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    ax = axs[0]
    a = prob.aset["freq"]
    ax.fill_between(w, a["lo"][1:], a["hi"][1:], color="#2a78d6", alpha=0.13, lw=0, label=r"ambiguity box $[\ell(\omega),h(\omega)]$")
    rng = np.random.default_rng(5)
    for i in range(12):
        S = SP.on_grid(SP.sample_env(sysm, rng, "in")["freq"], prob.w)
        ax.loglog(w, S[1:], color="#9a9a9a", lw=0.6, alpha=0.8, label="test spectra (drift prior)" if i == 0 else None)
    ax.loglog(w, prob.S_nom["freq"][1:], color="#0b0b0b", lw=1.8, label=r"nominal estimate $\hat{S}(\omega)$")
    for m in ["GRAPE", "DRiFT"]:
        p = {k: jnp.asarray(v) for k, v in pulses[m][0].items()}
        ph, _, _ = prob.filter_functions(p)
        _, Sst = SP.knapsack_np(np.asarray(ph["freq"]), prob.weights, a["lo"], a["hi"], a["budget"])
        ax.loglog(w, np.maximum(Sst[1:], 1e-14), color=COLORS[m], lw=1.5, ls=LS[m],
                  label=f"worst case $S^\\star$ vs {SHORT[m]}")
    ax.set_xlabel(r"angular frequency $\omega$ (rad ns$^{-1}$)")
    ax.set_ylabel(r"$S_{\mathrm{freq}}(\omega)$ (rad$^2$ ns$^{-1}$)")
    ax.set_ylim(1e-11, 1e-2)
    ax.set_title("(a) Ambiguity set (frequency noise)", loc="left")
    ax.grid(True, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=7.0)
    ax = axs[1]
    for i, m in enumerate(["GRAPE", "QS-FO", "FF-nominal", "CVaR", "DRiFT"]):
        p = {k: jnp.asarray(v) for k, v in pulses[m][0].items()}
        ph, _, _ = prob.filter_functions(p)
        ax.loglog(w, np.asarray(ph["freq"])[1:], color=COLORS[m], ls=LS[m], lw=1.5, label=SHORT[m])
        ax.plot([w[0] * 0.18 * 1.45 ** i], [float(ph["freq"][0])], marker=MARK[m], color=COLORS[m], ms=4.5,
                ls="none", mec="white", mew=0.5)
    ax.set_xlabel(r"angular frequency $\omega$ (rad ns$^{-1}$)")
    ax.set_ylabel(r"filter function $\bar{\Phi}_{\mathrm{freq}}(\omega)$ (ns$^2$)")
    ax.set_title("(b) Filter functions, X gate", loc="left")
    ax.text(0.02, 0.03, "markers: DC bin", transform=ax.transAxes, fontsize=7.0, color="#52514e")
    ax.grid(True, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=7.0)
    save(fig, "fig2_spectral_set_filter_functions")


# ---------------------------------------------------------------- Fig 3: pulses
def fig3():
    rows = []
    for gate in ["X", "SQiSW"]:
        if os.path.exists(f"pulses_{gate}.pkl"):
            rows.append(gate)
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained")
    ctrl_col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    for i, (gate, m) in enumerate([("X", "GRAPE"), ("X", "DRiFT"), ("SQiSW", "DRiFT")]):
        if gate not in rows:
            continue
        sysm = models.get_system(gate)
        tm = C.GATE_2Q if gate == "SQiSW" else C.GATE_1Q
        prob = MT.Problem(sysm, tm)
        p = pickle.load(open(f"pulses_{gate}.pkl", "rb"))[m][0]
        u = np.asarray(prob.unpack({k: jnp.asarray(v) for k, v in p.items()})) / C.TWO_PI * 1e3  # MHz
        t = (np.arange(prob.N) + 0.5) * prob.dt
        names = [r"$\Omega_x$", r"$\Omega_y$"] if gate != "SQiSW" else \
            [r"$\Omega_{x,1}$", r"$\Omega_{y,1}$", r"$\Omega_{x,2}$", r"$\Omega_{y,2}$", r"$g$"]
        ax = axs[i]
        for c in range(u.shape[1]):
            ax.step(t, u[:, c], where="mid", color=ctrl_col[c], lw=1.3, ls=["-", "--", "-.", ":", "-"][c],
                    label=names[c])
        ax.axhline(0, color="#7a7a7a", lw=0.5)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel(r"amplitude / $2\pi$ (MHz)")
        ttl = {0: "(a) X, GRAPE", 1: "(b) X, DRiFT", 2: r"(c) $\sqrt{\mathrm{iSWAP}}$, DRiFT"}[i]
        ax.set_title(ttl, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5 if gate == "SQiSW" else 2,
                  frameon=False, handlelength=1.8, columnspacing=0.8, fontsize=7.0)
        ax.grid(True)
    save(fig, "fig3_pulses")


# ---------------------------------------------------------------- Fig 4: main comparison
def fig4():
    gates = [g for g in ["X", "SX", "SQiSW"] if load_e1(g) is not None]
    if not gates:
        return
    fig, axs = plt.subplots(2, len(gates), figsize=(7.2, 4.9), layout="constrained", squeeze=False)
    for j, g in enumerate(gates):
        R = load_e1(g)
        for i, key in enumerate(["coh_in", "coh_ood"]):
            ax = axs[i, j]
            data = [R[m][key].mean(0) * 1e4 for m in ORDER]   # seed-averaged per environment
            bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=True, whis=(5, 95),
                            medianprops=dict(color="#0b0b0b", lw=1.1),
                            flierprops=dict(marker=".", ms=2.5, mfc="#7a7a7a", mec="none"))
            for b, m in zip(bp["boxes"], ORDER):
                b.set(facecolor=COLORS[m], alpha=0.75, edgecolor="#333333", lw=0.7)
            for k, m in enumerate(ORDER):
                ax.plot(k + 1, np.mean(data[k]), marker=MARK[m], color="white", mec="#0b0b0b", ms=4.5, mew=0.8)
            ax.set_xticks(range(1, len(ORDER) + 1))
            ax.set_xticklabels([SHORT[m] for m in ORDER], rotation=40, ha="right")
            ax.set_ylabel(r"coherent infidelity ($\times10^{-4}$)")
            ax.set_title(f"({'abcdef'[i * len(gates) + j]}) {GATE_TITLE[g]}, "
                         f"{'in-distribution' if i == 0 else 'out-of-distribution'}", loc="left", fontsize=8)
            ax.grid(True, axis="y")
    save(fig, "fig4_main_comparison")


# ---------------------------------------------------------------- Fig 5: noise scaling + mismatch
def fig5():
    gates = [g for g in ["X", "SQiSW"] if os.path.exists(f"results_E45_{g}.json")]
    if not gates:
        return
    fig, axs = plt.subplots(1, 2 * len(gates), figsize=(7.2, 3.1), layout="constrained", squeeze=False)
    axs = axs[0]
    scales = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    alphas = [0.98, 0.99, 1.0, 1.01, 1.02]
    for j, g in enumerate(gates):
        R = json.load(open(f"results_E45_{g}.json"))
        ax = axs[2 * j]
        for m in ORDER:
            mu = [np.mean(R[f"scale={s}|{m}"]) for s in scales]
            ci = [ST.bootstrap_ci(np.array(R[f"scale={s}|{m}"]), n=800) for s in scales]
            ax.plot(scales, mu, color=COLORS[m], marker=MARK[m], ls=LS[m], lw=1.4, ms=4.5)
            ax.fill_between(scales, [c[0] for c in ci], [c[1] for c in ci], color=COLORS[m], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(scales)
        ax.set_xticklabels([f"{s:g}" for s in scales])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("noise amplitude scale")
        ax.set_ylabel("coherent infidelity")
        ax.set_title(f"({'abcd'[2 * j]}) {GSHORT[g]}: noise", loc="left", fontsize=8)
        ax.grid(True, which="major")
        ax = axs[2 * j + 1]
        for m in ORDER:
            mu = [np.mean(R[f"alpha={a}|{m}"]) for a in alphas]
            ax.plot([(a - 1) * 100 for a in alphas], mu, color=COLORS[m], marker=MARK[m], ls=LS[m], lw=1.4, ms=4.5)
        ax.set_yscale("log")
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"])
        ax.set_xlabel(r"anharmonicity error (%)")
        ax.set_ylabel("coherent infidelity")
        ax.set_title(f"({'abcd'[2 * j + 1]}) {GSHORT[g]}: $\\alpha$ mismatch", loc="left", fontsize=8)
        ax.grid(True, which="major")
    fig.get_layout_engine().set(h_pad=0.05)
    method_legend(fig, ORDER, ncol=4, y=-0.16)
    save(fig, "fig5_robustness_generalisation")


# ---------------------------------------------------------------- Fig 6: ablation + sensitivity
def fig6():
    if not os.path.exists("results_E2_X.npz"):
        return
    z = np.load("results_E2_X.npz")
    names = ["Base", "+A", "+B", "+C", "+A+B", "+A+C", "+B+C", "Full"]
    have3 = os.path.exists("results_E3_X.json")
    fig, axs = plt.subplots(1, 3 if have3 else 1, figsize=(7.2, 2.7), layout="constrained", squeeze=False)
    axs = axs[0]
    ax = axs[0]
    mus, cis = [], []
    for n in names:
        M = z[f"I_in__{n}"] - (z[f"Imk__{n}"] - z[f"I0__{n}"])[:, None]
        mus.append(M.mean() * 1e4)
        lo, hi = ST.bootstrap_ci(M, n=1500)
        cis.append((M.mean() * 1e4 - lo * 1e4, hi * 1e4 - M.mean() * 1e4))
    col = ["#9a9a9a"] + ["#8fbbef"] * 3 + ["#5b98e3"] * 3 + ["#2a78d6"]
    ax.bar(range(len(names)), mus, yerr=np.array(cis).T, color=col, edgecolor="#333333", lw=0.6,
           capsize=2.5, error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.2)
    ax.set_ylabel(r"coherent infidelity ($\times10^{-4}$)")
    ax.set_title("(a) Ablation, X gate", loc="left")
    ax.grid(True, axis="y")
    ax.set_ylim(0, max(m + c[1] for m, c in zip(mus, cis)) * 1.18)
    if have3:
        R = json.load(open("results_E3_X.json"))
        ax = axs[1]
        ks = [1.25, 1.5, 2.0, 3.0, 4.0, 6.0]
        mu = [np.mean(R[f"kappa={k}"]) * 1e4 for k in ks]
        sd = [np.std(R[f"kappa={k}"]) * 1e4 for k in ks]
        xs = np.arange(len(ks))
        ax.errorbar(xs, mu, yerr=sd, color=COLORS["DRiFT"], marker="o", ms=4.5, lw=1.4, capsize=2.5)
        ax.axvline(ks.index(C.KAPPA), color="#7a7a7a", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k:g}" for k in ks])
        ax.set_xlabel(r"ambiguity factor $\kappa$")
        ax.set_ylabel(r"coherent infidelity ($\times10^{-4}$)")
        ax.set_title(r"(b) Sensitivity to $\kappa$", loc="left")
        ax.grid(True)
        ax = axs[2]
        Ts = [12.0, 16.0, 20.0, 24.0, 28.0]
        for m in ["GRAPE", "FF-nominal", "CVaR", "DRiFT"]:
            mu = [np.mean(R[f"T={T}|{m}"]) * 1e4 for T in Ts]
            ax.plot(Ts, mu, color=COLORS[m], marker=MARK[m], ls=LS[m], lw=1.4, ms=4.5, label=SHORT[m])
        ax.set_xticks(Ts)
        ax.set_xlabel("gate duration $T$ (ns)")
        ax.set_ylabel(r"coherent infidelity ($\times10^{-4}$)")
        ax.set_title("(c) Duration trade-off", loc="left")
        ax.grid(True, which="major")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False, fontsize=7.0,
                  columnspacing=0.8)
    save(fig, "fig6_ablation_sensitivity")


# ---------------------------------------------------------------- Fig 7: prediction & certificate
def fig7():
    R = load_e1("X")
    R2 = load_e1("SQiSW")
    if R is None:
        return
    envs = json.load(open("envs_X.json"))["in"]
    member = np.array([e["in_set"] for e in envs])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    ax = axs[0]
    allp, allm = [], []
    for m in ORDER:
        pred = (R[m]["pred"] - R[m]["I0"][:, None])[0]
        mc = (R[m]["coh_in"] - R[m]["I0"][:, None])[0]
        ax.loglog(pred, mc, ls="none", marker=MARK[m], ms=3.2, color=COLORS[m], alpha=0.7, mec="none")
        allp.append(pred)
        allm.append(mc)
    allp, allm = np.concatenate(allp), np.concatenate(allm)
    ok = (allp > 0) & (allm > 0)
    lim = [min(allp[ok].min(), allm[ok].min()) * 0.7, max(allp[ok].max(), allm[ok].max()) * 1.4]
    ax.plot(lim, lim, color="#0b0b0b", lw=0.9, ls="--")
    r = np.corrcoef(np.log(allp[ok]), np.log(allm[ok]))[0, 1]
    ax.text(0.04, 0.95, f"log-log Pearson r = {r:.3f}\n7 methods x 100 environments\n(markers as in Fig. 4)",
            transform=ax.transAxes, va="top", fontsize=7.2, linespacing=1.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("filter-function prediction (noise part)")
    ax.set_ylabel("exact Monte Carlo (noise part)")
    ax.set_title("(a) First-order prediction vs ground truth, X", loc="left")
    ax.grid(True, which="major")
    ax = axs[1]
    for g, RR, mk in [("X", R, "o"), ("SQiSW", R2, "s")]:
        if RR is None:
            continue
        mem = np.array([e["in_set"] for e in json.load(open(f"envs_{g}.json"))["in"]])
        cert = RR["DRiFT"]["cert"]
        ratio = RR["DRiFT"]["coh_in"] / cert[:, None]
        for flag, col, lab in [(True, COLORS["DRiFT"], r"spectrum in $\mathcal{S}$"), (False, "#8a8a8a", r"spectrum outside $\mathcal{S}$")]:
            vals = ratio[:, mem == flag].ravel()
            if len(vals) == 0:
                continue
            xs = np.full(len(vals), {"X": 0, "SQiSW": 1}[g]) + (-0.17 if flag else 0.17) + \
                np.random.default_rng(0).uniform(-0.09, 0.09, len(vals))
            ax.plot(xs, vals, ls="none", marker=mk, ms=2.6, color=col, alpha=0.45, mec="none",
                    label=f"{lab}" if g == "X" else None)
    ax.axhline(1.0, color="#0b0b0b", lw=0.9, ls="--")
    ax.text(1.42, 1.02, "certificate", fontsize=7.4, va="bottom", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["X", r"$\sqrt{\mathrm{iSWAP}}$"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"actual / certified infidelity")
    ax.set_title("(b) Certificate coverage, DRiFT pulses", loc="left")
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, markerscale=2)
    save(fig, "fig7_prediction_certificate")


# ---------------------------------------------------------------- Fig 8: drift
def fig8():
    gates = [g for g in ["X", "SQiSW"] if os.path.exists(f"results_E10_{g}.json")]
    if not gates:
        return
    fig, axs = plt.subplots(1, 1 + len(gates), figsize=(7.2, 2.9), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1.5] + [1] * len(gates)))
    g0 = gates[-1]
    R = json.load(open(f"results_E10_{g0}.json"))
    ax = axs[0]
    for m in ["GRAPE", "FF-nominal", "CVaR", "DRiFT"]:
        y = np.array(R[m][0]) * 1e3
        ax.plot(np.arange(len(y)), y, color=COLORS[m], ls=LS[m], lw=1.2)
    mem = np.array(R["member"][0])
    ymin, ymax = ax.get_ylim()
    for h in np.where(~mem)[0]:
        ax.axvspan(h - 0.5, h + 0.5, color="#e34948", alpha=0.07, lw=0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"total infidelity ($\times10^{-3}$)")
    ax.set_title(f"(a) 72 h drift, {GATE_TITLE[g0]}", loc="left", fontsize=8)
    ax.text(0.99, 0.97, "shaded: spectrum outside $\\mathcal{S}$", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.0, color="#b03a3a")
    ax.grid(True)
    for j, g in enumerate(gates):
        R = json.load(open(f"results_E10_{g}.json"))
        ax = axs[1 + j]
        for m in ORDER:
            A = np.sort(np.array(R[m]).ravel()) * 1e4
            ax.plot(A, np.arange(1, len(A) + 1) / len(A), color=COLORS[m], ls=LS[m], lw=1.3)
        ax.set_xlabel(r"hourly total infidelity ($\times10^{-4}$)")
        ax.set_ylabel("fraction of hours (CDF)")
        ax.set_title(f"({'bc'[j]}) {GATE_TITLE[g]}", loc="left", fontsize=8)
        ax.grid(True, which="major")
    method_legend(fig, ORDER, ncol=4, y=-0.17)
    save(fig, "fig8_drift_uptime")


# ---------------------------------------------------------------- Fig 9: convergence + cost
def fig9():
    if not (os.path.exists("hist_X.pkl") and os.path.exists("results_E6.json")):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    H = pickle.load(open("hist_X.pkl", "rb"))
    ax = axs[0]
    for m in ORDER[1:]:
        hs = np.array([h for h in H[m] if len(h)])
        if len(hs) == 0:
            continue
        med = np.median(hs, 0)
        ax.semilogy(np.arange(1, len(med) + 1), med, color=COLORS[m], ls=LS[m], lw=1.3, label=SHORT[m])
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (median of 10 seeds)")
    ax.set_title("(a) Convergence, X gate", loc="left")
    ax.grid(True, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=7.0)
    R = json.load(open("results_E6.json"))
    ax = axs[1]
    for g, mk in [("X", "-"), ("SQiSW", "--")]:
        for m in ["GRAPE", "Ensemble", "CVaR", "DRiFT"]:
            ks = sorted([k for k in R if k.startswith(g + "|") and k.endswith("|" + m)],
                        key=lambda k: int(k.split("|")[1][2:]))
            Ns = [int(k.split("|")[1][2:]) for k in ks]
            ax.loglog(Ns, [R[k] * 1e3 for k in ks], color=COLORS[m], marker=MARK[m], ls=mk, lw=1.2, ms=4,
                      label=f"{SHORT[m]} ({'1Q' if g == 'X' else '2Q'})")
    ax.set_xticks([20, 32, 40, 64, 80, 128])
    ax.set_xticklabels(["20", "32", "40", "64", "80", "128"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("PWC steps $N$")
    ax.set_ylabel("time per iteration (ms)")
    ax.set_title("(b) Cost per value-and-gradient", loc="left")
    ax.grid(True, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=6.6,
              columnspacing=0.6)
    save(fig, "fig9_convergence_cost")


if __name__ == "__main__":
    for f in [fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        try:
            f()
        except Exception as ex:   # keep producing the remaining figures
            print("figure failed:", f.__name__, repr(ex))
