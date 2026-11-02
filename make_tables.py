"""Tables (CSV) and a machine-readable summary of every number quoted in the manuscript."""
import json
import os
import numpy as np
import pandas as pd
import config as C
import methods as MT
import stats_utils as ST
from make_figures import load_e1, ORDER

summary = {}


def fmt(x, s=None, scale=1e4):
    return f"{x * scale:.3f}" if s is None else f"{x * scale:.3f} ± {s * scale:.3f}"


def table_main():
    rows, stat_rows = [], []
    for g in ["X", "SX", "SQiSW"]:
        R = load_e1(g)
        if R is None:
            continue
        thr = C.FID_THRESHOLDS[0] if g == "SQiSW" else C.FID_THRESHOLDS[1]
        summary[g] = {}
        for m in ORDER:
            r = R[m]
            seed_means = r["coh_in"].mean(1)
            per_env = r["coh_in"].mean(0)
            row = {
                "Gate": g, "Method": MT.LABELS[m],
                "Mean coh. infid. (x1e-4)": fmt(seed_means.mean(), seed_means.std()),
                "CVaR0.1 (x1e-4)": fmt(np.mean([ST.cvar(x) for x in r["coh_in"]])),
                "Worst (x1e-4)": fmt(np.mean(r["coh_in"].max(1))),
                f"Yield F>={thr} (%)": f"{np.mean(1 - r['tot_in'] >= thr) * 100:.1f}",
                "OOD mean (x1e-4)": fmt(r["coh_ood"].mean()),
                "OOD CVaR0.1 (x1e-4)": fmt(np.mean([ST.cvar(x) for x in r["coh_ood"]])),
                "Leakage (x1e-6)": f"{r['L'].mean() * 1e6:.2f}",
                "Certificate (x1e-4)": fmt(r["cert"].mean()),
                "Train time (s)": f"{r['tt'].mean():.1f}",
            }
            rows.append(row)
            lo, hi = ST.bootstrap_ci(r["coh_in"])
            summary[g][m] = {"coh_mean": float(seed_means.mean()), "coh_std": float(seed_means.std()),
                             "ci95": [lo, hi], "cvar": float(np.mean([ST.cvar(x) for x in r["coh_in"]])),
                             "worst": float(np.mean(r["coh_in"].max(1))),
                             "yield": float(np.mean(1 - r["tot_in"] >= thr)),
                             "ood_mean": float(r["coh_ood"].mean()),
                             "ood_cvar": float(np.mean([ST.cvar(x) for x in r["coh_ood"]])),
                             "tot_mean": float(r["tot_in"].mean()), "cert": float(r["cert"].mean()),
                             "I0": float(r["I0"].mean()), "Imk": float(r["Imk"].mean()),
                             "leak": float(r["L"].mean()), "train_time": float(r["tt"].mean())}
        for split in ["coh_in", "coh_ood"]:
            per_env = {m: R[m][split].mean(0) for m in ORDER}
            t = ST.paired_tests(per_env)
            summary[g][f"tests_{split}"] = t
            for m, d in t["pairs"].items():
                stat_rows.append({"Gate": g, "Split": "in" if split == "coh_in" else "OOD",
                                  "Friedman chi2": f"{t['friedman_chi2']:.1f}", "Friedman p": f"{t['friedman_p']:.2e}",
                                  "Comparison": f"DRiFT vs {MT.LABELS[m]}", "Holm p": f"{d['p_holm']:.2e}",
                                  "Rank-biserial r": f"{d['rank_biserial']:.3f}",
                                  "Rel. change (%)": f"{(per_env['DRiFT'].mean() / per_env[m].mean() - 1) * 100:.1f}"})
    pd.DataFrame(rows).to_csv("table_main_results.csv", index=False)
    pd.DataFrame(stat_rows).to_csv("table_statistics.csv", index=False)


def table_ablation():
    rows = []
    for g in ["X", "SQiSW"]:
        f = f"results_E2_{g}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        names = sorted({k.split("__")[1] for k in z.files}, key=lambda n: ["Base", "+A", "+B", "+C", "+A+B",
                                                                         "+A+C", "+B+C", "Full"].index(n))
        summary.setdefault("ablation", {})[g] = {}
        for n in names:
            M = z[f"I_in__{n}"] - (z[f"Imk__{n}"] - z[f"I0__{n}"])[:, None]
            sm = M.mean(1)
            rows.append({"Gate": g, "Variant": n, "Mean coh. infid. (x1e-4)": fmt(sm.mean(), sm.std()),
                         "CVaR0.1 (x1e-4)": fmt(np.mean([ST.cvar(x) for x in M])),
                         "Certificate (x1e-4)": fmt(z[f"cert__{n}"].mean()), "Seeds": len(sm)})
            summary["ablation"][g][n] = {"mean": float(sm.mean()), "std": float(sm.std()),
                                         "cvar": float(np.mean([ST.cvar(x) for x in M])),
                                         "cert": float(z[f"cert__{n}"].mean())}
    if rows:
        pd.DataFrame(rows).to_csv("table_ablation.csv", index=False)


def table_robust():
    rows = []
    for g in ["X", "SQiSW"]:
        f = f"results_E45_{g}.json"
        if not os.path.exists(f):
            continue
        R = json.load(open(f))
        summary.setdefault("robust", {})[g] = {}
        for m in ORDER:
            d = {"Gate": g, "Method": MT.LABELS[m]}
            for s in [1.0, 2.0, 4.0, 6.0]:
                d[f"x{s:g} noise (x1e-4)"] = fmt(np.mean(R[f"scale={s}|{m}"]))
            for a in [0.98, 1.02]:
                d[f"alpha x{a} (x1e-4)"] = fmt(np.mean(R[f"alpha={a}|{m}"]))
            rows.append(d)
            summary["robust"][g][m] = {k: float(np.mean(v)) for k, v in R.items() if k.endswith("|" + m)}
    if rows:
        pd.DataFrame(rows).to_csv("table_robustness.csv", index=False)


def table_certificate():
    rows = []
    for g in ["X", "SX", "SQiSW"]:
        R = load_e1(g)
        if R is None or not os.path.exists(f"envs_{g}.json"):
            continue
        mem = np.array([e["in_set"] for e in json.load(open(f"envs_{g}.json"))["in"]])
        summary.setdefault("certificate", {})[g] = {}
        for m in ORDER:
            ratio = R[m]["coh_in"] / R[m]["cert"][:, None]
            pred = R[m]["pred"] - R[m]["I0"][:, None]
            mc = R[m]["coh_in"] - R[m]["I0"][:, None]
            rel = np.median(np.abs(pred - mc) / np.maximum(mc, 1e-12))
            d = {"Gate": g, "Method": MT.LABELS[m], "In-set envs (%)": f"{mem.mean() * 100:.0f}",
                 "Coverage in-set (%)": f"{np.mean(ratio[:, mem] <= 1) * 100:.1f}",
                 "Coverage all (%)": f"{np.mean(ratio <= 1) * 100:.1f}",
                 "Median actual/cert": f"{np.median(ratio[:, mem]):.3f}",
                 "Median |pred-MC|/MC (%)": f"{rel * 100:.1f}"}
            rows.append(d)
            summary["certificate"][g][m] = {"in_set_frac": float(mem.mean()),
                                            "coverage_in": float(np.mean(ratio[:, mem] <= 1)),
                                            "coverage_all": float(np.mean(ratio <= 1)),
                                            "median_ratio": float(np.median(ratio[:, mem])),
                                            "median_pred_err": float(rel)}
    if rows:
        pd.DataFrame(rows).to_csv("table_certificate.csv", index=False)


def table_cost_drift():
    if os.path.exists("results_E6.json"):
        R = json.load(open("results_E6.json"))
        pd.DataFrame([{"Setting": k, "ms per value+grad": f"{v * 1e3:.2f}"} for k, v in R.items()]) \
            .to_csv("table_cost.csv", index=False)
        summary["cost"] = R
    rows = []
    for g in ["X", "SQiSW"]:
        f = f"results_E10_{g}.json"
        if not os.path.exists(f):
            continue
        R = json.load(open(f))
        mem = np.array(R["member"])
        summary.setdefault("drift", {})[g] = {"member_frac": float(mem.mean())}
        for m in ORDER:
            A = np.array(R[m])
            cert = R[m + "|cert"]
            rows.append({"Gate": g, "Method": MT.LABELS[m], "Mean (x1e-4)": fmt(A.mean()),
                         "Median (x1e-4)": fmt(np.median(A)), "95th pct (x1e-4)": fmt(np.quantile(A, 0.95)),
                         "Worst hour (x1e-4)": fmt(A.max()),
                         "Hours in set & below cert. (%)": f"{np.mean(A[mem] <= cert) * 100:.1f}"})
            summary["drift"][g][m] = {"mean": float(A.mean()), "median": float(np.median(A)),
                                      "p95": float(np.quantile(A, 0.95)), "worst": float(A.max()),
                                      "cert_total": float(cert), "cert_hold_in_set": float(np.mean(A[mem] <= cert))}
    if rows:
        pd.DataFrame(rows).to_csv("table_drift.csv", index=False)


def table_sensitivity():
    if not os.path.exists("results_E3_X.json"):
        return
    R = json.load(open("results_E3_X.json"))
    pd.DataFrame([{"Setting": k, "Mean coh. infid. (x1e-4)": fmt(np.mean(v), np.std(v))} for k, v in R.items()]) \
        .to_csv("table_sensitivity.csv", index=False)
    summary["sensitivity"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in R.items()}


if __name__ == "__main__":
    for f in [table_main, table_ablation, table_robust, table_certificate, table_cost_drift, table_sensitivity]:
        f()
    json.dump(summary, open("results_summary.json", "w"), indent=1, default=float)
    print("tables written")
