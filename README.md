# DRiFT — distributionally robust filter-function pulse synthesis

Code, data and figures for the manuscript
**"Distributionally robust pulse synthesis for transmon gates under ambiguous, drifting noise spectra"**
(target journal: *Quantum Science and Technology*).

DRiFT designs control pulses for superconducting transmon gates that stay accurate when the noise
spectrum is only known to lie inside a band (for example, from noise spectroscopy with error bars and a
Ramsey variance bound) and keeps drifting. Instead of averaging over sampled spectra, DRiFT minimises the
**worst-case first-order infidelity over a spectral ambiguity set**. For a box-and-budget set this worst case
has a closed form (a fractional knapsack over the filter function), so the robust objective costs the same
as a single nominal filter-function evaluation and returns a **fidelity certificate** for every spectrum in
the set.

## What is in this folder (flat layout, no sub-folders)

| File | Purpose |
| --- | --- |
| `config.py` | Device, noise, ambiguity-set and optimisation parameters (units: ns, rad/ns) |
| `models.py` | Transmon qutrit and two-qutrit (tunable exchange) Hamiltonians, targets, noise operators |
| `spectra.py` | Noise components (OU, telegraph, quasi-static), drift prior, ambiguity set, knapsack worst case, noise synthesis |
| `core.py` | Differentiable propagation, average gate fidelity, leakage, leakage-resolved filter functions with exact Gauss–Legendre intra-step integration |
| `methods.py` | DRiFT and the six baselines (calibrated DRAG, GRAPE, sampling-based robust, quasi-static first-order, nominal-PSD filter function, risk-aware CVaR); Adam optimiser |
| `noise_sim.py` | Ground truth: exact Monte Carlo with non-Gaussian, time-correlated noise plus Lindblad T1/T_phi |
| `run_experiments.py` | Experiments E1–E10 (main comparison, ablation, sensitivity, robustness, generalisation, cost, drift) |
| `stats_utils.py` | CVaR, Friedman and Wilcoxon tests, Holm correction, rank-biserial effect size, two-level bootstrap |
| `fig_architecture.py`, `make_figures.py`, `make_tables.py` | Figure 1, Figures 2–9, CSV tables and `results_summary.json` |
| `test_core.py` | Checks: intra-step quadrature vs Van Loan exponential, knapsack vs linear program, filter-function prediction vs Monte Carlo |
| `results_*.npz/json`, `pulses_*.pkl`, `hist_*.pkl`, `envs_*.json` | Raw results produced by the runs reported in the paper |
| `fig*.png/pdf`, `table_*.csv`, `results_summary.json` | Figures, tables and every number quoted in the manuscript |
| `log_*.txt` | Console logs of the reported runs |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q test_core.py          # about 1 min
python run_experiments.py e1 --gate X     # main comparison for the X gate (about 15 min, 2-core CPU)
python make_figures.py && python make_tables.py
```

`run_all.sh` reproduces everything (about 6 h on a 2-core CPU, no GPU needed).

## Model summary

* Transmons kept as three-level systems (leakage included), anharmonicity −2π·220 MHz, T1 = 100 µs, T_phi = 200 µs.
* Single-qubit X and √X gates (20 ns, two drive quadratures) and a two-qubit √iSWAP gate (40 ns, four local drives plus a tunable exchange coupler).
* Noise: 1/f frequency and amplitude noise (sums of Ornstein–Uhlenbeck processes), random-telegraph two-level-system noise, quasi-static offsets; spectra drift between calibrations.
* Controls: sine basis (zero at both ends, band-limited), amplitude bounds 2π·100 MHz (drives) and 2π·30 MHz (coupler).
* Evaluation: exact time-domain Monte Carlo (no perturbative approximation) on 100 in-distribution and 60 out-of-distribution environments (60 and 40 for √iSWAP), common random numbers across methods, 10 seeds (1Q) and 5 seeds (2Q).

## Results

Coherent infidelity is the noise-driven part of the gate error, measured with exact Monte Carlo. Mean values on the in-distribution environments:

| Gate | DRiFT | Best baseline | Change |
| --- | --- | --- | --- |
| X (20 ns) | 5.45e-5 | 6.20e-5 (sampling-based robust) | −12% |
| √X (20 ns) | 6.76e-5 | 1.07e-4 (quasi-static first-order) | −37% |
| √iSWAP (40 ns) | 3.10e-4 | 5.55e-4 (risk-aware CVaR) | −44% |

Each comparison has Holm-corrected Wilcoxon p ≤ 4e-5. The certified bound held in every evaluated environment. Each robust iteration costs the same as one nominal filter-function evaluation, to within 5%.

Limitations, reported in full in `RESULTS.md`:
* Optimised pulses are more sensitive to anharmonicity miscalibration than analytic DRAG.
* DRiFT's two-qubit solutions vary across seeds.

## Citation

See `CITATION.cff`. The Zenodo DOI of this release will be added after deposition.

## License

MIT (see `LICENSE`).
