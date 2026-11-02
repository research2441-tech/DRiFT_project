# Results summary

All numbers come from `results_summary.json` and the `table_*.csv` files produced by `make_tables.py`.
Coherent infidelity is the noise-driven part of the average gate infidelity: it is measured by exact
Monte Carlo and excludes the Markovian T1/T_phi floor, which is reported separately.

## E1 — main comparison (in-distribution environments)

| Gate | DRiFT mean (95% CI) | Best baseline (mean) | GRAPE | Change vs best baseline |
| --- | --- | --- | --- | --- |
| X (20 ns) | 5.45e-5 (4.59–6.38e-5) | Sampling-based robust, 6.20e-5 | 8.23e-5 | −12.0% |
| √X (20 ns) | 6.76e-5 (5.25–8.39e-5) | Quasi-static first-order, 1.07e-4 | 1.46e-4 | −36.6% |
| √iSWAP (40 ns) | 3.10e-4 (1.70–4.63e-4) | Risk-aware CVaR, 5.55e-4 | 1.25e-3 | −44.2% |

* Friedman tests across the 7 methods: p < 1e-50 for every gate.
* Holm-corrected Wilcoxon tests, DRiFT against each baseline: in-distribution p ≤ 4.0e-5 for √iSWAP and ≤ 4.0e-17 for the single-qubit gates; out-of-distribution p ≤ 1.6e-10 for all gates.
* Out-of-distribution means (DRiFT vs the best baseline on that split): X 1.54e-4 vs 1.63e-4; √X 2.05e-4 vs 2.95e-4; √iSWAP 8.15e-4 vs 1.69e-3.
* Tail risk (CVaR at 10%): DRiFT 1.15e-4 / 1.56e-4 / 5.59e-4 for X / √X / √iSWAP, the lowest of all methods for each gate.

## E2 — ablation
Components: A = spectral distributionally robust objective, B = exact Gauss–Legendre intra-step integral,
C = leakage-resolved filter function.

* **X gate, 10 seeds:** the base variant gives 7.66e-5. Adding A gives 6.71e-5, adding C gives 7.02e-5, and A and C together give 5.46e-5; the full model gives 5.45e-5.
* **Component B** has no measurable effect at 0.5 ns steps. It is kept because it removes discretisation error from the certificate.
* **√iSWAP, 3 seeds, one component left out at a time:** the full model gives 4.05e-4. Without A it gives 6.79e-4, without C 6.59e-4, and without B 4.74e-4.

## E3 — sensitivity (X gate, 5 seeds)

* **Ambiguity factor κ:** results stabilise for κ ≥ 2 (4.68e-5 at κ = 2, 4.46e-5 at κ = 6). The variance budget slack has little effect once it is ≥ 1.5.
* **Gate duration:** DRiFT stays between 4.4 and 5.9e-5 over 12–28 ns. GRAPE rises from 4.7e-5 to 1.34e-4, and nominal-PSD filter-function design rises to 9.1e-5.
* **Basis size:** DRiFT is lower than GRAPE for every number of basis modes K from 6 to 20.

## E4/E5 — robustness and generalisation

* **Noise amplitude scaled 0.5–6×:** DRiFT has the lowest infidelity at every scale except √iSWAP at 0.5×, where it is 1.54e-4 against 1.43e-4 for the sampling-based robust baseline. At 6× it gives 1.80e-3 (X) and 1.16e-2 (√iSWAP), against the best baselines' 1.95e-3 and 1.78e-2.
* **Anharmonicity mismatch of ±1–2% (limitation):** every numerically optimised pulse degrades more than calibrated DRAG, which stays at about 7e-5 on X.
  * X gate: DRiFT is the most tolerant optimised method at −2% and −1% (3.29e-4 and 1.17e-4). The sampling-based robust method is slightly better at +1% and +2% (1.09e-4 and 2.74e-4, against 1.13e-4 and 3.00e-4).
  * √iSWAP: DRiFT is the most sensitive method, at 1.46–1.54e-3 for ±1% and 4.38–5.06e-3 for ±2%, because its solutions exploit the second excited level.

## E6 — computational cost

Each DRiFT value-and-gradient evaluation costs the same as one nominal filter-function evaluation, to within 5%:

| Gate | DRiFT | Nominal filter function | Sampling-based robust | CVaR |
| --- | --- | --- | --- | --- |
| X (N = 40) | 8.5 ms | 8.3 ms | 10.5 ms (16 samples) | 8.7 ms |
| √iSWAP (N = 64) | 96 ms | 91 ms | 55 ms | 93 ms |

## E8 — prediction accuracy and certificate

* **Prediction accuracy:** the leakage-resolved filter-function prediction agrees with exact non-Gaussian Monte Carlo with a median relative error of 3–6% (log–log r = 0.997, X gate).
* **Certificate:** the observed coherent infidelity never exceeded the certified bound in any environment, for any method or gate. That holds for 100% of environments both inside and outside the ambiguity set. For DRiFT pulses, the median ratio of actual to certified error is 0.24–0.30.

## E10 — 72-hour drift (6 realisations, static pulses)

| Gate | DRiFT mean | DRiFT 95th pct | DRiFT worst hour | Best baseline values |
| --- | --- | --- | --- | --- |
| X | 1.56e-4 | 2.55e-4 | 3.53e-4 | 1.66e-4 / 2.68e-4 / 3.68e-4 |
| √iSWAP | 1.09e-3 | 1.62e-3 | 2.41e-3 | 1.06e-3 / 1.96e-3 / 2.70e-3 |

For √iSWAP, the quasi-static first-order baseline has the lowest mean error (1.06e-3); DRiFT is lowest at the 95th percentile and in the worst hour.

## Notes for the manuscript

* **Seed variability on √iSWAP:** three of five DRiFT seeds reach about 1.7e-4 and two stop near 5e-4, so restarting from several initial pulses is recommended. The drift run uses seed 0, which is one of the weaker seeds.
* **Two-qubit training schedule:** all √iSWAP methods use the same schedule, learning rate 0.005 for 1800 iterations. It was chosen after a convergence check showed that 900 iterations at 0.01 were not enough for the robust objectives, and it is applied uniformly to every method.
