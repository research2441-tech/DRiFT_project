"""Physical constants, device parameters and default hyper-parameters.

Units: time in ns, angular frequency in rad/ns (1 rad/ns = 159.15 MHz).
All power spectral densities are two-sided, with variance = int S(w) dw / (2 pi).
"""
import numpy as np

TWO_PI = 2.0 * np.pi

# ---------------- transmon device ----------------
ALPHA = -TWO_PI * 0.220          # anharmonicity (rad/ns)
T1 = 100.0e3                      # energy relaxation time (ns)
TPHI = 200.0e3                    # Markovian pure dephasing time (ns)
OMEGA_MAX = TWO_PI * 0.100        # drive amplitude bound per quadrature (rad/ns)
G_MAX = TWO_PI * 0.030            # tunable exchange coupling bound (rad/ns)
LEVELS = 3                        # transmon levels kept (qutrit)

# ---------------- gate timing ----------------
GATE_1Q = dict(T=20.0, N=40, K=12)   # duration (ns), PWC steps, sine-basis modes
GATE_2Q = dict(T=40.0, N=64, K=16)

# ---------------- frequency grid for filter functions ----------------
W_LO = 1.0e-3                     # below this the noise is treated as quasi-static (DC bin)
W_HI = 6.0                        # rad/ns
N_W = 40                          # log-spaced grid points in (W_LO, W_HI)
GL_NODES = 6                      # Gauss-Legendre nodes per PWC step (exact intra-step integral)

# ---------------- nominal noise model ("spectroscopy estimate") ----------------
# frequency (dephasing) noise, rad/ns
FREQ_1F_RMS = TWO_PI * 30.0e-6    # rms of the 1/f part above 1 ms (rad/ns)
FREQ_QS_RMS = TWO_PI * 60.0e-6    # quasi-static offset rms (slower than 1 ms)
TLS_B = TWO_PI * 0.40e-3          # telegraph amplitude of a near-resonant TLS (rad/ns)
TLS_RATE_NOM = 0.05               # nominal TLS switching rate (1/ns)
TLS_RATE_RANGE = (1.0e-4, 1.0)    # plausible switching-rate range (1/ns)
# relative amplitude noise (dimensionless)
AMP_1F_RMS = 1.5e-3
AMP_QS_RMS = 2.5e-3
# 1/f synthesis: OU processes with log-spaced rates
OU_RATE_RANGE = (1.0e-6, 10.0)
N_OU = 14

# ---------------- ambiguity set (DRiFT) ----------------
KAPPA = 2.0                       # multiplicative confidence factor on the nominal PSD
BUDGET_SLACK = 1.25               # factor on the measured total variance

# ---------------- optimisation ----------------
ITERS_1Q = 700
ITERS_2Q = 1800
LR = 0.02                         # 1Q learning rate
LR_2Q = 0.005                     # 2Q learning rate
CVAR_ALPHA = 0.2
CVAR_K = 32
ENSEMBLE_M = 16
BOUND_PENALTY = 50.0

# ---------------- evaluation ----------------
EVAL_TRAJ_1Q = 256
EVAL_TRAJ_2Q = 96
SUBSTEPS = 2
N_TEST_IN = 100
N_TEST_OOD = 60
FID_THRESHOLDS = (0.999, 0.9995)
