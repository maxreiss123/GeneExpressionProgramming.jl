"""The protocol of the GEP-SBP vs PySR comparison: the equations, the data, the units, the stop.

Both methods see the same data: `make_data(name, seed, noise)` samples the inputs
uniformly from the AI-Feynman ranges with a numpy generator seeded by (seed, equation
index) and evaluates the target formula. The train set is drawn per seed, the test set
once per equation (seed 0 of a separate stream), so every run of an equation is scored on
the same test points. The problems (formulas, ranges, units) are the AI-Feynman benchmark
of the physo package (`physo.benchmark.FeynmanDataset`), used here as a library.

With `noise` > 0, the training targets get noise * rms(y) * N(0, 1), the convention of
AI-Feynman and SRBench, from a stream of its own: the inputs are the same at every noise
level. The test targets stay noise-free, so the test R^2 measures how close a model is to
the true function.

`stop_nrmse` is where both methods stop a run: the normalised RMSE the true formula itself
has on the noisy training targets (the noise floor), and 1e-5 without noise. A model that
fits the training data as well as the truth does (up to 1e-4 relative, for rounding)
stops the search.

Units come from the benchmark's unit table ([m, s, kg, T, V]), which is dimensionally
homogeneous for all 100 bulk equations; `si_units` rewrites them as SI exponents
[kg, m, s, K, mol, A, cd], the order GeneExpressionProgramming.jl uses (V = kg m^2 s^-3 A^-1)
and the one pysr_run.py writes PySR's unit strings in. The basis change is invertible, so
both methods get the same unit information.
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")          # generated, not committed
RESULTS_DIR = os.path.join(HERE, "results")

N_TRAIN = 1000
N_TEST = 10000
SEEDS = list(range(1, 11))
NOISES = [0.0, 0.05, 0.1]
EXACT_NRMSE = 1e-5          # the stop without noise

# 15 equations in which the units carry information: the target has units, every input
# has units, the inputs span at least three independent SI base dimensions (rank of the
# input unit matrix >= 3), and the formula uses only + - * / powers sqrt exp, which both
# methods can express. Grouped by structure; within a group, by the order of the lectures.
EQUATIONS = [
    # products and quotients of the inputs (monomials)
    ("I.14.3", "monomial"),      # U = m g z
    ("I.34.8", "monomial"),      # omega = q v B / p
    ("I.43.16", "monomial"),     # v = mu_drift q Volt / d
    ("II.34.2a", "monomial"),    # I = q v / (2 pi r)
    ("III.21.20", "monomial"),   # j = -rho_c_0 q A_vec / m
    # monomials with squares and cubes
    ("I.12.2", "power law"),     # F = q1 q2 / (4 pi epsilon r^2)
    ("I.32.5", "power law"),     # P = q^2 a^2 / (6 pi epsilon c^3)
    ("I.38.12", "power law"),    # r = 4 pi epsilon hbar^2 / (m q^2)
    ("II.13.17", "power law"),   # B = 2 I / (4 pi epsilon c^2 r)
    ("III.15.14", "power law"),  # m = hbar^2 / (2 E d^2)
    # sums, differences and exponentials
    ("I.13.12", "sum / transcendental"),    # U = G m1 m2 (1/r2 - 1/r1)
    ("II.2.42", "sum / transcendental"),    # P = kappa (T2 - T1) A / d
    ("II.11.3", "sum / transcendental"),    # x = q Ef / (m (omega_0^2 - omega^2))
    ("II.21.32", "sum / transcendental"),   # V = q / (4 pi epsilon r (1 - v/c))
    ("III.14.14", "sum / transcendental"),  # I = I_0 (exp(q Volt / (kb T)) - 1)
]


def problem(name):
    return Feyn.FeynmanProblem(eq_name=name, original_var_names=True)


def si_units(u):
    m, s, kg, T, V = (float(x) for x in u)
    # + 0.0 turns -0.0 into 0.0: Julia hashes the two apart
    return [x + 0.0 for x in (kg + V, m + 2 * V, s - 3 * V, T, 0.0, -V, 0.0)]


def _sample(pb, n, rng):
    X = np.stack([rng.uniform(lo, hi, n) for lo, hi in zip(pb.X_lows, pb.X_highs)])
    return X, pb.target_function(X)


def make_data(name, seed, noise=0.0):
    """(X_train, y_train, X_test, y_test), X of shape (n_vars, n); y_train noisy."""
    pb = problem(name)
    idx = [e for e, _ in EQUATIONS].index(name)
    Xtr, ytr = _sample(pb, N_TRAIN, np.random.default_rng([seed, idx, 1]))
    Xte, yte = _sample(pb, N_TEST, np.random.default_rng([0, idx, 2]))
    if noise > 0:
        rng = np.random.default_rng([seed, idx, 3, round(noise * 1000)])
        ytr = ytr + noise * np.sqrt(np.mean(ytr ** 2)) * rng.standard_normal(len(ytr))
    return Xtr, ytr, Xte, yte


def stop_nrmse(name, seed, noise=0.0):
    """The normalised training RMSE at which a run stops (see the module docstring)."""
    if noise == 0:
        return EXACT_NRMSE
    pb = problem(name)
    Xtr, ytr, _, _ = make_data(name, seed, noise)
    clean = pb.target_function(Xtr)
    # normalised by the n-1 standard deviation, as pysr_run.py (numpy) and gep_run.jl
    # (Julia) do
    floor = np.sqrt(np.mean((ytr - clean) ** 2)) / np.std(ytr, ddof=1)
    # slack for rounding: the true formula itself must stop the run
    return float(floor * (1 + 1e-4))


def tag(noise):
    """Suffix of the data and result folders of a noise level; none without noise."""
    return "" if noise == 0 else "_noise%g" % noise


def r2(y, p):
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.all(np.isfinite(p)):
        return float("nan")
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
