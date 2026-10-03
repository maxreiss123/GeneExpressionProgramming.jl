"""Shared protocol for the Python baselines on ODEBench.

The corruption (after the ODEFormer repository's code, numpy RNG), the local-polynomial
derivative targets, the RK4 integration on the reference grid and the variance-weighted
R^2 live here, once. odebench_gep.jl implements the same protocol, with two differences:
the RNG behind the corruption (Julia's vs numpy's: a different draw, the same
distribution), and the integration step cap -- the baselines score their final model with
hmax = 10/1024 (three RK4 steps per reference interval), the Julia harness with 10/2048
(five). SINDy uses its own differentiation rather than `local_poly_derivatives`.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_systems():
    systems = []
    for s in json.load(open(os.path.join(HERE, "strogatz_extended.json"))):
        sol1, sol2 = s["solutions"][0][0], s["solutions"][0][1]
        if not (sol1["success"] and sol2["success"]):
            continue
        systems.append(dict(
            id=s["id"], dim=s["dim"], description=s["eq_description"],
            truth=s["substituted"][0],
            t=np.array(sol1["t"]), y=np.array(sol1["y"]),          # d x 512
            t2=np.array(sol2["t"]), y2=np.array(sol2["y"])))
    return systems


def corrupt(sys, sigma, rho, seed):
    """As the reference harness (odebench/solve_and_plot.py) does: drop int(rho*N) random
    indices, multiply by (1 + sigma*randn). The numpy seed, derived from (seed, system
    id), is this harness's own."""
    np.random.seed(seed * 100003 + sys["id"])
    t, y = sys["t"].copy(), sys["y"].T.copy()                      # N x d
    drop = np.random.choice(len(t), int(rho * len(t)), replace=False)
    t = np.delete(t, drop, axis=0)
    y = np.delete(y, drop, axis=0)
    y *= 1 + sigma * np.random.randn(*y.shape)
    return t, y


def local_poly_derivatives(t, y, window=9, degree=3):
    """Same estimator as the Julia harness: derivative from a local least-squares
    polynomial over `window` neighbouring samples. `y` is N x d; returns N x d."""
    n, d = y.shape
    dy = np.zeros_like(y)
    half = window // 2
    for i in range(n):
        lo = min(max(0, i - half), max(0, n - window))
        hi = min(lo + window, n)
        ts = t[lo:hi] - t[i]
        deg = min(degree, len(ts) - 1)
        V = np.vander(ts, deg + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V, y[lo:hi], rcond=None)
        dy[i] = coef[1]
    return dy


def integrate_on_grid(f, x0, tgrid, hmax=10.0 / 2048, xcap=1e8):
    """Fixed-step RK4 along the reference grid with the Julia harness's divergence guard.
    The default step cap is the Julia harness's too; the baselines pass coarser ones."""
    d = len(x0)
    out = np.full((len(tgrid), d), np.nan)
    x = np.array(x0, dtype=float)
    out[0] = x
    for gi in range(1, len(tgrid)):
        span = tgrid[gi] - tgrid[gi - 1]
        nsub = max(1, int(np.ceil(span / hmax)))
        h = span / nsub
        for _ in range(nsub):
            try:
                k1 = f(x)
                k2 = f(x + h / 2 * k1)
                k3 = f(x + h / 2 * k2)
                k4 = f(x + h * k3)
            except (FloatingPointError, ValueError, OverflowError):
                return out
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > xcap:
                return out
        out[gi] = x
    return out


def r2_vw(y_true, y_pred):
    """sklearn's variance-weighted R^2, as the reference evaluation computes it:
    1 - sum(SS_res) / sum(SS_tot) over components; -inf for a non-finite prediction."""
    if not np.all(np.isfinite(y_pred)):
        return -np.inf
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    if sstot <= 0:
        return 1.0 if ssres < 1e-12 else -np.inf
    return 1 - ssres / sstot


def score_model(f, sys):
    """Reconstruction and generalization R^2 for a fitted vector field `f(x) -> dx`, at
    the default step cap (unused by the current harnesses)."""
    pred = integrate_on_grid(f, sys["y"][:, 0], sys["t"])
    r2r = r2_vw(sys["y"].T, pred)
    pred2 = integrate_on_grid(f, sys["y2"][:, 0], sys["t2"])
    r2g = r2_vw(sys["y2"].T, pred2)
    return r2r, r2g


def accuracy(results, key):
    vals = [r[key] for r in results]
    return float(np.mean([v is not None and np.isfinite(v) and v > 0.9 for v in vals]))
