"""PDE-FIND baseline on the shared PDE-discovery protocol.

Sequentially thresholded ridge regression (STRidge, Rudy et al., Sci. Adv. 2017) over
the standard candidate library {u^p * d^q u/dx^q : p in 0..2, q in 1..max} plus
{1, u, u^2}, on the feature matrices `pde_common.make_condition` serves to every method.
Ridge weight and threshold are chosen by BIC on a validation split (the last 20 % of the
training rows); the selected model is then scored on clean test features against the
analytic right-hand side.

    python pdefind_baseline.py
"""

import itertools
import json
import os
import time

import numpy as np

from pde_common import (HERE, DATA, SIGMAS, make_condition, r2, R2_RECOVERED)


def build_library(names, X):
    """Columns u^p * (q-th spatial derivative), p in 0..2; q=0 means no derivative
    factor, so p=q=0 is the constant term."""
    u = X[:, 0]
    cols, labels = [], []
    for p, q in itertools.product(range(3), range(X.shape[1])):
        v = u ** p if p else np.ones(len(u))
        lab = {0: "1", 1: "u", 2: "u^2"}[p]
        if q > 0:
            v = v * X[:, q]
            lab = (lab + "*" if p else "") + names[q]
        if p == 0 and q == 0:
            lab = "1"
        cols.append(v)
        labels.append(lab)
    return np.stack(cols, axis=1), labels


def stridge(A, y, lam, tol, iters=10):
    """STRidge on column-normalized A; returns coefficients in original scale."""
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1
    An = A / norms
    w = np.linalg.lstsq(An.T @ An + lam * np.eye(A.shape[1]), An.T @ y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(w) < tol
        w[small] = 0
        big = ~small
        if not np.any(big):
            break
        w[big] = np.linalg.lstsq(An[:, big].T @ An[:, big] + lam * np.eye(big.sum()),
                                 An[:, big].T @ y, rcond=None)[0]
    return w / norms


def fit_condition(cond):
    X, y = cond["X_train"], cond["y_train"]
    A, labels = build_library(cond["names"], X)
    ntr = int(0.8 * len(y))
    At, yt, Av, yv = A[:ntr], y[:ntr], A[ntr:], y[ntr:]

    t0 = time.time()
    best = dict(bic=np.inf, w=None, tol=None)
    # tolerance grid spans the normalized-coefficient magnitudes
    for lam in (1e-6, 1e-4):
        for tol in np.logspace(-4, 0.5, 24):
            w = stridge(At, yt, lam, tol)
            k = int(np.sum(w != 0))
            resid = yv - Av @ w
            mse = float(np.mean(resid ** 2))
            if not np.isfinite(mse) or mse <= 0:
                continue
            bic = len(yv) * np.log(mse) + k * np.log(len(yv))
            if bic < best["bic"]:
                best = dict(bic=bic, w=w, tol=tol)
    t_fit = time.time() - t0

    w = best["w"]
    Atest, _ = build_library(cond["names"], cond["X_test"])
    pred = Atest @ w
    r2t = r2(cond["rhs_test"], pred)
    expr = " + ".join(f"{w[i]:+.4f}*{labels[i]}" for i in np.flatnonzero(w)) or "0"
    return dict(pde=cond["pde"], sigma=cond["sigma"], truth=cond["truth"],
                r2_test=float(r2t), recovered=bool(r2t > R2_RECOVERED),
                expression=expr, n_terms=int(np.sum(w != 0)), time_s=t_fit)


def main():
    results = []
    for name in ("heat", "burgers", "kdv", "ks"):
        for sigma in SIGMAS:
            c = make_condition(name, sigma)
            r = fit_condition(c)
            results.append(r)
            print(f"{r['pde']:8s} sigma={r['sigma']:<5g} R2={r['r2_test']:8.4f} "
                  f"terms={r['n_terms']}  {'OK ' if r['recovered'] else 'no '} "
                  f"{r['expression'][:90]}", flush=True)
    out = os.path.join(HERE, "results", "pdefind.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(method="PDE-FIND", results=results), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
