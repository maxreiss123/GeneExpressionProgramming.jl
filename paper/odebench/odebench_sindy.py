"""SINDy on ODEBench, under the shared protocol (odebench_common.py).

PySINDy with a degree-3 polynomial plus one-frequency Fourier library and STLSQ, using
its own differentiation (plain or Savitzky-Golay-smoothed finite differences), fitted on
a 2 x 2 grid of threshold and smoothing window per system. Selection as in the GEP
harness: every fit is integrated from the clean initial state, the best reconstruction
R^2 against the clean trajectory wins, and the winner is rescored at a finer step and
once on the generalization trajectory. Four systems run in parallel.

    python odebench_sindy.py [--sigma 0.05] [--rho 0.5] [--seed 1]
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysindy as ps

from odebench_common import (HERE, load_systems, corrupt, score_model, accuracy,
                             integrate_on_grid, r2_vw)

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# per system: STLSQ threshold x differentiation (unsmoothed or window 15 on clean data,
# window 15 or 31 with noise), in the spirit of the per-baseline hyperparameter search of
# the reference paper. Candidates are ranked at the COARSE step cap (one RK4 step per
# reference interval) and only the winner is rescored at FINE (three): integration in
# Python dominates the cost.
THRESHOLDS = (0.05, 0.2)
COARSE = 10.0 / 256
FINE = 10.0 / 1024


def fit_one(sys, sigma, rho, seed):
    t, y = corrupt(sys, sigma, rho, seed)
    windows = (None, 15) if sigma == 0 else (15, 31)
    best = dict(r2r=-np.inf, r2g=-np.inf, expr=None, model=None)
    t0 = time.time()
    for thr in THRESHOLDS:
        for win in windows:
            if win is None:
                diff = ps.FiniteDifference()
            else:
                diff = ps.SmoothedFiniteDifference(
                    smoother_kws={"window_length": min(win, len(t) - 1 - (len(t) % 2 == 0)),
                                  "polyorder": 3})
            lib = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary(n_frequencies=1)
            model = ps.SINDy(feature_library=lib,
                             optimizer=ps.STLSQ(threshold=thr),
                             differentiation_method=diff)
            try:
                model.fit(y, t=t)
            except Exception:
                continue

            def f(x, model=model):
                return model.predict(x.reshape(1, -1))[0]

            pred = integrate_on_grid(f, sys["y"][:, 0], sys["t"], hmax=COARSE)
            r2r = r2_vw(sys["y"].T, pred)
            if np.isfinite(r2r) and r2r > best["r2r"]:
                best = dict(r2r=r2r, r2g=-np.inf,
                            expr="; ".join(model.equations(precision=4)), model=model)
    if best["model"] is not None:
        def f(x, model=best["model"]):
            return model.predict(x.reshape(1, -1))[0]
        pred = integrate_on_grid(f, sys["y"][:, 0], sys["t"], hmax=FINE)
        best["r2r"] = r2_vw(sys["y"].T, pred)
        pred2 = integrate_on_grid(f, sys["y2"][:, 0], sys["t2"], hmax=FINE)
        best["r2g"] = r2_vw(sys["y2"].T, pred2)
    return dict(id=sys["id"], dim=sys["dim"], description=sys["description"],
                r2_reconstruction=None if best["r2r"] == -np.inf else best["r2r"],
                r2_generalization=None if best["r2g"] == -np.inf else best["r2g"],
                expression=best["expr"], time_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    systems = load_systems()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fit_one, systems,
                                [a.sigma] * len(systems), [a.rho] * len(systems),
                                [a.seed] * len(systems)))
    for r in results:
        rr = r["r2_reconstruction"]
        rg = r["r2_generalization"]
        print(f"id {r['id']:2d}  d={r['dim']}  "
              f"r2_rec {rr if rr is None else round(max(rr, -9.9999), 4)!s:>8}  "
              f"r2_gen {rg if rg is None else round(max(rg, -9.9999), 4)!s:>8}  "
              f"{r['time_s']:5.1f}s", flush=True)

    acc_r = accuracy(results, "r2_reconstruction")
    acc_g = accuracy(results, "r2_generalization")
    print(f"\nsigma={a.sigma:.2f} rho={a.rho:.2f} seed={a.seed} : "
          f"accuracy (R2>0.9)  reconstruction {acc_r:.3f}   generalization {acc_g:.3f}")

    out = os.path.join(HERE, "results",
                       f"sindy_sigma{a.sigma:g}_rho{a.rho:g}_seed{a.seed}.json")
    json.dump(dict(method="SINDy", sigma=a.sigma, rho=a.rho, seed=a.seed,
                   results=results), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
