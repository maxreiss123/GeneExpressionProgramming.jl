"""PySR on ODEBench, under the shared protocol (odebench_common.py).

Componentwise symbolic regression on the same local-polynomial derivative targets and
with the same operators as the GEP harness, then candidate selection by reconstruction
R^2 as there. PySR is a strong classical baseline in the reference paper.

PySR's Julia backend is heavyweight per call, so the budget per component is capped
(15 iterations, 20 s timeout, serial) to keep a 63-system condition tractable on CPU;
the wall clock is reported as measured, not normalised.

    python odebench_pysr.py [--sigma 0.05] [--rho 0.0] [--seed 1]
"""

import argparse
import json
import os
import time
import warnings

import numpy as np

from odebench_common import (HERE, load_systems, corrupt, local_poly_derivatives,
                             integrate_on_grid, r2_vw, accuracy)

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from pysr import PySRRegressor   # noqa: E402  (after warnings config; slow import)


def make_regressor(seed):
    return PySRRegressor(
        niterations=15,
        populations=15,
        population_size=27,
        maxsize=25,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "square"],
        timeout_in_seconds=20,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        temp_equation_file=True,
        verbosity=0,
        progress=False,
    )


def fit_one(sys, sigma, rho, seed):
    t, y = corrupt(sys, sigma, rho, seed)
    window = 15 if sigma > 0 else 7
    dy = local_poly_derivatives(t, y, window=window, degree=3)

    t0 = time.time()
    # the two lowest-loss equations per component, as the GEP harness keeps its top two
    cand_fns, cand_exprs = [], []
    for k in range(sys["dim"]):
        reg = make_regressor(seed + 100 * k)
        try:
            reg.fit(y, dy[:, k])
            eqs = reg.equations_.sort_values("loss").head(2)
            fns = [reg.get_best() for _ in range(1)]  # unused; the lambdas below are used
            lams = []
            exprs = []
            for _, row in eqs.iterrows():
                lam = row["lambda_format"]
                lams.append(lam)
                exprs.append(str(row["equation"]))
            cand_fns.append(lams)
            cand_exprs.append(exprs)
        except Exception:
            cand_fns.append([])
            cand_exprs.append([None])
    t_fit = time.time() - t0

    if any(len(f) == 0 for f in cand_fns):
        return dict(id=sys["id"], dim=sys["dim"], description=sys["description"],
                    r2_reconstruction=None, r2_generalization=None,
                    expression=None, time_s=time.time() - t0)

    combos = [[0] * sys["dim"]]
    for k in range(sys["dim"]):
        if len(cand_fns[k]) > 1:
            c = list(combos[0])
            c[k] = 1
            combos.append(c)

    best = dict(r2r=-np.inf, r2g=-np.inf, expr=None)
    for combo in combos:
        lams = [cand_fns[k][combo[k]] for k in range(sys["dim"])]

        def f(x, lams=lams):
            return np.array([float(np.asarray(l(x.reshape(1, -1))).ravel()[0])
                             for l in lams])

        pred = integrate_on_grid(f, sys["y"][:, 0], sys["t"], hmax=10.0 / 256)
        r2r = r2_vw(sys["y"].T, pred)
        if np.isfinite(r2r) and r2r > best["r2r"]:
            pred = integrate_on_grid(f, sys["y"][:, 0], sys["t"], hmax=10.0 / 1024)
            pred2 = integrate_on_grid(f, sys["y2"][:, 0], sys["t2"], hmax=10.0 / 1024)
            best = dict(r2r=r2_vw(sys["y"].T, pred), r2g=r2_vw(sys["y2"].T, pred2),
                        expr="; ".join(cand_exprs[k][combo[k]] for k in range(sys["dim"])))
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

    out = os.path.join(HERE, "results",
                       f"pysr_sigma{a.sigma:g}_rho{a.rho:g}_seed{a.seed}.json")
    partial = out + ".partial"

    # resume: the Julia backend can be killed under memory pressure, so every completed
    # system is checkpointed and reloaded here -- a kill costs at most the system in flight
    done = {}
    if os.path.exists(partial):
        for r in json.load(open(partial))["results"]:
            done[r["id"]] = r
        print(f"resuming: {len(done)} systems already checkpointed", flush=True)

    results = []
    for sys in load_systems():
        if sys["id"] in done:
            r = done[sys["id"]]
        else:
            r = fit_one(sys, a.sigma, a.rho, a.seed)
            results_so_far = results + [r]
            json.dump(dict(method="PySR", sigma=a.sigma, rho=a.rho, seed=a.seed,
                           results=results_so_far), open(partial, "w"), indent=1)
        results.append(r)
        rr, rg = r["r2_reconstruction"], r["r2_generalization"]
        print(f"id {r['id']:2d}  d={r['dim']}  "
              f"r2_rec {rr if rr is None else round(max(rr, -9.9999), 4)!s:>8}  "
              f"r2_gen {rg if rg is None else round(max(rg, -9.9999), 4)!s:>8}  "
              f"{r['time_s']:6.1f}s", flush=True)

    print(f"\nsigma={a.sigma:.2f} rho={a.rho:.2f} seed={a.seed} : "
          f"accuracy (R2>0.9)  reconstruction {accuracy(results, 'r2_reconstruction'):.3f}"
          f"   generalization {accuracy(results, 'r2_generalization'):.3f}")

    json.dump(dict(method="PySR", sigma=a.sigma, rho=a.rho, seed=a.seed,
                   results=results), open(out, "w"), indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    print("wrote", out)


if __name__ == "__main__":
    main()
