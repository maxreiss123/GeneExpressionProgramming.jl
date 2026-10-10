"""Judge GEP-SBP's accuracy-complexity fronts (gep_run.jl front=true, stop=none).

    python judge_front.py [--noise 0.05 0.1]

For each run in results/gep_front<tag>/, with judge.py's symbolic check:

* `final`: the model the full-budget run returns, the best training fit;
* `pick`: from the front, the simplest model whose normalised training RMSE is within 1 %
  of the noise floor (`floor_nrmse`, the true formula's own); the most accurate model if
  none is that close;
* `any`: whether any model on the front is the true formula, the credit PhySO's Pareto
  front gets in judge.py's `recovered_any`.

Alongside, the same run under the noise-floor stop (results/summary<tag>.csv). Writes
results/summary_front<tag>.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import sympy

import common
import judge

PICK_SLACK = 1.01


def r2_of(expr_text, pb, Xte, yte):
    try:
        expr = judge.gep_to_sympy(expr_text, pb)
        f = sympy.lambdify(pb.X_sympy_symbols, expr, "numpy")
        return common.r2(yte, np.broadcast_to(np.asarray(f(*Xte), dtype=float), yte.shape))
    except Exception:
        return float("nan")


def judge_run(path):
    r = json.load(open(path))
    pb = common.problem(r["equation"])
    front = r["front"]
    close = [m for m in front if m["nrmse"] <= r["floor_nrmse"] * PICK_SLACK]
    pick = close[0] if close else front[-1]
    ok = lambda e: judge.check(pb, e, judge.gep_to_sympy)[0]
    _, _, Xte, yte = common.make_data(r["equation"], r["seed"], r["noise"])
    return dict(
        equation=r["equation"], seed=r["seed"], noise=r["noise"],
        final=ok(r["expression"]), pick=ok(pick["expression"]),
        any=any(ok(m["expression"]) for m in front),
        front_size=len(front), pick_complexity=pick["complexity"],
        r2_final=r["r2_test"], r2_pick=r2_of(pick["expression"], pb, Xte, yte),
        wall_time=r["wall_time"], pick_expression=pick["expression"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.1])
    for noise in ap.parse_args().noise:
        tag = common.tag(noise)
        files = sorted(glob.glob(os.path.join(common.RESULTS_DIR, "gep_front" + tag, "*.json")))
        if not files:
            continue
        df = pd.DataFrame([judge_run(f) for f in files])
        base = pd.read_csv(os.path.join(common.RESULTS_DIR, "summary%s.csv" % tag))
        base = base[base.method == "GEP-SBP"][["equation", "seed", "recovered", "wall_time"]]
        base = base.rename(columns={"recovered": "floor_stop", "wall_time": "wall_floor_stop"})
        df = df.merge(base, on=["equation", "seed"], how="left")
        df.to_csv(os.path.join(common.RESULTS_DIR, "summary_front%s.csv" % tag), index=False)
        print("noise %g: %d runs" % (noise, len(df)))
        print(df.groupby("equation")[["floor_stop", "final", "pick", "any"]].sum())
        print(df[["floor_stop", "final", "pick", "any"]].sum().to_dict(),
              "median wall %.0f s (floor stop %.1f s)" % (df.wall_time.median(),
                                                          df.wall_floor_stop.median()))


if __name__ == "__main__":
    main()
