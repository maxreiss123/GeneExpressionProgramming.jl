"""Audit of judge.py's symbolic check on every final model; writes
results/audit_symbolic<tag>.csv and prints the runs where the tests disagree.

    python audit_symbolic.py [--noise 0 0.05 0.1]

Each test is against numbers rather than against sympy alone:

* reading: the test R^2 of the model as judge.py reads it (the sympy tree evaluated node
  by node with numpy) against the R^2 the method reported (`r2_test`: Julia for GEP-SBP,
  PhySO's own evaluation for PhySO); for GEP-SBP also the printed model evaluated with
  numpy directly. A model read wrongly shows here.
* canonical form: the form judge.check rounds (judge.canonical) against the reading on
  the test points (`canonical_dev`, relative).
* verdict: judge.check (`symbolic`), the same with PhySO's pi-fraction step
  (`symbolic_pi`), and the criterion checked numerically (`numeric`): on the 10 000
  noise-free test points the ratio of model to formula is constant to 1 % (99th
  percentile of the deviation from the median ratio) and between 0.005 and 200 (the
  check's ratio, formula over model, must not round to 0), or their difference is
  constant to 1 % of the formula's standard deviation.
"""
import argparse
import functools
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import sympy

import common
import judge

TOL = 1e-2
_NP = {sympy.exp: np.exp, sympy.log: np.log, sympy.sin: np.sin, sympy.cos: np.cos,
       sympy.Abs: np.abs}


def tree_eval(e, env):
    """`e` evaluated node by node with numpy (no printer, no evaluation by sympy)."""
    if e.is_Symbol:
        return env[e]
    if e.is_Number or e.is_NumberSymbol:
        return float(e)
    a = [tree_eval(x, env) for x in e.args]
    if e.is_Add:
        return functools.reduce(np.add, a)
    if e.is_Mul:
        return functools.reduce(np.multiply, a)
    if e.is_Pow:
        return np.power(a[0], a[1])
    if e.func in _NP:
        return _NP[e.func](a[0])
    raise ValueError("no numpy rule for %s" % e.func)


def gep_numpy(text, X):
    s = text.replace("e^(", "np.exp(").replace("√(", "np.sqrt(").replace("ln(", "np.log(")
    s = s.replace("sin(", "np.sin(").replace("cos(", "np.cos(").replace("²", "**2")
    env = {"np": np, **{"x%d" % (i + 1): X[i] for i in range(len(X))}}
    return eval(s, {"__builtins__": {}}, env)


def values(f, shape):
    try:
        with np.errstate(all="ignore"):
            p = np.broadcast_to(np.asarray(f(), dtype=float), shape)
        return p if np.all(np.isfinite(p)) else None
    except Exception:
        return None


def criterion(p, y):
    """(ratio spread, |median ratio|, difference spread)"""
    r = p / y
    med = np.median(r)
    sr = np.quantile(np.abs(r - med), 0.99) / abs(med) if med != 0 else np.inf
    d = p - y
    sd = np.quantile(np.abs(d - np.median(d)), 0.99) / np.std(y)
    return float(sr), float(abs(med)), float(sd)


def audit(path):
    r = json.load(open(path))
    pb = common.problem(r["equation"])
    _, _, Xte, yte = common.make_data(r["equation"], r["seed"], r.get("noise", 0.0))
    env = dict(zip(pb.X_sympy_symbols, Xte))
    row = dict(method=r["method"], equation=r["equation"], seed=r["seed"],
               r2_reported=r["r2_test"], expression=r["expression"])
    physo = r["method"] == "PhySO"
    text = r["expression_raw"] if physo else r["expression"]
    if text is None:
        return dict(row, symbolic=False, symbolic_pi=False, numeric=False, note="no model")
    to_sympy = judge.physo_to_sympy if physo else judge.gep_to_sympy
    expr = to_sympy(text, pb)
    p = values(lambda: tree_eval(expr, env), yte.shape)
    row["r2_reading"] = np.nan if p is None else common.r2(yte, p)
    if not physo:
        q = values(lambda: gep_numpy(text, Xte), yte.shape)
        row["r2_string"] = np.nan if q is None else common.r2(yte, q)
    try:
        canon = judge.canonical(expr)
        c = values(lambda: tree_eval(canon, env), yte.shape)
        if c is None or p is None:
            dev = np.inf
        else:
            scale = np.max(np.abs(p))
            dev = float(np.max(np.abs(c - p)) / scale) if scale > 0 else float(np.max(np.abs(c)))
        row.update(canonical_dev=dev, canonical_form=str(canon))
    except Exception as ex:
        row.update(canonical_dev=np.inf, canonical_form="error: %s" % ex)
    row["symbolic"] = judge.check(pb, text, to_sympy)[0]
    row["symbolic_pi"] = judge.check(pb, text, to_sympy, handle_trigo=True)[0]
    if p is None:
        row.update(numeric=False, note="not finite on the test points")
    else:
        sr, med, sd = criterion(p, yte)
        row.update(spread_ratio=sr, ratio=med, spread_diff=sd,
                   numeric=bool((sr <= TOL and 0.005 <= med <= 200) or sd <= TOL), note="")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=common.NOISES)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    for noise in ap.parse_args().noise:
        tag = common.tag(noise)
        files = sorted(glob.glob(os.path.join(common.RESULTS_DIR, "gep" + tag, "*.json")) +
                       glob.glob(os.path.join(common.RESULTS_DIR, "physo" + tag, "*.json")))
        with mp.get_context("fork").Pool(4) as pool:
            df = pd.DataFrame(pool.map(audit, files, chunksize=1))
        df.insert(3, "noise", noise)
        df.to_csv(os.path.join(common.RESULTS_DIR, "audit_symbolic%s.csv" % tag), index=False)
        model = df.note != "no model"
        off = (df.r2_reading - df.r2_reported).abs() > 1e-6
        if "r2_string" in df:
            off |= (df.r2_string - df.r2_reported).abs() > 1e-6
        print("=== noise %g" % noise)
        print(df.groupby("method")[["symbolic", "symbolic_pi", "numeric"]].sum())
        print("readings off the reported R^2 by more than 1e-6: %d of %d"
              % ((off & model).sum(), model.sum()))
        print("largest canonical-form deviation: %.1e" % df[model].canonical_dev.max())
        dis = df[(df.symbolic != df.numeric) | (df.symbolic != df.symbolic_pi)]
        print(dis[["method", "equation", "seed", "r2_reported", "symbolic", "symbolic_pi",
                   "numeric", "spread_ratio", "spread_diff", "canonical_form"]].to_string())


if __name__ == "__main__":
    main()
