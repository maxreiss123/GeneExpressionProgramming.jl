"""Audit of judge.py's symbolic check on every final model; writes
results/audit_symbolic<tag>.csv and prints the runs where the tests disagree.

    python audit_symbolic.py [--noise 0 0.05 0.1]

Each test is against numbers rather than against sympy alone:

* reading: the test R^2 of the model as judge.py reads it (the sympy tree evaluated node
  by node with numpy) and of the printed model evaluated with numpy directly, against the
  R^2 the method reported (`r2_test`: Julia for GEP-SBP, PySR's own `predict` for PySR).
  A model read wrongly shows here.
* canonical form: the form judge.check rounds (judge.canonical) against the reading on
  the test points (`canonical_dev`, relative).
* verdict, on the 10 000 noise-free test points:
  - `symbolic`: judge.check, the reported criterion;
  - `symbolic_pi`: the same with the library's pi-fraction step;
  - `numeric`: the symbolic criterion checked numerically: the ratio of model to formula
    is constant to 1 % (99th percentile of the deviation from the median ratio) and
    between 0.005 and 200 (the check's ratio, formula over model, must not round to 0),
    or their difference is constant to 1 % of the formula's standard deviation. Like the
    symbolic criterion it accepts the formula times, or plus, a constant;
  - `equal`: the model is the formula, its constants included: the 99th percentile of
    |model / formula - 1| is at most 1 %, or that of |model - formula| at most 1 % of the
    formula's standard deviation (for formulas that cross zero).
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


def pysr_numpy(text, X):
    env = {"x%d" % (i + 1): X[i] for i in range(len(X))}
    env.update(square=np.square, cube=lambda a: a ** 3, sqrt=np.sqrt, exp=np.exp,
               log=np.log, sin=np.sin, cos=np.cos, Inf=np.inf, NaN=np.nan)
    return eval(text, {"__builtins__": {}}, env)


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


def equality(p, y):
    """(relative deviation, deviation relative to std): the 99th percentiles of
    |p / y - 1| and |p - y| / std(y)"""
    return (float(np.quantile(np.abs(p / y - 1), 0.99)),
            float(np.quantile(np.abs(p - y), 0.99) / np.std(y)))


def audit(path):
    r = json.load(open(path))
    pysr = r["method"] == "PySR"
    pb = common.problem(r["equation"])
    _, _, Xte, yte = common.make_data(r["equation"], r["seed"], r.get("noise", 0.0))
    env = dict(zip(pb.X_sympy_symbols, Xte))
    text = r["expression"]
    row = dict(method=r["method"], equation=r["equation"], seed=r["seed"],
               r2_reported=r["r2_test"], expression=text)
    if text is None:
        return dict(row, symbolic=False, symbolic_pi=False, numeric=False, equal=False,
                    note="no model")
    to_sympy = judge.pysr_to_sympy if pysr else judge.gep_to_sympy
    expr = to_sympy(text, pb)
    p = values(lambda: tree_eval(expr, env), yte.shape)
    row["r2_reading"] = np.nan if p is None else common.r2(yte, p)
    q = values(lambda: (pysr_numpy if pysr else gep_numpy)(text, Xte), yte.shape)
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
        row.update(numeric=False, equal=False, note="not finite on the test points")
    else:
        sr, med, sd = criterion(p, yte)
        er, ed = equality(p, yte)
        row.update(spread_ratio=sr, ratio=med, spread_diff=sd, rel_dev=er, abs_dev=ed,
                   numeric=bool((sr <= TOL and 0.005 <= med <= 200) or sd <= TOL),
                   equal=bool(er <= TOL or ed <= TOL), note="")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=common.NOISES)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    for noise in a.noise:
        tag = common.tag(noise)
        with mp.get_context("fork").Pool(4) as pool:
            df = pd.DataFrame(pool.map(audit, judge.load(noise), chunksize=1))
        df.insert(3, "noise", noise)
        df.to_csv(os.path.join(common.RESULTS_DIR, "audit_symbolic%s.csv" % tag), index=False)
        model = df.note != "no model"
        off = ((df.r2_reading - df.r2_reported).abs() > 1e-6) | \
              ((df.r2_string - df.r2_reported).abs() > 1e-6)
        print("=== noise %g" % noise)
        print(df.groupby("method")[["symbolic", "symbolic_pi", "numeric", "equal"]].sum())
        print("readings off the reported R^2 by more than 1e-6: %d of %d"
              % ((off & model).sum(), model.sum()))
        print("largest canonical-form deviation: %.1e" % df[model].canonical_dev.max())
        dis = df[(df.symbolic != df.numeric) | (df.symbolic != df.symbolic_pi) |
                 (df.symbolic != df.equal)]
        print(dis[["method", "equation", "seed", "r2_reported", "symbolic", "symbolic_pi",
                   "numeric", "equal", "spread_ratio", "spread_diff", "rel_dev", "abs_dev",
                   "canonical_form"]].to_string())


if __name__ == "__main__":
    main()
