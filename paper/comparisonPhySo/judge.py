"""Judge the runs of both methods with one symbolic criterion; write results/summary<tag>.csv.

    python judge.py [--noise 0]

Reads results/gep<tag>/ and results/physo<tag>/ (tag as common.tag: none without noise).

A model counts as a symbolic recovery if PhySO's `FeynmanProblem.compare_expression`
accepts it: after rounding its floats to two decimals (as SRBench does), the difference
to the true formula or the ratio of the two simplifies to a constant. The same check runs
on the final model of both methods (`recovered`), and on every model a run returns
(`recovered_any`): PhySO's Pareto front, the GEP hall of fame (3 models). A check that
takes longer than TIMEOUT seconds counts as a failure and is flagged in `timeout`.

GEP prints its models in its own notation (e^(...), ln(...), √(...), (...)², features
x1 ... xn); `gep_to_sympy` rewrites that to sympy with the problem's symbols. PhySO's
models are read from their infix strings with the constant values put in, and their
constant subexpressions evaluated with PhySO's protected operators (`physo_to_sympy`),
as PhySO computes them; its square roots and logarithms take the absolute value, as
PhySO's do. Both are read without sympy's evaluation, and floats that are exact integers
become integers (see `_parse`).

The check rounds each float of the expression as it is written, so before it each model
is brought to one coefficient per term (`canonical`): products of numbers are merged, and
numbers in a sum inside exp are taken out of it (exp(a + c) = e^c exp(a)). Otherwise the
verdict would depend on how a model happens to be written: 0.368 exp(u + 1) would round
to 0.37 exp(u + 1) = 1.006 exp(u) and fail, where the same model as 1.0007 exp(u) passes.
PhySO's rounding of the floats is applied with exact node replacement (`_round_floats`),
and its pi-fraction step only for formulas with a trigonometric function (`check`).
audit_symbolic.py checks the readings and the verdicts against the models' numbers.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import signal

import numpy as np
import pandas as pd
import sympy

import common
import physo.benchmark.utils.symbolic_utils as su

TIMEOUT = 60


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _parse(text, local):
    """sympy expression of `text`, without the evaluation that rewrites exp(a + 1.0) as
    2.718...*exp(a) (two-decimal rounding would turn that into 2.72*exp(a)); floats that
    are exact integers (the constant 1 both methods have) become sympy Integers."""
    expr = sympy.parsing.sympy_parser.parse_expr(text, local_dict=local, evaluate=False)
    return expr.xreplace({f: sympy.Integer(int(f)) for f in expr.atoms(sympy.Float)
                          if float(f).is_integer()})


def gep_to_sympy(expr, pb):
    s = expr.replace("√(", "sqrt(").replace("e^(", "exp(").replace("ln(", "log(")
    s = s.replace("²", "**2")
    local = {"x%d" % (i + 1): sym for i, sym in enumerate(pb.X_sympy_symbols)}
    local.update(exp=sympy.exp, log=sympy.log, sqrt=sympy.sqrt, sin=sympy.sin, cos=sympy.cos)
    return _parse(s, local)


# PhySO's protected operators (physo/physym/functions.py), which its models are fitted
# and scored with
EPS, INF, EXP_THRESHOLD = 1e-3, 1e6, 80.0


class _C:
    """A constant subexpression of a PhySO model, evaluated as PhySO evaluates it: x/y
    is 1 and x**-1 is 0 where |y|, |x| <= EPS, exp levels off above EXP_THRESHOLD, log
    and sqrt take |x|. A constant like cos(exp(1/(c1**2 - c1)) - c2) at c1 = c2 = 1 is
    cos(exp(0) - 1) = 1 to PhySO, while sympy makes it nan. With a symbolic operand the
    constant turns into a sympy Float."""

    def __init__(self, v):
        self.v = float(v)

    def _sympy_(self):
        return sympy.Float(self.v)

    @staticmethod
    def _val(o):
        return o.v if isinstance(o, _C) else float(o) if isinstance(o, (int, float)) else None

    def _op(self, o, f, reverse=False):
        w = self._val(o)
        if w is None:                                  # symbolic operand
            a, b = sympy.Float(self.v), o
            return f(b, a) if reverse else f(a, b)
        a, b = (w, self.v) if reverse else (self.v, w)
        return _C(f(a, b))

    __add__ = lambda s, o: s._op(o, lambda a, b: a + b)
    __radd__ = lambda s, o: s._op(o, lambda a, b: a + b, True)
    __sub__ = lambda s, o: s._op(o, lambda a, b: a - b)
    __rsub__ = lambda s, o: s._op(o, lambda a, b: a - b, True)
    __mul__ = lambda s, o: s._op(o, lambda a, b: a * b)
    __rmul__ = lambda s, o: s._op(o, lambda a, b: a * b, True)
    __neg__ = lambda s: _C(-s.v)

    def __truediv__(self, o):
        w = self._val(o)
        if w is None:
            return sympy.Float(self.v) / o
        return _C(self.v / w if abs(w) > EPS else 1.0)

    def __rtruediv__(self, o):
        # o / constant: protected_div gives 1 whatever o is
        if abs(self.v) <= EPS:
            return _C(1.0)
        return o / sympy.Float(self.v) if self._val(o) is None else _C(self._val(o) / self.v)

    def __pow__(self, p):
        p = self._val(p)
        if p == -1:
            return _C(1.0 / self.v if abs(self.v) > EPS else 0.0)
        if p == 0.5:                                   # sqrt prints as **(0.5)
            return _C(np.sqrt(abs(self.v)))
        if p in (2, 3, 4) and abs(self.v) > INF:
            return _C(INF ** p * (np.sign(self.v) if p == 3 else 1.0))
        return _C(self.v ** p)


def _unary(sym_f, num_f):
    return lambda a: _C(num_f(a.v)) if isinstance(a, _C) else sym_f(a)


_PHYSO_FUNCS = dict(
    exp=_unary(sympy.exp, lambda x: np.exp(min(x, EXP_THRESHOLD))),
    log=_unary(lambda a: sympy.log(sympy.Abs(a)),
               lambda x: np.log(abs(x)) if abs(x) >= EPS else np.log(EPS)),
    sqrt=_unary(lambda a: sympy.sqrt(sympy.Abs(a)), lambda x: np.sqrt(abs(x))),
    sin=_unary(sympy.sin, np.sin), cos=_unary(sympy.cos, np.cos))


def _protected_roots(expr):
    """PhySO writes its square root of a variable expression as `**(0.5)` and computes
    it as the root of the absolute value."""
    return expr.replace(lambda e: e.is_Pow and e.exp == 0.5,
                        lambda e: sympy.sqrt(sympy.Abs(e.base)))


def physo_to_sympy(raw, pb):
    """The model from its infix string (Python syntax), the constants and number
    literals as `_C`, the variables as the problem's sympy symbols."""
    text = re.sub(r"(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?)", r"_C(\1)", raw["infix"])
    scope = dict(pb.sympy_X_symbols_dict, _C=_C, **_PHYSO_FUNCS)
    scope.update({k: _C(v) for k, v in raw["constants"].items()})
    with sympy.evaluate(False):
        expr = eval(text, {"__builtins__": {}}, scope)
    expr = sympy.Float(expr.v) if isinstance(expr, _C) else _protected_roots(sympy.sympify(expr))
    return expr.xreplace({f: sympy.Integer(int(f)) for f in expr.atoms(sympy.Float)
                          if float(f).is_integer()})


def _evaluated(e):
    """`e` rebuilt node by node with sympy's evaluation (no printing and parsing back)."""
    return e.func(*[_evaluated(a) for a in e.args]) if e.args else e


def canonical(expr):
    """One coefficient per term: sympy's evaluation, products and the sums in exponents
    expanded, numbers evaluated."""
    e = _evaluated(expr)
    e = sympy.expand(e, deep=True, mul=True, multinomial=False, power_exp=True,
                     power_base=False, log=False)
    return e.evalf()


def _round_floats(expr, round_decimal=2):
    """PhySO's rounding rule (physo.benchmark.utils.symbolic_utils.round_floats): e as
    2.72, a float less than 10^-round_decimal from an integer as that integer (so below 0.01
    as 0), any other rounded to round_decimal decimals. PhySO replaces the floats one by
    one with sympy's `subs`, which can put a value on the wrong node (1.0 x + 0.0126 y
    comes out as 0.0126 x + 0.01 y); here every float is replaced at once, exactly
    (`xreplace`)."""
    expr = expr.xreplace({a: sympy.Float(round(float(a), round_decimal))
                          for a in expr.atoms(sympy.core.numbers.Exp1)})
    lim = 10.0 ** -round_decimal
    repl = {}
    for a in expr.atoms(sympy.Float):
        v = float(a)
        n = round(v)
        repl[a] = sympy.Integer(n) if abs(v - n) < lim else sympy.Float(round(v, round_decimal))
    return expr.xreplace(repl)


su.round_floats = _round_floats       # compare_expression rounds through this name


def has_trig(pb):
    return pb.formula_sympy_eval.has(sympy.sin, sympy.cos, sympy.tan)


def check(pb, expr, to_sympy, handle_trigo=None):
    """(recovered, timed_out). PhySO's check also tries the expression with every float
    replaced by a fraction p/q of pi (q <= 10) less than 0.01 pi from it, for constants
    such as pi/2 in a sine; that turns any coefficient below 0.031 into 0. It is used
    only where the formula has a trigonometric function (`handle_trigo` None), as none
    of the 15 here has."""
    if handle_trigo is None:
        handle_trigo = has_trig(pb)
    if expr is None:
        return False, False
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        ok, _ = pb.compare_expression(canonical(to_sympy(expr, pb)),
                                      handle_trigo=handle_trigo)
        return bool(ok), False
    except _Timeout:
        return False, True
    except Exception:
        return False, False
    finally:
        signal.alarm(0)


def judge(path):
    r = json.load(open(path))
    r["expression_sympy"] = r["expression"]
    pb = common.problem(r["equation"])
    if r["method"] == "PhySO":
        to_sympy, others = physo_to_sympy, r["pareto_raw"]
        r["expression"] = r["expression_raw"]
    else:
        to_sympy, others = gep_to_sympy, r["hall_of_fame"]
    rec, tout = check(pb, r["expression"], to_sympy)
    rec_any = rec
    for e in others:
        if rec_any:
            break
        rec_any = check(pb, e, to_sympy)[0]
    group = dict(common.EQUATIONS)[r["equation"]]
    r2 = r["r2_test"]
    return dict(method=r["method"], equation=r["equation"], group=group, seed=r["seed"],
                recovered=rec, recovered_any=rec_any, timeout=tout,
                r2_test=float("nan") if r2 is None else float(r2),
                wall_time=r["wall_time"], epochs_run=r["epochs_run"],
                evaluations=r["evaluations"],
                expression=r["expression"] if r["method"] == "GEP-SBP" else r["expression_sympy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    noise = ap.parse_args().noise
    tag = common.tag(noise)
    files = sorted(glob.glob(os.path.join(common.RESULTS_DIR, "gep" + tag, "*.json")) +
                   glob.glob(os.path.join(common.RESULTS_DIR, "physo" + tag, "*.json")))
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(judge, files, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["method", "equation", "seed"])
    df.insert(4, "noise", noise)
    df.to_csv(os.path.join(common.RESULTS_DIR, "summary%s.csv" % tag), index=False)
    order = [e for e, _ in common.EQUATIONS]
    per_eq = (df.groupby(["equation", "method"])
                .agg(recovery=("recovered", "mean"), r2_median=("r2_test", "median"),
                     wall_median=("wall_time", "median"), runs=("seed", "count"))
                .unstack("method").reindex(order))
    pd.set_option("display.width", 200)
    print(per_eq.round(4))
    print(df.groupby("method").agg(recovery=("recovered", "mean"),
                                   recovery_any=("recovered_any", "mean"),
                                   r2_median=("r2_test", "median"),
                                   wall_median=("wall_time", "median"),
                                   wall_total=("wall_time", "sum"),
                                   timeouts=("timeout", "sum"),
                                   runs=("seed", "count")))


if __name__ == "__main__":
    main()
