"""Judge the runs of both methods with one symbolic criterion; write results/summary<tag>.csv.

    python judge.py [--noise 0]
    python judge.py --runs-dir results/pysr_niter100 [--noise 0]

Reads results/gep<tag>/ (GEP-SBP) and results/pysr<tag>/ (PySR; tag as common.tag: none
without noise). With --runs-dir it judges only the runs in that folder (PySR's when the
folder name starts with "pysr", GEP-SBP's otherwise) and writes
results/summary_<folder>.csv.

The criterion is SRBench's symbolic solution (La Cava et al., 2021): after every float of
the model and of the formula is rounded to two decimals, the model counts as recovered if
its difference to the formula, or the ratio of the two, simplifies to a constant. So a
model is accepted up to an additive or a multiplicative constant; test R^2 shows whether
it has the formula's own constants. The check is `compare_expression` of the physo
package's AI-Feynman benchmark (`FeynmanProblem.compare_expression`), used as a library,
with these corrections:

* one coefficient per term (`canonical`): products of numbers are merged, and numbers in
  a sum inside exp are taken out of it (exp(a + c) = e^c exp(a)), before the rounding,
  so that the verdict does not depend on how a model happens to be written;
* exact rounding (`_round_floats`): the rounding rule of the library (e as 2.72, a float
  less than 0.01 from an integer as that integer, so below 0.01 as 0, any other to two
  decimals) with every float replaced at once; the library's `subs` can put a value on
  the wrong node;
* no pi fractions: the library also tries every float as a fraction p/q of pi (q <= 10)
  less than 0.01 pi from it, for constants in a sine, which turns any coefficient below
  0.031 into 0; none of the 15 formulas has a trigonometric function, so it is left out
  (`handle_trigo`, see `check`);
* no zero models (`_degenerate`): the library divides the formula by the model after
  rounding and simplifying it; for a model that becomes 0 the ratio is nan, which sympy's
  `is_constant` calls constant, so the library accepted it;
* a timeout that holds: a check that takes longer than TIMEOUT seconds counts as a
  failure and is flagged in `timeout`; the alarm is not an Exception, which the library
  catches around each of its steps.

The check runs on the model a run returns (`recovered`) and on every model it reports
(`recovered_any`): PySR's hall of fame, which is its accuracy-complexity front (the best
model of every size), and GEP-SBP's hall of fame (3 models). PySR returns the model its
`model_selection="best"` picks: of the models within 1.5x the lowest loss, the one whose
loss fell most per unit of size. `recovered_accuracy` judges the model with the lowest
loss instead (the one GEP-SBP returns); for GEP-SBP the two are the same model.

GEP-SBP prints its models in its own notation (e^(...), ln(...), √(...), (...)², features
x1 ... xn; `gep_to_sympy`), PySR in Julia syntax (square(x) for x^2, constants to 17
significant digits; `pysr_to_sympy`). Both are read without sympy's evaluation, which
would rewrite exp(a + 1.0) as 2.718...*exp(a), and floats that are exact integers become
integers (`_parse`). PySR's sqrt and log return NaN outside their domain, where sympy's
are complex; a model with such values has an infinite loss in PySR, so it is never a
returned model's reading on the data. audit_symbolic.py checks the readings and the
verdicts against the models' numbers.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import signal

import pandas as pd
import sympy

import common
import physo.benchmark.utils.symbolic_utils as su

TIMEOUT = 60
RESULTS_DIR = common.RESULTS_DIR


class _Timeout(BaseException):
    """Not an Exception: the library's compare_expression catches Exception around each
    of its steps, which would swallow the alarm and let the check run on without one."""


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


def pysr_to_sympy(text, pb):
    local = {"x%d" % (i + 1): s for i, s in enumerate(pb.X_sympy_symbols)}
    local.update(square=lambda a: sympy.Pow(a, 2, evaluate=False),
                 cube=lambda a: sympy.Pow(a, 3, evaluate=False),
                 exp=sympy.exp, log=sympy.log, sqrt=sympy.sqrt, sin=sympy.sin, cos=sympy.cos,
                 Inf=sympy.oo, NaN=sympy.nan)
    return _parse(text, local)


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
    """The library's rounding rule (physo.benchmark.utils.symbolic_utils.round_floats): e
    as 2.72, a float less than 10^-round_decimal from an integer as that integer (so below
    0.01 as 0), any other rounded to round_decimal decimals. The library replaces the
    floats one by one with sympy's `subs`, which can put a value on the wrong node
    (1.0 x + 0.0126 y comes out as 0.0126 x + 0.01 y); here every float is replaced at
    once, exactly (`xreplace`)."""
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


def _degenerate(e):
    """Zero everywhere, or with an undefined or infinite number in it."""
    return e == 0 or e.has(sympy.nan, sympy.zoo, sympy.oo, sympy.S.NegativeInfinity)


def check(pb, expr, to_sympy, handle_trigo=None):
    """(recovered, timed_out). The library's pi-fraction step is used only where the
    formula has a trigonometric function (`handle_trigo` None), as none of the 15 here
    has; audit_symbolic.py reports the verdicts with it as well."""
    if handle_trigo is None:
        handle_trigo = has_trig(pb)
    if expr is None:
        return False, False
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        trial = canonical(to_sympy(expr, pb))
        ok, _ = pb.compare_expression(trial, handle_trigo=handle_trigo)
        # the library takes the ratio of the formula to the model as it cleans it
        # (rounded, simplified); for a model that cleans to 0 that is nan, which sympy's
        # is_constant calls constant
        if ok and _degenerate(su.clean_sympy_expr(trial)):
            ok = False
        return bool(ok), False
    except _Timeout:
        return False, True
    except Exception:
        return False, False
    finally:
        signal.alarm(0)


def judge_gep(path):
    r = json.load(open(path))
    pb = common.problem(r["equation"])
    rec, tout = check(pb, r["expression"], gep_to_sympy)
    rec_any = rec
    for e in r["hall_of_fame"]:
        if rec_any:
            break
        rec_any = check(pb, e, gep_to_sympy)[0]
    r2 = r["r2_test"]
    return dict(method=r["method"], equation=r["equation"],
                group=dict(common.EQUATIONS)[r["equation"]], seed=r["seed"],
                recovered=rec, recovered_any=rec_any, timeout=tout,
                r2_test=float("nan") if r2 is None else float(r2),
                wall_time=r["wall_time"], epochs_run=r["epochs_run"],
                evaluations=r["evaluations"], expression=r["expression"],
                recovered_accuracy=rec,
                r2_test_accuracy=float("nan") if r2 is None else float(r2))


def judge_pysr(path):
    r = json.load(open(path))
    pb = common.problem(r["equation"])
    rec, tout = check(pb, r["expression"], pysr_to_sympy)
    rec_acc = rec if r["expression_accuracy"] == r["expression"] else \
        check(pb, r["expression_accuracy"], pysr_to_sympy)[0]
    rec_any = rec or rec_acc
    for e in r["pareto"]:
        if rec_any:
            break
        rec_any = check(pb, e, pysr_to_sympy)[0]
    r2 = r["r2_test"]
    return dict(method="PySR", equation=r["equation"], group=dict(common.EQUATIONS)[r["equation"]],
                seed=r["seed"], recovered=rec, recovered_any=rec_any, timeout=tout,
                r2_test=float("nan") if r2 is None else float(r2),
                wall_time=r["wall_time"], epochs_run=r["epochs_run"],
                evaluations=r["evaluations"], recovered_accuracy=rec_acc,
                r2_test_accuracy=r["r2_test_accuracy"], expression=r["expression_sympy"])


def judge_any(path):
    return judge_pysr(path) if os.path.basename(os.path.dirname(path)).startswith("pysr") \
        else judge_gep(path)


def load(noise):
    tag = common.tag(noise)
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "gep" + tag, "*.json")) +
                  glob.glob(os.path.join(RESULTS_DIR, "pysr" + tag, "*.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args()
    if a.runs_dir:
        files = sorted(glob.glob(os.path.join(a.runs_dir, "*.json")))
        out = "summary_%s.csv" % os.path.basename(os.path.normpath(a.runs_dir))
    else:
        files = load(a.noise)
        out = "summary%s.csv" % common.tag(a.noise)
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(judge_any, files, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["method", "equation", "seed"])
    df.insert(4, "noise", a.noise)
    df.to_csv(os.path.join(RESULTS_DIR, out), index=False)
    order = [e for e, _ in common.EQUATIONS]
    per_eq = (df.groupby(["equation", "method"])
                .agg(recovery=("recovered", "mean"), r2_median=("r2_test", "median"),
                     wall_median=("wall_time", "median"), runs=("seed", "count"))
                .unstack("method").reindex(order))
    pd.set_option("display.width", 200)
    print(per_eq.round(4))
    print(df.groupby("method").agg(recovery=("recovered", "mean"),
                                   recovery_any=("recovered_any", "mean"),
                                   recovery_accuracy=("recovered_accuracy", "mean"),
                                   r2_median=("r2_test", "median"),
                                   wall_median=("wall_time", "median"),
                                   wall_total=("wall_time", "sum"),
                                   timeouts=("timeout", "sum"),
                                   runs=("seed", "count")))


if __name__ == "__main__":
    main()
