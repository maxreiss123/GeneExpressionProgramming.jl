"""PySR runs on the equations of the comparison; one process runs a share of them.

    python pysr_run.py [--noise S] [--worker k/K] [--equations A,B] [--seeds 1,2]
                       [--runs EQ:SEED,...] [--niterations N] [--stop floor|none]
                       [--units si|none] [--out DIR]

The protocol is common.py's: the 15 equations, the data (common.make_data, drawn in
memory; gep_run.jl reads the same samples from export_data.py's files), the units (the
AI-Feynman unit table, as SI exponents) and the stop.

PySR 2.7.0 (SymbolicRegression.jl 2.7) with its defaults (31 populations of 27, maxsize 30,
BFGS for the constants, its default plugins, `model_selection="best"`), except where the
shared protocol fixes a setting:

* operators + - * / square sqrt exp log sin cos, those of gep_run.jl;
* the units of the inputs and the target as `X_units` and `y_units`, with
  `dimensionless_constants_only=True`: like GEP-SBP's linear scaling coefficients, the
  constants carry no units. A candidate whose units do not check gets PySR's default
  penalty (1000) added to its loss;
* the budget: 11 iterations. A PySR iteration runs, in each of the 31 populations, 380
  cycles of ceil(27 / 15) = 2 tournament rounds; a round is a mutation (one candidate) or,
  with probability 0.2, a crossover (two). With the 31 * 27 random starting candidates,
  837 + 11 * 31 * 760 * 1.2 ~ 3.1e5 candidates (a little fewer, as some mutations are
  no-ops), as GEP-SBP's 1 700 + 430 * 700;
* the stop: `early_stop_condition` at the mean squared error (stop_nrmse * std(y))^2,
  with stop_nrmse from common.stop_nrmse and the n-1 standard deviation, the fit at which
  gep_run.jl stops: 1e-5 without noise, the noise floor with it. PySR checks it after
  every population's cycle;
* one CPU thread (`parallelism="serial"`, juliacall started with one thread),
  `deterministic=True` with `random_state` = seed, and 64-bit floats (PySR's default is
  32), as the data and GEP-SBP are;
* `print_precision=17`: PySR reads its models back from the equations it prints, with 5
  significant digits by default, so its predictions and expressions would carry rounded
  constants. This changes no search step.

Every run builds a fresh `PySRRegressor` and times its `fit` (the search, then PySR's
reading of the hall of fame), so the time covers what a user waits for. A warm-up fit on
the first run's data (1 iteration, not recorded) compiles the Julia code first, so the
wall times exclude the compilation, as gep_run.jl's do.

A read-only plugin (`RunRecord`) takes from the search state at its end the number of
population cycles completed and SymbolicRegression.jl's own count of evaluations (which
also counts the constant optimiser's function calls); it changes nothing in the search.

For the sensitivity checks of the README, `--niterations 100` gives PySR its default
budget (about 9 times the shared one) and `--units none` leaves the units out.

Writes results/pysr<tag>/<EQUATION>_s<SEED>.json (tag as common.tag): the selected model
(`expression`, in PySR's syntax with the features x1 ... xn), the hall of fame (`pareto`,
PySR's accuracy-complexity front: the best model of every size), the model with the
lowest loss, R^2 on the train and test sets, the wall time and the budget used.
Runs whose JSON exists are skipped.
"""
import argparse
import os
import tempfile
import time

# juliacall before torch (imported by physo, through common): the other order can crash
from pysr import PySRRegressor, jl  # noqa: E402
from pysr.plugins import AbstractPlugin  # noqa: E402

import json  # noqa: E402

import numpy as np  # noqa: E402

import common  # noqa: E402

RESULTS_DIR = common.RESULTS_DIR
UNIT_NAMES = ["kg", "m", "s", "K", "mol", "A", "cd"]   # the order of common.si_units
BINARY = ["+", "-", "*", "/"]
UNARY = ["square", "sqrt", "exp", "log", "sin", "cos"]
POPULATIONS, POPULATION_SIZE, NCYCLES, TOURNAMENT_N, P_CROSSOVER = 31, 27, 380, 15, 0.2
ROUNDS_PER_CYCLE = NCYCLES * -(-POPULATION_SIZE // TOURNAMENT_N)       # 760

jl.seval("""
module RunRecord
import SymbolicRegression
struct Recorder <: SymbolicRegression.AbstractPlugin end
const LAST = Ref{Any}(nothing)
function SymbolicRegression.on_search_end!(_, ::Recorder, state, dataset, options, ropt)
    LAST[] = (sum(sum, state.num_evals), sum(state.cycles_remaining))
    return nothing
end
end
""")


class RunRecord(AbstractPlugin):
    def julia_plugin(self):
        return jl.RunRecord.Recorder()


def unit_string(exponents):
    """A DynamicQuantities unit string of SI exponents (common.si_units)."""
    parts = []
    for name, e in zip(UNIT_NAMES, exponents):
        if e == 0:
            continue
        e = int(e) if float(e).is_integer() else e
        parts.append(name if e == 1 else "%s^(%s)" % (name, e))
    return " * ".join(parts) or "1"


def regressor(seed, niterations, threshold, tmpdir):
    return PySRRegressor(
        binary_operators=BINARY, unary_operators=UNARY,
        niterations=niterations, populations=POPULATIONS, population_size=POPULATION_SIZE,
        ncycles_per_iteration=NCYCLES, tournament_selection_n=TOURNAMENT_N,
        crossover_probability=P_CROSSOVER,
        early_stop_condition=threshold, dimensionless_constants_only=True,
        precision=64, parallelism="serial", deterministic=True, random_state=seed,
        print_precision=17, verbosity=0, progress=False,
        temp_equation_file=True, tempdir=tmpdir, plugins=[RunRecord()],
    )


def run_one(name, seed, noise, niterations, stop_mode, tmpdir, out=None, units=True):
    pb = common.problem(name)
    Xtr, ytr, Xte, yte = common.make_data(name, seed, noise)
    stop = common.stop_nrmse(name, seed, noise) if stop_mode == "floor" else 0.0
    # PySR stops once a model's loss (the mean squared error) is below this
    threshold = (stop * np.std(ytr, ddof=1)) ** 2 if stop > 0 else None
    x_units = [unit_string(common.si_units(u)) for u in pb.X_units] if units else None
    y_units = unit_string(common.si_units(pb.y_units)) if units else None
    variables = ["x%d" % (i + 1) for i in range(pb.n_vars)]

    model = regressor(seed, niterations, threshold, tmpdir)
    t0 = time.perf_counter()
    model.fit(Xtr.T, ytr, X_units=x_units, y_units=y_units, variable_names=variables)
    wall = time.perf_counter() - t0
    sr_evals, cycles_left = jl.seval("RunRecord.LAST[]")
    cycles = niterations * POPULATIONS - int(cycles_left)

    hof = model.equations_
    best = model.get_best()
    accurate = hof.loc[hof["loss"].idxmin()]

    def r2(X, y, row=None):
        try:
            p = model.predict(X.T) if row is None else model.predict(X.T, index=row)
            return common.r2(y, np.asarray(p, dtype=float))
        except Exception:
            return float("nan")

    res = dict(
        method="PySR", equation=name, seed=seed,
        expression=str(best["equation"]), expression_sympy=str(best["sympy_format"]),
        loss=float(best["loss"]), complexity=int(best["complexity"]),
        expression_accuracy=str(accurate["equation"]), loss_min=float(accurate["loss"]),
        pareto=[str(e) for e in hof["equation"]],
        pareto_loss=[float(v) for v in hof["loss"]],
        pareto_complexity=[int(v) for v in hof["complexity"]],
        variables=variables, X_units=x_units, y_units=y_units,
        r2_train=r2(Xtr, ytr), r2_test=r2(Xte, yte),
        r2_test_accuracy=r2(Xte, yte, int(hof["loss"].idxmin())),
        wall_time=wall, epochs_run=cycles / POPULATIONS, cycles_run=cycles,
        evaluations=int(round(POPULATIONS * POPULATION_SIZE +
                              cycles * ROUNDS_PER_CYCLE * (1 + P_CROSSOVER))),
        sr_num_evals=float(sr_evals),
        epochs_budget=niterations, threads=int(jl.seval("Threads.nthreads()")),
        noise=noise, stop_nrmse=stop, stop_mse=threshold,
        pysr_version=_version("pysr"),
        symbolic_regression_jl=str(jl.seval("pkgversion(SymbolicRegression)")),
        julia_version=str(jl.seval("string(VERSION)")),
    )
    if out is not None:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "%s_s%d.json" % (name, seed)), "w") as f:
            json.dump(res, f, indent=1)
    print("%s s%d  r2_test=%.6f  wall=%.1fs  iterations=%.2f  %s"
          % (name, seed, res["r2_test"], wall, res["epochs_run"], res["expression_sympy"]),
          flush=True)
    return res


def _version(pkg):
    from importlib.metadata import version
    return version(pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--worker", default="1/1")
    ap.add_argument("--equations", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--niterations", type=int, default=11)
    ap.add_argument("--stop", choices=["floor", "none"], default="floor")
    ap.add_argument("--units", choices=["si", "none"], default="si")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(RESULTS_DIR, "pysr" + common.tag(a.noise))

    eqs = [e for e, _ in common.EQUATIONS]
    if a.equations:
        eqs = [e for e in eqs if e in a.equations.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else common.SEEDS
    jobs = [(e, s) for e in eqs for s in seeds]
    if a.runs:
        jobs = [(r.split(":")[0], int(r.split(":")[1])) for r in a.runs.split(",")]
    k, K = (int(v) for v in a.worker.split("/"))
    jobs = jobs[k - 1::K]
    todo = [j for j in jobs if not os.path.exists(os.path.join(out, "%s_s%d.json" % j))]
    if not todo:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # compile the code paths before anything is timed
        run_one(*todo[0], a.noise, 1, a.stop, tmpdir, units=a.units == "si")
        for name, seed in todo:
            run_one(name, seed, a.noise, a.niterations, a.stop, tmpdir, out=out,
                    units=a.units == "si")


if __name__ == "__main__":
    main()
