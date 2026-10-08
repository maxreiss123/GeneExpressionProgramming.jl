"""One PhySO run on one equation and seed of the comparison.

    python physo_run.py EQUATION SEED [--noise S] [--epochs N] [--threads T]

PhySO 1.1.11 as in its Feynman benchmark (benchmarking/FeynmanBenchmark/feynman_config.py):
run configuration config1 (10 000 candidates per epoch, LBFGS for the free constants),
operators mul add sub div inv n2 sqrt neg exp log sin cos, the fixed constant 1 and two
dimensionless free constants, the units of PhySO's own Feynman table. Deviations from that
benchmark, both for the shared budget: `--epochs` epochs (so `epochs * 10 000` candidate
evaluations instead of 1e6), sequential mode with `--threads` torch threads, and an early
stop. Without noise the run stops once a candidate reaches reward 1 - 1e-5 (normalised RMSE
~1e-5), as PhySO's docs recommend with free constants; with `--noise` the training targets
are noisy (common.make_data) and it stops at the noise floor, the normalised RMSE of the
true formula itself (common.stop_nrmse). GEP stops at the same fit.

Writes results/physo<tag>/<EQUATION>_s<SEED>.json (tag as common.tag): the best expression
and the Pareto front
(sympy strings with the constants evaluated, and the raw infix strings with the constant
values apart), R^2 on the train and test sets, the wall time of the physo.SR call, and
the number of epochs run.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import common
import physo

BATCH_SIZE = physo.config.config1.config1["learning_config"]["batch_size"]

OP_NAMES = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("equation")
    ap.add_argument("seed", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(common.RESULTS_DIR, "physo" + common.tag(a.noise))
    stop = common.stop_nrmse(a.equation, a.seed, a.noise)

    torch.set_num_threads(a.threads)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    pb = common.problem(a.equation)
    Xtr, ytr, Xte, yte = common.make_data(a.equation, a.seed, a.noise)
    dimless = np.zeros(len(pb.y_units))

    logger = physo.learn.monitoring.RunLogger(save_path=None, do_save=False)
    vis = physo.learn.monitoring.RunVisualiser(epoch_refresh_rate=10**9, do_show=False,
                                               do_prints=False, do_save=False)
    t0 = time.perf_counter()
    try:
        expr, logs = physo.SR(
            Xtr, ytr,
            X_names=pb.X_names, X_units=pb.X_units,
            y_name=pb.y_name, y_units=pb.y_units,
            fixed_consts=[1.], fixed_consts_units=[dimless],
            free_consts_names=["c1", "c2"], free_consts_units=[dimless, dimless],
            op_names=OP_NAMES,
            run_config=physo.config.config1.config1,
            get_run_logger=lambda: logger, get_run_visualiser=lambda: vis,
            stop_reward=1 / (1 + stop), stop_after_n_epochs=0,
            epochs=a.epochs, max_n_evaluations=None,
            parallel_mode=False,
        )
    except IndexError:
        # physo.SR builds the Pareto front of the candidates with a positive reward and
        # fails on an empty one: no candidate of the whole run met the units
        expr = None
    logs = logger
    wall = time.perf_counter() - t0

    def sym(prog):
        try:
            e = prog.get_infix_sympy(evaluate_consts=True)
            # ClassSR programs give one expression per realization; SR has one
            return str(e[0] if isinstance(e, (list, tuple, np.ndarray)) else e)
        except Exception:
            # sympy fails to print some models with degenerate constants (a
            # ZeroDivisionError in evalf); the string is for display only, and
            # judge.py reads the raw infix string
            return prog.get_infix_str()

    def predict(prog, X):
        with torch.no_grad():
            out = prog.execute(torch.tensor(X, dtype=torch.float64))
        return out.detach().cpu().numpy().astype(float)

    def raw(prog):
        # the infix string with the constants by name, and their values: judge.py
        # substitutes them without the evaluation sympy's `subs` does
        return dict(infix=prog.get_infix_str(), constants=prog.get_sympy_local_dicts()[0])

    if expr is None:
        pareto, pareto_r = [], []
    else:
        _, pareto, pareto_r, _ = logs.get_pareto_front()
    res = dict(
        method="PhySO", equation=a.equation, seed=a.seed,
        expression=None if expr is None else sym(expr),
        expression_raw=None if expr is None else raw(expr),
        pareto=[sym(p) for p in pareto], pareto_raw=[raw(p) for p in pareto],
        pareto_reward=[float(r) for r in pareto_r],
        r2_train=float("nan") if expr is None else common.r2(ytr, predict(expr, Xtr)),
        r2_test=float("nan") if expr is None else common.r2(yte, predict(expr, Xte)),
        wall_time=wall, epochs_run=len(logs.epochs_history),
        evaluations=len(logs.epochs_history) * BATCH_SIZE,
        epochs_budget=a.epochs, threads=a.threads, noise=a.noise, stop_nrmse=stop,
        physo_version=physo.__version__,
    )
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "%s_s%d.json" % (a.equation, a.seed)), "w") as f:
        json.dump(res, f, indent=1)
    print("%s s%d  r2_test=%.6f  wall=%.1fs  %s" % (a.equation, a.seed, res["r2_test"], wall,
                                                    res["expression"]))


if __name__ == "__main__":
    main()
