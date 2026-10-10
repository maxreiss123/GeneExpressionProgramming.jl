"""Write the comparison's data and metadata for the Julia harness (gep_run.jl).

    python export_data.py [--noise 0 0.05 0.1]

For each noise level, data<tag>/meta.json lists the equations with their variable names,
SI units and formula, and the normalised training RMSE at which each run stops
(`stop_nrmse`, per equation and seed; see common.py); data<tag>/<EQUATION>/train_s<SEED>.csv
and data<tag>/<EQUATION>/test.csv hold the samples of common.make_data (columns x1 ... xn,
y), which physo_run.py draws in memory. The tag is empty without noise and _noise<level>
with it (common.tag).
"""
import argparse
import json
import os

import numpy as np

import common


def export(noise):
    root = common.DATA_DIR + common.tag(noise)
    meta, stops = [], {}
    for name, group in common.EQUATIONS:
        pb = common.problem(name)
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        header = ",".join(["x%d" % (i + 1) for i in range(pb.n_vars)] + ["y"])
        for seed in common.SEEDS:
            Xtr, ytr, Xte, yte = common.make_data(name, seed, noise)
            np.savetxt(os.path.join(d, "train_s%d.csv" % seed), np.vstack([Xtr, ytr]).T,
                       delimiter=",", header=header, comments="", fmt="%.17g")
        np.savetxt(os.path.join(d, "test.csv"), np.vstack([Xte, yte]).T,
                   delimiter=",", header=header, comments="", fmt="%.17g")
        stops[name] = {str(s): common.stop_nrmse(name, s, noise) for s in common.SEEDS}
        meta.append(dict(
            name=name, group=group, formula=pb.formula_original,
            variables=list(map(str, pb.X_names)),
            units=[common.si_units(u) for u in pb.X_units],
            target=str(pb.y_name), target_units=common.si_units(pb.y_units)))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(dict(seeds=common.SEEDS, n_train=common.N_TRAIN, n_test=common.N_TEST,
                       noise=noise, stop_nrmse=stops,
                       unit_order=["kg", "m", "s", "K", "mol", "A", "cd"], equations=meta),
                  f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=common.NOISES)
    for noise in ap.parse_args().noise:
        export(noise)


if __name__ == "__main__":
    main()
