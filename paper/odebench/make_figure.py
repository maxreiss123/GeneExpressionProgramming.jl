"""Accuracy on ODEBench: this package against ODEFormer.

Reads the per-condition JSON of odebench_gep.jl (results/gep_sigma*_rho*_seed*.json) and
plots the fraction of the 63 systems with variance-weighted R^2 > 0.9 -- the paper's
metric -- against the noise level, for both subsampling settings and both tasks.

One group of columns per noise level, one column per subsampling setting. The ODEFormer
values are quoted from the FIM-ODE paper (arXiv:2602.08733), which re-evaluated
ODEFormer on ODEBench: reconstruction accuracy 63.1 % clean and 61.5 % at sigma = 0.03
and 0.05, no subsampling; its own tables were not retrieved. They are drawn as reference
ticks on the rho = 0 columns of the levels run here (0 and 0.05; 0.03 was not), on the
reconstruction panel only; for generalization the figure states that no reference value
is available.
"""

import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

C_RHO0 = "#1c5cab"      # blue 550  } one hue, two steps: the full grid and half of
C_RHO5 = "#86b6ef"      # blue 250  } it (validated as an ordinal pair)
C_ODEF = "#1baf7a"
INK2 = "#52514e"
MUTED = "#9a9892"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})

# ODEFormer on ODEBench as re-evaluated by FIM-ODE (arXiv:2602.08733); rho = 0,
# reconstruction only. See module docstring.
ODEFORMER_RECON_RHO0 = {0.0: 0.631, 0.03: 0.615, 0.05: 0.615}


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", "gep_sigma*_rho*_seed*.json"))):
        d = json.load(open(p))
        key = (float(d["sigma"]), float(d["rho"]))
        n = len(d["results"])
        acc = lambda k: sum(
            1 for r in d["results"]
            if r[k] is not None and r[k] > 0.9) / n
        med_t = float(np.median([r["fit_time_s"] + r["selection_time_s"]
                                 for r in d["results"]]))
        runs[key] = dict(rec=acc("r2_reconstruction"), gen=acc("r2_generalization"),
                         n=n, time=med_t)
    return runs


def main():
    runs = load()
    if not runs:
        raise SystemExit("no results yet")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    sigmas = sorted({s for s, _ in runs})
    x = np.arange(len(sigmas))
    w = 0.32

    for ax, task, title in ((axes[0], "rec", "(a) reconstruction"),
                            (axes[1], "gen", "(b) generalization")):
        for k, (rho, colour, label) in enumerate(
                ((0.0, C_RHO0, "GEP.jl, full grid (ρ = 0)"),
                 (0.5, C_RHO5, "GEP.jl, half the samples (ρ = 0.5)"))):
            ys = [100 * runs[(s, rho)][task] if (s, rho) in runs else np.nan
                  for s in sigmas]
            # a white edge keeps the two columns of a group apart
            ax.bar(x + (k - 0.5) * w, ys, width=w, color=colour, edgecolor="white",
                   linewidth=1.0, label=label, zorder=2)
        if task == "rec":
            first = True
            for i, s in enumerate(sigmas):
                if s in ODEFORMER_RECON_RHO0:
                    v = 100 * ODEFORMER_RECON_RHO0[s]
                    xc = x[i] - 0.5 * w
                    ax.plot([xc - 0.62 * w, xc + 0.62 * w], [v, v], color=C_ODEF,
                            lw=1.6, solid_capstyle="round", zorder=3)
                    ax.plot([xc], [v], marker="D", ms=4.5, color=C_ODEF, mew=0,
                            ls="", zorder=4,
                            label="ODEFormer, ρ = 0 (quoted, arXiv:2602.08733)"
                            if first else None)
                    first = False
        else:
            ax.text(0.98, 0.98, "no retrievable ODEFormer reference\nfor this task",
                    transform=ax.transAxes, fontsize=6.5, color=MUTED,
                    ha="right", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(["%g" % s for s in sigmas])
        ax.set_xlabel("noise level σ")
        ax.set_title(title, loc="left", color=INK2)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("systems with R² > 0.9 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("ODEFormer"))
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.8,
               handlelength=1.4, columnspacing=1.6)
    n = next(iter(runs.values()))["n"]
    fig.suptitle(f"ODEBench ({n} systems): fraction with variance-weighted R² > 0.9",
                 x=0.02, y=0.995, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.subplots_adjust(top=0.84)     # tight_layout leaves a band under the headline
    out = os.path.join(HERE, "results", "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"odebench_accuracy.{ext}"))
    print("wrote results/figures/odebench_accuracy.{pdf,png}")

    # companion table
    print(f"\n{'sigma':>6} {'rho':>5} {'recon':>7} {'gen':>7} {'med s/system':>13}")
    for (s, r), v in sorted(runs.items()):
        print(f"{s:6.2f} {r:5.2f} {v['rec']:7.3f} {v['gen']:7.3f} {v['time']:13.1f}")


if __name__ == "__main__":
    main()
