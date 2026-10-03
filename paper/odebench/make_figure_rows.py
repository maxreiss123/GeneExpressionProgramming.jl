"""ODEBench results in the layout of the ODEFormer paper's method-comparison figure:
one row per method, sigma-coded markers, accuracy panels for rho = 0 and rho = 0.5,
then complexity and inference-time box plots.

Rows come from the local runs (this package, SINDy, and PySR when its results exist),
plus an ODEFormer row quoted from the FIM-ODE paper (arXiv:2602.08733; reconstruction,
rho = 0 only). The complexity and time panels show locally measured methods only. SITE
has no row: it was not run on ODEBench (the PDE benchmark in ../pdebench runs it).

Complexity is the token count of the returned expressions (operands plus operators, one
tokenizer for every local method; their output formats differ, so it is approximate).
Inference time is the per-system wall clock including candidate selection.
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

INK2 = "#52514e"
MUTED = "#9a9892"
SIGMA_COLORS = {0.0: "#3b3d8f", 0.01: "#8f3b8f", 0.02: "#c8506e", 0.05: "#e8a13c"}
SIGMA_MARKERS = {0.0: "|", 0.01: "3", 0.02: "4", 0.05: "+"}
BOX_COLORS = {"GEP.jl": "#2a78d6", "SINDy": "#1baf7a", "PySR": "#c8a13c"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})

ODEFORMER_QUOTED = {(0.0, 0.0): 0.631, (0.05, 0.0): 0.615}   # arXiv:2602.08733, recon


def tokens(expr):
    if not expr:
        return None
    return len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*(?:e-?\d+)?|[+\-*/^]", expr))


def load_method(pattern, expr_key, time_keys):
    conds = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        d = json.load(open(p))
        rs = d["results"]
        n = len(rs)
        acc = lambda k: sum(1 for r in rs if r[k] is not None and r[k] > 0.9) / n
        comp, times = [], []
        for r in rs:
            e = r.get(expr_key)
            if isinstance(e, list):
                e = " + ".join(e)
            c = tokens(e)
            if c:
                comp.append(c)
            times.append(sum(r[k] for k in time_keys))
        conds[(float(d["sigma"]), float(d["rho"]))] = dict(
            rec=acc("r2_reconstruction"), gen=acc("r2_generalization"),
            comp=comp, times=times)
    return conds


def main():
    methods = {}
    gep = load_method("gep_sigma*_seed1.json", "expressions",
                      ["fit_time_s", "selection_time_s"])
    if gep:
        methods["GEP.jl"] = gep
    sindy = load_method("sindy_sigma*_seed1.json", "expression", ["time_s"])
    if sindy:
        methods["SINDy"] = sindy
    pysr = load_method("pysr_sigma*_seed1.json", "expression", ["time_s"])
    if pysr:
        methods["PySR"] = pysr

    rows = list(methods.keys()) + ["ODEFormer"]
    ypos = {m: i for i, m in enumerate(reversed(rows))}

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 0.55 * len(rows) + 1.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9, 0.9]))

    sigmas = sorted({s for m in methods.values() for (s, _) in m})
    for ax, rho in ((axes[0], 0.0), (axes[1], 0.5)):
        for m, conds in methods.items():
            for s in sigmas:
                if (s, rho) in conds:
                    ax.plot(100 * conds[(s, rho)]["rec"], ypos[m],
                            SIGMA_MARKERS[s], color=SIGMA_COLORS[s], ms=9, mew=1.6)
        for (s, r), v in ODEFORMER_QUOTED.items():
            if r == rho:
                ax.plot(100 * v, ypos["ODEFormer"], SIGMA_MARKERS[s],
                        color=SIGMA_COLORS[s], ms=9, mew=1.6)
        if rho == 0.5:
            ax.text(0.97, ypos["ODEFormer"], "n/a", ha="right", va="center",
                    color=MUTED, fontsize=7,
                    transform=ax.get_yaxis_transform())
        ax.text(0.04, 0.96, f"ρ = {rho:g}", transform=ax.transAxes, va="top",
                fontsize=8, bbox=dict(fc="white", ec=INK2, lw=0.6, pad=2))
        ax.set_xlim(0, 100)
        ax.set_xlabel("% Accuracy (R² > 0.9)")

    for ax, key, label, log in ((axes[2], "comp", "complexity", True),
                                (axes[3], "times", "inference time [sec.]", True)):
        for m, conds in methods.items():
            vals = [v for c in conds.values() for v in c[key]]
            if not vals:
                continue
            bp = ax.boxplot([vals], positions=[ypos[m]], vert=False, widths=0.5,
                            whis=(0, 100), patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=BOX_COLORS.get(m, "#999"), alpha=0.75, lw=0.8)
            for el in ("whiskers", "caps", "medians"):
                for a in bp[el]:
                    a.set(color=INK2, lw=0.8)
        ax.text(0.96, ypos["ODEFormer"], "not run here", ha="right", va="center",
                color=MUTED, fontsize=7, transform=ax.get_yaxis_transform())
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)

    for i, ax in enumerate(axes):
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        if i == 0:
            labels = [r + (" *" if r == "ODEFormer" else "") for r in
                      [rows[len(rows) - 1 - k] for k in range(len(rows))]]
            ax.set_yticklabels(labels)
            for tick in ax.get_yticklabels():
                if "GEP" in tick.get_text():
                    tick.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], ls="", marker=SIGMA_MARKERS[s],
                          color=SIGMA_COLORS[s], ms=9, mew=1.6, label=f"σ={s:g}")
               for s in sigmas]
    fig.legend(handles=handles, ncol=len(sigmas), loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.01,
             "* ODEFormer: reconstruction accuracy as re-evaluated by the FIM-ODE paper "
             "(arXiv:2602.08733); ρ=0 only — its primary tables were not retrievable, and "
             "it was not run here (pretrained weights unreachable). Local methods: 63 systems, "
             "single seed, identical corruption, integrator and metric.",
             fontsize=6, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(HERE, "results", "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"odebench_methods.{ext}"))
    print("wrote results/figures/odebench_methods.{pdf,png}")

    for m, conds in methods.items():
        ts = [v for c in conds.values() for v in c["times"]]
        print(f"{m:8s} median time {np.median(ts):6.1f} s/system over {len(ts)} runs")


if __name__ == "__main__":
    main()
