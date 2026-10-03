"""
Paper-ready figures for the GeneExpressionProgramming.jl vs SITE comparison
(arXiv:2507.01466v1, "Symbolic identification of tensor equations in
multidimensional physical fields").

Reads the JSON produced by run_all.sh from `results/` and writes vector PDF plus
300 dpi PNG into `results/figures/`.

    python make_figures.py [--results results] [--outdir results/figures]
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------- style --
# categorical slots, fixed order, never cycled (validated: worst adjacent CVD
# dE 9.1, worst adjacent normal-vision dE 19.6 on a light surface).  Every series is
# also named by a row label, a tick label or a marker, so identity never rests on colour
# alone -- which is also what a greyscale print of the figure needs.
C_GEP = "#2a78d6"      # slot 1  blue    GeneExpressionProgramming.jl (+ DHC)
C_GEP2 = "#eb6834"     # slot 2  orange  GeneExpressionProgramming.jl (no DHC)
C_SITE1 = "#1baf7a"    # slot 3  aqua    SITE  TLR + RNC
C_SITE2 = "#eda100"    # slot 4  yellow  SITE  TLR only
C_SITE3 = "#e87ba4"    # slot 5  magenta SITE  RNC only
C_SHADOW = "#8a8f98"   # slot 6  grey    tensor-native path
C_SCALE = "#008300"    # slot 7  green   linear scaling (fitted coefficients)
C_ALLOC_BEFORE = "#86b6ef"   # blue 250  } one hue, two steps, for a before/after
C_ALLOC_AFTER = "#1c5cab"    # blue 550  } pair (validated as an ordinal ramp)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9892"

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
TOL = 1e-6

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": "#e3e2de",
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})


def finish(ax):
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s.pdf / .png" % name)


# ---------------------------------------------------------------------- data --
def load(results):
    out = {"gep": [], "site": {}, "shadow": [], "tensor": [], "batched": [],
           "reynolds": None, "precompile": None, "dsmc": None, "memory": None}
    for path in sorted(glob.glob(os.path.join(results, "shadow_*.json"))):
        with open(path) as f:
            out["shadow"].append(json.load(f))
    for path in sorted(glob.glob(os.path.join(results, "tensor_gep_*.json"))):
        with open(path) as f:
            out["tensor"].append(json.load(f))
    for path in sorted(glob.glob(os.path.join(results, "tensor_batched_*.json"))):
        with open(path) as f:
            out["batched"].append(json.load(f))
    for path in sorted(glob.glob(os.path.join(results, "gep_*.json"))):
        with open(path) as f:
            out["gep"].append(json.load(f))
    for path in sorted(glob.glob(os.path.join(results, "site_*.json"))):
        with open(path) as f:
            d = json.load(f)
        key = os.path.basename(path)[len("site_"):-len(".json")]
        out["site"][key] = d
    rp = os.path.join(results, "reynolds_gep.json")
    if os.path.exists(rp):
        out["reynolds"] = json.load(open(rp))
    pp = os.path.join(results, "precompile.json")
    if os.path.exists(pp):
        out["precompile"] = json.load(open(pp))
    mp = os.path.join(results, "memory.json")
    if os.path.exists(mp):
        out["memory"] = json.load(open(mp))
    dp = os.path.join(results, "dsmc_gep.json")
    if os.path.exists(dp):
        out["dsmc"] = json.load(open(dp))
    return out


def select_shadow(shadow, config="dhc", data="clean", scaling=False, units=False):
    """Runs of the tensor-native path. The four clean configurations differ only in
    `linear_scaling` / `si_units`, so all four have to be pinned or the selection mixes
    the fitted-coefficient runs into the plain ones."""
    return sorted([r for r in shadow
                   if r["config"] == config and r.get("data", "clean") == data
                   and bool(r.get("linear_scaling", False)) == scaling
                   and bool(r.get("si_units", False)) == units],
                  key=lambda r: r["seed"])


def select(gep, data="clean", config="dhc", constopt=False, scaling=False):
    runs = [r for r in gep if r["data"] == data and r["config"] == config
            and bool(r.get("constant_optimisation", False)) == constopt
            and bool(r.get("linear_scaling", False)) == scaling]
    return sorted(runs, key=lambda r: r["seed"])


def maxwell_basis(datadir):
    """The four ground-truth terms of the Maxwell stress tensor, on the clean data."""
    rows = np.genfromtxt(os.path.join(datadir, "maxwell_clean.csv"), delimiter=",",
                         names=True)
    EE, BB, d = rows["EE"], rows["BB"], rows["delta"]
    E2, B2, T = rows["E2"], rows["B2"], rows["T"]
    basis = np.column_stack([EPS0 * EE, EPS0 * E2 * d, BB / MU0, B2 * d / MU0])
    return basis, T


def effective_coefficients(y_pred, basis):
    """Project a discovered model onto the ground-truth terms (truth: 1, -.5, 1, -.5)."""
    w, *_ = np.linalg.lstsq(basis, np.asarray(y_pred, dtype=float), rcond=None)
    return w


TRUTH = np.array([1.0, -0.5, 1.0, -0.5])


# ------------------------------------------------------------------ figure 1 --
def seed_offsets(n, step=0.12):
    """Small vertical offsets, so the seeds of one row stay apart where they coincide."""
    return (np.arange(n) - (n - 1) / 2) * step


def fig_convergence(d, outdir):
    """Generations and wall-clock to the tolerance, for every seed of each configuration.

    One row per configuration and one dot per seed. A filled dot reached the tolerance
    (1e-6) at that generation and time; a hollow one stopped at the generation cap
    without it. The tick is the median over the seeds that reached it. The count beside
    each row is how many seeds did; SITE ran once per configuration.
    """
    def gep(r):
        return (r["converged_epoch"] if r["converged"] else r["epochs_run"],
                r["solve_time_s"], bool(r["converged"]))

    def site(r):
        return (r["converged_generation"] if r["converged"] else r["generations_run"],
                r["wall_time_s"], bool(r["converged"]))

    sh = d["shadow"]
    spec = [
        ("GEP.jl tensor, SI units + fitted coeff.", C_GEP,
         select_shadow(sh, scaling=True, units=True), gep),
        ("GEP.jl scalar, dim. check + linear scaling", C_GEP,
         select(d["gep"], scaling=True), gep),
        ("GEP.jl tensor, order check + fitted coeff.", C_GEP,
         select_shadow(sh, scaling=True), gep),
        ("GEP.jl scalar, dimensional check", C_GEP, select(d["gep"]), gep),
        ("GEP.jl tensor, order check", C_GEP, select_shadow(sh), gep),
        ("GEP.jl tensor, no check", C_GEP, select_shadow(sh, config="nodhc"), gep),
        ("GEP.jl scalar, no dimensional check", C_GEP, select(d["gep"], config="nodhc"),
         gep),
    ] + [(label, C_SITE1, [d["site"][key]] if key in d["site"] else [], site)
         for key, label in (("tlr_and_rnc", "SITE, TLR + RNC"),
                            ("only_tlr", "SITE, TLR only"),
                            ("only_rnc", "SITE, RNC only"))]
    rows = [(label, colour, [norm(r) for r in runs])
            for label, colour, runs, norm in spec if runs]
    if not rows:
        return

    # top to bottom, with a gap between this package and the reference implementation
    ys, y = [], 0.0
    for label, colour, _ in rows:
        if colour == C_SITE1 and ys and rows[len(ys) - 1][1] != C_SITE1:
            y += 0.6
        ys.append(y)
        y += 1.0
    ys = -np.asarray(ys)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, k, title, lim in ((axes[0], 0, r"(a) generations to reach $10^{-6}$",
                               (0.7, 4e3)),
                              (axes[1], 1, r"(b) seconds to reach $10^{-6}$",
                               (0.08, 2e3))):
        for yy, (label, colour, pts) in zip(ys, rows):
            v = np.array([p[k] for p in pts], dtype=float)
            ok = np.array([p[2] for p in pts])
            off = yy + seed_offsets(len(v))
            # no surface ring on the filled dots: on coincident seeds it would read
            # as a hollow dot, which here means the tolerance was not reached
            ax.scatter(v[ok], off[ok], s=24, color=colour, linewidth=0, zorder=3)
            ax.scatter(v[~ok], off[~ok], s=24, facecolor="white", edgecolor=colour,
                       linewidth=1.2, zorder=3)
            if ok.sum() >= 2:
                m = float(np.median(v[ok]))
                ax.plot([m, m], [yy - 0.34, yy + 0.34], color=INK, lw=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.set_title(title, loc="left", color=INK2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        finish(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(["%s  %d/%d" % (label, sum(p[2] for p in pts), len(pts))
                             for label, _, pts in rows], fontsize=6.8)
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, color=C_GEP, mew=0,
               label="GEP.jl, one seed"),
        Line2D([], [], ls="", marker="o", ms=5, color=C_SITE1, mew=0,
               label="SITE, one run"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=INK2, mew=1.2,
               label=r"stopped at the cap, $10^{-6}$ not reached"),
        Line2D([], [], ls="", marker="|", ms=8, mew=1.2, color=INK,
               label="median of the seeds that reached it"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               handletextpad=0.3, columnspacing=1.4)
    fig.suptitle("Maxwell stress tensor on the paper's 150 samples: every seed, with the "
                 "count that reached the tolerance", x=0.01, y=0.995, ha="left",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.subplots_adjust(top=0.87)     # tight_layout leaves a band under the headline
    save(fig, outdir, "fig1_convergence")


# ------------------------------------------------------------------ figure 2 --
def fig_cost(d, outdir):
    """Time to solution, and where the Julia wall-clock actually goes."""
    dhc = select(d["gep"])
    conv = [r for r in dhc if r["converged"]]
    if not dhc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))

    # --- (a) time to solution -------------------------------------------------
    rows = []

    def fitted_tensor(units, label):
        runs = [r for r in select_shadow(d["shadow"], scaling=True, units=units)
                if r["converged"]]
        if runs:
            med = float(np.median([r["solve_time_s"] for r in runs]))
            gen = int(np.median([r["converged_epoch"] for r in runs]))
            rows.append((label, med, "%d gen" % gen, C_SCALE, "here"))

    fitted_tensor(True, "GEP.jl tensor +units +scaling (here)")
    ls_conv = [r for r in select(d["gep"], scaling=True) if r["converged"]]
    if ls_conv:
        med = float(np.median([r["solve_time_s"] for r in ls_conv]))
        gen = int(np.median([r["converged_epoch"] for r in ls_conv]))
        rows.append(("GEP.jl +DHC +scaling (here)", med, "%d gen" % gen, C_SCALE, "here"))
    fitted_tensor(False, "GEP.jl tensor +check +scaling (here)")
    if conv:
        med = float(np.median([r["solve_time_s"] for r in conv]))
        gen = int(np.median([r["converged_epoch"] for r in conv]))
        rows.append(("GEP.jl +DHC (here)", med, "%d gen" % gen, C_GEP, "here"))
    for cfg, lbl in (("dhc", "GEP.jl tensor +check (here)"),
                     ("nodhc", "GEP.jl tensor, no check (here)")):
        runs = select_shadow(d["shadow"], config=cfg)
        conv = [r for r in runs if r["converged"]]
        if not runs:
            continue
        med = float(np.median([r["solve_time_s"] for r in (conv or runs)]))
        tag = ("%d gen" % int(np.median([r["converged_epoch"] for r in conv]))) if conv \
            else "%d gen, no conv." % int(np.median([r["epochs_run"] for r in runs]))
        rows.append((lbl, med, tag, C_SHADOW, "here"))
    nodhc_conv = [r for r in select(d["gep"], config="nodhc") if r["converged"]]
    if nodhc_conv:
        med = float(np.median([r["solve_time_s"] for r in nodhc_conv]))
        gen = int(np.median([r["converged_epoch"] for r in nodhc_conv]))
        rows.append(("GEP.jl no DHC (here)", med, "%d gen" % gen, C_GEP2, "here"))
    for key, color, label in (("tlr_and_rnc", C_SITE1, "SITE TLR+RNC (here)"),
                              ("only_tlr", C_SITE2, "SITE TLR only (here)"),
                              ("only_rnc", C_SITE3, "SITE RNC only (here)")):
        r = d["site"].get(key)
        if r:
            tag = ("%d gen" % r["converged_generation"]) if r["converged"] else \
                  ("%d gen, no conv." % r["generations_run"])
            rows.append((label, r["wall_time_s"], tag, color, "here"))
    # the paper's own numbers (Table 1, i9-13900K)
    for label, t, tag, color in (("SITE TLR+RNC (paper)", 20, "23 gen", C_SITE1),
                                 ("SITE TLR only (paper)", 580, "714 gen", C_SITE2),
                                 ("SITE RNC only (paper)", 1666, "2000 gen, no conv.", C_SITE3)):
        rows.append((label, t, tag, color, "paper"))

    ax = axes[0]
    ypos = np.arange(len(rows))[::-1]
    for y, (label, t, tag, color, origin) in zip(ypos, rows):
        ax.barh(y, t, height=0.62, color=color, alpha=1.0 if origin == "here" else 0.35,
                edgecolor=color, linewidth=0.8,
                hatch=None if origin == "here" else "///")
        ax.text(t * 1.15, y, tag, va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3e4)
    ax.set_xlabel("time to reported solution (s)")
    ax.set_title("(a) cost of one identification", loc="left", color=INK2)
    finish(ax)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=6, label="measured here (4 threads, container)"),
        Line2D([], [], color=MUTED, lw=6, alpha=0.35, label="reported in the paper (i9-13900K)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.5)

    # --- (b) Julia wall-clock breakdown --------------------------------------
    ax = axes[1]
    pre = d["precompile"] or {}

    def num(v):
        """JSON nulls (Julia NaN) come back as None."""
        return np.nan if v is None else float(v)

    solve_runs = [r for r in dhc if r["converged"]] or dhc
    parts = [
        ("deps\npre-\ncompile", num(pre.get("deps_precompile_s")), "#c9c8c2"),
        ("pkg\npre-\ncompile", num(pre.get("package_precompile_s")), "#a8a7a1"),
        ("pkg\nload", float(np.median([r["load_time_s"] for r in dhc])), "#7d7c76"),
        ("JIT\nwarm-up", float(np.median([r["warmup_time_s"] for r in dhc])), "#52514e"),
        ("solve\n(timed)", float(np.median([r["solve_time_s"] for r in solve_runs])), C_GEP),
    ]
    xs = np.arange(len(parts))
    vals = [p[1] for p in parts]
    ax.bar(xs, [0.0 if not np.isfinite(v) else v for v in vals],
           color=[p[2] for p in parts], width=0.66)
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.text(x, v * 1.12, "%.1f s" % v, ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in parts], fontsize=5.2)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max([v for v in vals if np.isfinite(v)] + [1]) * 4)
    ax.set_ylabel("seconds")
    ax.set_title("(b) where the Julia time goes\n(one-off costs vs. the timed solve)",
                 loc="left", color=INK2, fontsize=8)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig2_cost")


# ------------------------------------------------------------------ figure 3 --
def fig_noise(d, outdir, datadir):
    """The coefficients the scalar search with the dimensional check recovers, per noise
    level (mean and sd over the seeds). How their error compares with the other
    configurations and with SITE is figure 9."""
    basis, _ = maxwell_basis(datadir)
    levels = [("clean", 0.0), ("noise005", 5.0), ("noise010", 10.0), ("noise020", 20.0)]
    per_coeff = {}
    for tag, pct in levels:
        runs = [r for r in select(d["gep"], data=tag) if r.get("y_pred_clean")]
        if runs:
            per_coeff[pct] = np.array([effective_coefficients(r["y_pred_clean"], basis)
                                       for r in runs])
    if not per_coeff:
        return

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    names = [r"$\varepsilon_0 E_iE_j$", r"$\varepsilon_0 E_kE_k\delta_{ij}$",
             r"$\mu_0^{-1}B_iB_j$", r"$\mu_0^{-1}B_kB_k\delta_{ij}$"]
    width = 0.2
    idx = np.arange(4)
    shades = ["#bfd6f2", "#7fadea", "#4a8fdf", C_GEP]
    for k, pct in enumerate(sorted(per_coeff)):
        cc = per_coeff[pct]
        ax.bar(idx + (k - 1.5) * width, cc.mean(axis=0), width=width * 0.9,
               yerr=cc.std(axis=0), color=shades[k], capsize=1.8,
               error_kw=dict(elinewidth=0.7, ecolor=INK2),
               label="%g%% noise" % pct)
    for k, t in enumerate(TRUTH):
        ax.plot([k - 0.44, k + 0.44], [t, t], color=INK, lw=1.0, ls=(0, (2, 1.6)),
                zorder=5)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("identified coefficient")
    ax.set_title("Recovered coefficients, scalar path with the dimensional check "
                 "(dashed: truth)", loc="left", color=INK2, fontsize=8)
    ax.set_ylim(top=1.35)
    ax.legend(loc="upper center", ncol=4, fontsize=6.5, handlelength=1.2,
              columnspacing=1.0)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig3_noise")


# ------------------------------------------------------------------ figure 4 --
def fig_parity(d, outdir, datadir):
    runs = [r for r in select(d["gep"]) if r.get("y_pred_clean")]
    if not runs:
        return
    best = min(runs, key=lambda r: r["clean_data_loss"])
    rows = np.genfromtxt(os.path.join(datadir, "maxwell_clean.csv"), delimiter=",",
                         names=True)
    T = rows["T"]
    pred = np.asarray(best["y_pred_clean"], dtype=float)
    diag = rows["delta"] > 0.5

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lim = [min(T.min(), pred.min()) * 1.05, max(T.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.scatter(T[~diag], pred[~diag], s=7, color=C_GEP, alpha=0.75, linewidths=0,
               label="off-diagonal $T_{ij},\\,i\\neq j$", zorder=3)
    ax.scatter(T[diag], pred[diag], s=7, color=C_GEP2, alpha=0.75, linewidths=0,
               label="diagonal $T_{ii}$", zorder=2)
    ax.set_xlabel(r"reference $T_{ij}$  (Pa)")
    ax.set_ylabel(r"identified $\hat{T}_{ij}$  (Pa)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title("GEP.jl, best of %d seeds ($\\mathcal{L}=%.1e$)"
                 % (len(runs), best["clean_data_loss"]), loc="left", color=INK2)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig4_parity")


# ------------------------------------------------------------------ figure 5 --
def fig_reynolds(d, outdir):
    r = d["reynolds"]
    if not r:
        return
    sizes = [100, 75, 50, 25]
    mean = np.array([r["summary"][str(s)]["mean"] for s in sizes])
    std = np.array([r["summary"][str(s)]["std"] for s in sizes])
    paper_mean = np.array([-0.6617] * 4)
    paper_std = np.array([0.0, 1e-4, 2e-4, 4e-4])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(4)
    ax = axes[0]
    ax.axhline(-2 / 3, color=MUTED, lw=0.8, ls=(0, (2, 1.6)), zorder=1)
    ax.text(3.45, -2 / 3 + 0.00025, r"$-2/3$", color=INK2, fontsize=6.8, ha="right")
    ax.errorbar(x + 0.1, paper_mean, yerr=paper_std, ls="", marker="D", ms=5,
                color=C_SITE1, elinewidth=1.2, capsize=0,
                label="SITE (paper, Table 3; its own data)", zorder=3)
    ax.scatter(x - 0.1, mean, s=30, color=C_GEP, edgecolor="white", linewidth=0.7,
               zorder=3, label="GEP.jl +DHC (reconstructed data)")
    ax.set_xticks(x)
    ax.set_xticklabels(["100\n(100%)", "75\n(75%)", "50\n(50%)", "25\n(25%)"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("data points used")
    ax.set_ylabel(r"identified coefficient of $\varepsilon\,\delta_{ij}$")
    ax.set_title("(a) sub-sampling stability (mean, bar: sd)", loc="left", color=INK2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", fontsize=6.5)
    finish(ax)

    ax = axes[1]
    ax.scatter(x - 0.1, np.where(std > 0, std, np.nan), s=30, color=C_GEP,
               edgecolor="white", linewidth=0.7, zorder=3, label="GEP.jl +DHC")
    ax.scatter(x[1:] + 0.1, paper_std[1:], s=30, marker="D", color=C_SITE1,
               edgecolor="white", linewidth=0.7, zorder=3, label="SITE (paper)")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-2)
    ax.set_xticks(x)
    ax.set_xticklabels(["100", "75", "50", "25"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("data points used")
    ax.set_ylabel("sd of the coefficient over runs")
    ax.set_title("(b) run-to-run scatter", loc="left", color=INK2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", fontsize=6.5)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig5_reynolds")


# ------------------------------------------------------------------ figure 6 --
def fig_reliability(d, outdir):
    dhc, nodhc = select(d["gep"]), select(d["gep"], config="nodhc")
    scaled = select(d["gep"], scaling=True)
    if not dhc or not nodhc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax = axes[0]
    groups = [("dimensional check\n+ linear scaling", scaled, C_SCALE),
              ("dimensional check", dhc, C_GEP),
              ("neither", nodhc, C_GEP2)] if scaled else \
        [("with dimensional\nhomogeneity check", dhc, C_GEP), ("without", nodhc, C_GEP2)]
    for k, (label, runs, color) in enumerate(groups):
        rate = 100.0 * sum(r["converged"] for r in runs) / len(runs)
        ax.bar(k, rate, width=0.5, color=color)
        ax.text(k, rate + 2, "%d of %d seeds" % (sum(r["converged"] for r in runs), len(runs)),
                ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel(r"runs reaching $\mathcal{L}<10^{-6}$  (%)")
    ax.set_title("(a) reliability over seeds", loc="left", color=INK2)
    finish(ax)

    ax = axes[1]
    for k, (label, runs, color) in enumerate(groups):
        vals = [r["clean_data_loss"] for r in runs]
        ax.scatter(np.full(len(vals), k) + np.linspace(-0.09, 0.09, len(vals)), vals,
                   s=22, color=color, alpha=0.85, linewidths=0)
        ax.plot([k - 0.2, k + 0.2], [np.median(vals)] * 2, color=INK, lw=1.2)
    ax.axhline(TOL, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["+scaling", "+DHC", "no DHC"][:len(groups)] if scaled
                       else ["+DHC", "no DHC"], fontsize=7)
    ax.set_ylabel(r"final loss on clean data")
    ax.set_title("(b) final loss per seed (bar: median)", loc="left", color=INK2)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig6_reliability")


# ------------------------------------------------------------------ figure 7 --
def fig_throughput(d, outdir):
    """Candidate expressions per second of wall-clock: population x generations / solve
    time (1600 per generation assumed for SITE), so selection and variation count too."""
    rows = []

    def rate(runs, key_pop="population"):
        vals = [r[key_pop] * r["epochs_run"] / r["solve_time_s"] for r in runs
                if r.get("solve_time_s", 0) > 0]
        return float(np.median(vals)) if vals else None

    r = rate(d["batched"] or select_shadow(d["shadow"]))
    if r:
        rows.append(("GEP.jl tensor\n(batched, order check)", r, C_SCALE))
    r = rate(select(d["gep"]))
    if r:
        rows.append(("GEP.jl scalar\n(stacked, dim. check)", r, C_GEP))
    site = d["site"].get("tlr_and_rnc")
    if site:
        rows.append(("SITE\n(geppy)",
                     1600 * site["generations_run"] / site["wall_time_s"], C_SITE1))
    r = rate(d["tensor"])
    if r:
        rows.append(("GEP.jl tensor\n(as released)", r, "#c9c8c2"))
    if len(rows) < 2:
        return

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], color=[r[2] for r in rows], width=0.62)
    for x, r in zip(xs, rows):
        ax.text(x, r[1] * 1.15, "%s/s" % ("%.3g" % r[1]), ha="center", fontsize=6.8,
                color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1.0, min(r[1] for r in rows) / 6),
                top=max(r[1] for r in rows) * 6)
    ax.set_ylabel("expressions evaluated per second")
    ax.set_title("Evaluation throughput on the 150-sample Maxwell case",
                 loc="left", color=INK2, fontsize=8)
    finish(ax)
    fig.tight_layout()
    save(fig, outdir, "fig7_throughput")



# ------------------------------------------------------------------ figure 8 --
def fig_scaling_tradeoff(d, outdir):
    """Why fitting the coefficients is cheaper overall despite costing more per step.

    Linear scaling evaluates every gene separately and solves a small least-squares
    system, and the scaled runs mostly stop after their first generation, the one that
    also builds and repairs the initial population (gene by gene under scaling). Panel
    (a) divides each run's time by the generations it ran, so for those runs it is that
    first generation; the search without scaling spreads it over a hundred more.
    """
    plain = [r for r in select(d["gep"]) if r["converged"]]
    scaled = [r for r in select(d["gep"], scaling=True) if r["converged"]]
    if not plain or not scaled:
        return

    def per_gen_ms(runs):
        return [1000 * r["solve_time_s"] / max(r["epochs_run"], 1) for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    groups = [("without\nscaling", plain, C_GEP), ("with\nscaling", scaled, C_SCALE)]

    panels = [
        ("(a) time per generation run", "milliseconds",
         [per_gen_ms(r) for _, r, _ in groups]),
        ("(b) generations to converge", "generations",
         [[r["converged_epoch"] for r in runs] for _, runs, _ in groups]),
        ("(c) time to solution", "seconds",
         [[r["solve_time_s"] for r in runs] for _, runs, _ in groups]),
    ]
    for ax, (title, ylabel, series) in zip(axes, panels):
        xs = np.arange(len(groups))
        meds = [float(np.median(v)) for v in series]
        ax.bar(xs, meds, color=[g[2] for g in groups], width=0.6)
        for x, v, m in zip(xs, series, meds):
            # every seed, so the reader sees the spread behind the median
            ax.scatter(np.full(len(v), x) + 0.16, v, s=9, color=INK2, zorder=3,
                       alpha=0.75)
            ax.text(x - 0.16, m, "%.0f" % m if m >= 10 else "%.1f" % m, ha="center",
                    va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([g[0] for g in groups], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK2, fontsize=8)
        ax.set_ylim(bottom=0)
        finish(ax)

    ratio = (np.median([r["converged_epoch"] for r in plain])
             / np.median([r["converged_epoch"] for r in scaled]),
             np.median([r["solve_time_s"] for r in plain])
             / np.median([r["solve_time_s"] for r in scaled]))
    fig.suptitle("Linear scaling: %.0fx fewer generations, %.1fx less time to solution"
                 % ratio, x=0.005, ha="left", fontsize=8.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, outdir, "fig8_scaling_tradeoff")



# ------------------------------------------------------------------ figure 9 --
def fig_noise_paths(d, outdir, datadir):
    """Coefficient error under noise, one panel per noise level.

    One row per configuration of this package, a dot per seed and a tick at the mean;
    SITE is the paper's Table 2 (TLR + RNC), its mean as a diamond and one sd either
    side as a line, cut at zero.
    """
    basis, _ = maxwell_basis(datadir)

    def errs(runs):
        return np.array([100 * np.mean(np.abs(
            (effective_coefficients(r["y_pred_clean"], basis) - TRUTH) / TRUTH))
            for r in runs if r.get("y_pred_clean")])

    configs = [
        ("GEP.jl scalar, dim. check", lambda tag: select(d["gep"], data=tag)),
        ("GEP.jl scalar, dim. check\n+ linear scaling",
         lambda tag: select(d["gep"], data=tag, scaling=True)),
        ("GEP.jl tensor, SI units\n+ fitted coeff.",
         lambda tag: [r for r in d["shadow"] if r.get("data") == tag
                      and r.get("si_units", False)]),
    ]
    site = {"noise005": (0.25, 0.27), "noise010": (0.45, 0.55), "noise020": (1.53, 1.40)}
    levels = [("noise005", "5 % noise"), ("noise010", "10 % noise"),
              ("noise020", "20 % noise")]
    if not any(len(errs(pick(tag))) for _, pick in configs for tag, _ in levels):
        return

    ys = [3.0, 2.0, 1.0, -0.2]           # the three configurations, then SITE
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (tag, title) in zip(axes, levels):
        top = 0.0
        for yy, (_, pick) in zip(ys, configs):
            e = errs(pick(tag))
            if not len(e):
                continue
            ax.scatter(e, yy + seed_offsets(len(e)), s=20, color=C_GEP, linewidth=0,
                       zorder=3)
            m = float(e.mean())
            ax.plot([m, m], [yy - 0.32, yy + 0.32], color=INK, lw=1.2, zorder=4)
            top = max(top, float(e.max()))
        m, s = site[tag]
        ax.plot([max(0.0, m - s), m + s], [ys[-1]] * 2, color=C_SITE1, lw=1.4,
                solid_capstyle="round", zorder=2)
        ax.scatter([m], [ys[-1]], s=26, marker="D", color=C_SITE1, linewidth=0,
                   zorder=3)
        right = max(top, m + s) * 1.08
        ax.set_xlim(-0.03 * right, right)     # dots at zero stay whole
        ax.set_title(title, loc="left", color=INK2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        finish(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c[0] for c in configs] + ["SITE, TLR + RNC\n(paper)"],
                            fontsize=6.8)
    axes[1].set_xlabel("mean relative error of the four coefficients (%)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, color=C_GEP, mew=0, label="one seed"),
        Line2D([], [], ls="", marker="|", ms=8, mew=1.2, color=INK,
               label="mean over the seeds"),
        Line2D([], [], color=C_SITE1, lw=1.4, marker="D", ms=5, mew=0,
               label="SITE: mean and sd as published"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Recovering the coefficients under noise", x=0.01, y=0.995, ha="left",
                 fontsize=8.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.subplots_adjust(top=0.86)     # tight_layout leaves a band under the headline
    save(fig, outdir, "fig9_noise_paths")


# ----------------------------------------------------------------- figure 10 --
def fig_speed_memory(d, outdir):
    """The speed case, and what it costs in memory.

    Two different comparisons, deliberately kept apart. Throughput against the reference
    implementation is measured on the same machine and task. Allocation against
    DynamicExpressions compares two evaluators within Julia, the only place where bytes
    are directly comparable.
    """
    mem = d.get("memory")
    if not mem:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))

    # --- (a) throughput -------------------------------------------------------
    ax = axes[0]

    def rate(runs):
        v = [r["population"] * r["epochs_run"] / r["solve_time_s"] for r in runs
             if r.get("solve_time_s", 0) > 0]
        return float(np.median(v)) if v else None

    rows = []
    r = rate(d["batched"])
    r and rows.append(("GEP.jl tensor\n(batched, order check)", r, C_SCALE))
    r = rate(select(d["gep"]))
    r and rows.append(("GEP.jl scalar\n(stacked, dim. check)", r, C_GEP))
    site = d["site"].get("tlr_and_rnc")
    if site:
        rows.append(("SITE\n(geppy)", 1600 * site["generations_run"] / site["wall_time_s"],
                     C_SITE1))
    if rows:
        xs = np.arange(len(rows))
        ax.bar(xs, [x[1] for x in rows], color=[x[2] for x in rows], width=0.6)
        for x, row in zip(xs, rows):
            ax.text(x, row[1] * 1.25, "%.3g/s" % row[1], ha="center", fontsize=6.8,
                    color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([x[0] for x in rows], fontsize=6.8)
        ax.set_yscale("log")
        ax.set_ylim(top=max(x[1] for x in rows) * 8)
        ax.set_ylabel("expressions evaluated per second")
        ax.set_title("(a) throughput, same machine", loc="left", color=INK2, fontsize=8)
        finish(ax)

    # --- (b) allocation against the data size: before and after, per data size -----
    ax = axes[1]
    a = mem["allocation_per_fit_mb"]
    before = np.asarray(a["dynamic_expressions"], dtype=float)
    after = np.asarray(a["batched_buffers"], dtype=float)
    x = np.arange(len(a["samples"]))
    for xi, b, f in zip(x, before, after):
        ax.plot([xi, xi], [f, b], color="#d3d2cc", lw=2.2, solid_capstyle="round",
                zorder=1)
        ax.text(xi + 0.13, np.sqrt(b * f), "÷%.0f" % (b / f) if b / f >= 10
                else "÷%.1f" % (b / f), va="center", fontsize=6.8, color=INK2)
    ax.scatter(x, before, s=34, color=C_ALLOC_BEFORE, edgecolor="white", linewidth=0.8,
               zorder=3, label="DynamicExpressions (earlier)")
    ax.scatter(x, after, s=34, color=C_ALLOC_AFTER, edgecolor="white", linewidth=0.8,
               zorder=3, label="batched buffers (now)")
    ax.set_xticks(x)
    ax.set_xticklabels(["{:,}".format(n).replace(",", " ") for n in a["samples"]])
    ax.set_xlim(-0.45, len(x) - 0.35)
    ax.set_yscale("log")
    ax.set_xlabel("samples")
    ax.set_ylabel("MB allocated per fit")
    ax.set_title("(b) allocation, scalar path", loc="left", color=INK2, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.5)
    finish(ax)

    fig.tight_layout()
    save(fig, outdir, "fig10_speed_memory")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--outdir", default=os.path.join(here, "results", "figures"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = load(args.results)
    print("loaded %d GEP.jl runs, %d SITE runs" % (len(d["gep"]), len(d["site"])))
    fig_convergence(d, args.outdir)
    fig_cost(d, args.outdir)
    fig_noise(d, args.outdir, args.data)
    fig_parity(d, args.outdir, args.data)
    fig_reynolds(d, args.outdir)
    fig_reliability(d, args.outdir)
    fig_throughput(d, args.outdir)
    fig_scaling_tradeoff(d, args.outdir)
    fig_noise_paths(d, args.outdir, args.data)
    fig_speed_memory(d, args.outdir)


if __name__ == "__main__":
    main()
