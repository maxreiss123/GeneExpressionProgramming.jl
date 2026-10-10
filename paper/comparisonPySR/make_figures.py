"""Figures of the comparison in the SciencePlots `science` style: a vector PDF and a
300 dpi PNG each, in results/figures/.

    python make_figures.py

Reads results/summary<tag>.csv (judge.py) for the noise levels of common.NOISES,
results/summary_gep_nounits.csv (judge.py --runs-dir) and results/summary_front<tag>.csv
(judge_front.py).

* fig1_benchmark: per noise level, (a) symbolic solution rate, (b) accuracy solution
  rate (test R^2 > 0.999), both with 95 % Wilson intervals, (c) test error 1 - R^2 and
  (d) wall time per run, as box plots.
* fig2_per_equation: recovered seeds per equation, one panel per noise level.
* fig3_candidates: without noise, (a) the candidates each run evaluated against its
  wall time, with lines of equal throughput, and (b) the runs recovered per group of
  equations, for GEP-SBP, GEP-SBP without units and PySR.
* fig4_front: GEP-SBP with the whole budget and an accuracy-complexity front, against
  its noise-floor stop and PySR, on the four equations of judge_front.py.
* fig5_time_per_equation: median wall time per equation, one panel per noise level.

PySR is drawn in Paul Tol's vibrant orange next to GEP-SBP's blue (the pair passes the
dataviz palette checks for colour-vision deficiency and normal vision). SciencePlots'
LaTeX rendering is replaced by matplotlib's mathtext (its `no-latex` style), so no TeX
installation is needed.
"""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401  (registers the styles)
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

import common  # noqa: E402

RESULTS_DIR = common.RESULTS_DIR

logging.getLogger("fontTools").setLevel(logging.ERROR)   # font subsetting chatter
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlepad": 4, "hatch.linewidth": 0.6,
})

DOUBLE = 7.0                          # width of a two-column figure, in
METHODS = ["GEP-SBP", "PySR"]
COLOR = {"GEP-SBP": "#0077BB", "PySR": "#EE7733"}     # Paul Tol's vibrant blue and orange
LIGHT = {"GEP-SBP": "#99C9E6", "PySR": "#F7C2A3"}     # the same, lighter, for box faces
INK, INK2, RULE = "#222222", "#555555", "#CCCCCC"
GROUPS = ["monomial", "power law", "sum / transcendental"]
GROUP_NAME = {"monomial": "Monomials", "power law": "Power laws",
              "sum / transcendental": "Sums, exponential"}
FORMULA = {
    "I.14.3": r"$mgz$", "I.34.8": r"$qvB/p$", "I.43.16": r"$\mu qV/d$",
    "II.34.2a": r"$qv/(2\pi r)$", "III.21.20": r"$-\rho qA/m$",
    "I.12.2": r"$q_1q_2/(4\pi\epsilon r^2)$", "I.32.5": r"$q^2a^2/(6\pi\epsilon c^3)$",
    "I.38.12": r"$4\pi\epsilon\hbar^2/(mq^2)$", "II.13.17": r"$2I/(4\pi\epsilon c^2r)$",
    "III.15.14": r"$\hbar^2/(2Ed^2)$",
    "I.13.12": r"$Gm_1m_2(1/r_2-1/r_1)$", "II.2.42": r"$\kappa(T_2-T_1)A/d$",
    "II.11.3": r"$qE/(m(\omega_0^2-\omega^2))$", "II.21.32": r"$q/(4\pi\epsilon r(1-v/c))$",
    "III.14.14": r"$I_0(e^{qV/k_BT}-1)$",
}
ORDER = [e for e, _ in common.EQUATIONS]


def sigma(n):
    return r"$\sigma = %g$" % n if n else r"$\sigma = 0$"


def wilson(k, n, z=1.96):
    """95 % Wilson score interval of the proportion k/n."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def title(ax, label, text):
    ax.set_title(r"$\mathbf{(%s)}$ %s" % (label, text), loc="left")


def no_ticks(ax, axis):
    getattr(ax, axis + "axis").set_minor_locator(NullLocator())
    ax.tick_params(axis=axis, which="both", length=0)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


def legend_handles():
    return [Patch(facecolor=COLOR[m], edgecolor=COLOR[m], label=m) for m in METHODS]


# ---------------------------------------------------------------- figure 1

def rate_bars(ax, dfs, column, threshold=None):
    """Grouped bars of the share of runs that pass, per noise level, with intervals."""
    noises = sorted(dfs)
    width = 0.36
    for j, m in enumerate(METHODS):
        rates, lo, hi = [], [], []
        for n in noises:
            d = dfs[n][dfs[n].method == m]
            ok = d[column] > threshold if threshold is not None else d[column].astype(bool)
            k, cnt = int(ok.sum()), len(d)
            a, b = wilson(k, cnt)
            rates.append(100 * k / cnt)
            lo.append(100 * (k / cnt - a))
            hi.append(100 * (b - k / cnt))
        x = np.arange(len(noises)) + (j - 0.5) * (width + 0.03)
        ax.bar(x, rates, width, color=COLOR[m], zorder=2, label=m)
        ax.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6,
                    capsize=1.5, capthick=0.6, zorder=3)
        for xi, r, h in zip(x, rates, hi):
            ax.text(xi, r + h + 1.5, "%.0f" % r, ha="center", va="bottom", fontsize=6.5,
                    color=INK)
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([sigma(n) for n in noises])
    no_ticks(ax, "x")
    ax.tick_params(axis="x", top=False)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.6, len(noises) - 0.4)


def boxes(ax, dfs, values, log=True):
    noises = sorted(dfs)
    width = 0.32
    for j, m in enumerate(METHODS):
        data = [values(dfs[n][dfs[n].method == m]) for n in noises]
        x = np.arange(len(noises)) + (j - 0.5) * (width + 0.05)
        ax.boxplot(data, positions=x, widths=width, whis=(5, 95), patch_artist=True,
                   showfliers=True, manage_ticks=False,
                   boxprops=dict(facecolor=LIGHT[m], edgecolor=COLOR[m], lw=0.8),
                   medianprops=dict(color=COLOR[m], lw=1.4),
                   whiskerprops=dict(color=COLOR[m], lw=0.8),
                   capprops=dict(color=COLOR[m], lw=0.8),
                   flierprops=dict(marker="o", ms=1.8, mfc=COLOR[m], mec="none", alpha=0.6))
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([sigma(n) for n in noises])
    no_ticks(ax, "x")
    ax.tick_params(axis="x", top=False)
    ax.set_xlim(-0.6, len(noises) - 0.4)


def one_minus_r2(d):
    v = 1.0 - d.r2_test.to_numpy(dtype=float)
    v[~np.isfinite(v) | (v >= 1.0)] = 1.0             # R^2 <= 0 or no model
    return np.clip(v, 1e-16, None)


def fig_benchmark(dfs, out):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.4),
                             gridspec_kw=dict(hspace=0.45, wspace=0.22))
    (a, b), (c, d) = axes
    rate_bars(a, dfs, "recovered")
    a.set_ylabel("runs recovered (%)")
    title(a, "a", "Symbolic solution rate")
    rate_bars(b, dfs, "r2_test", threshold=0.999)
    b.set_ylabel(r"runs with test $R^2 > 0.999$ (%)")
    title(b, "b", "Accuracy solution rate")
    boxes(c, dfs, one_minus_r2)
    c.set_ylim(3e-17, 3)
    c.set_ylabel(r"$1 - R^2$ on the test set")
    title(c, "c", "Test error")
    boxes(d, dfs, lambda x: x.wall_time.to_numpy())
    d.set_ylabel("wall time per run (s)")
    title(d, "d", "Wall time (one CPU thread)")
    a.legend(handles=legend_handles(), loc="lower left", ncol=2,
             bbox_to_anchor=(0.0, 1.13), frameon=False)
    save(fig, out, "fig1_benchmark")


# ---------------------------------------------------------------- figures 2, 3

def per_equation(dfs, value, xlabel, log=False, err=None):
    noises = sorted(dfs)
    groups = dict(common.EQUATIONS)
    ypos, y, prev = {}, 0.0, None
    for e in ORDER:
        if prev is not None and groups[e] != prev:
            y += 0.6
        ypos[e] = y
        prev = groups[e]
        y += 1.0
    h = 0.38
    fig, axes = plt.subplots(1, len(noises), figsize=(DOUBLE, 4.3), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for i, (ax, n) in enumerate(zip(axes, noises)):
        for j, m in enumerate(METHODS):
            d = dfs[n][dfs[n].method == m]
            for e in ORDER:
                de = d[d.equation == e]
                v = value(de)
                yy = ypos[e] + (j - 0.5) * h
                ax.barh(yy, v, height=h * 0.92, color=COLOR[m], edgecolor=COLOR[m], lw=0,
                        zorder=2)
                if err is not None:
                    lo, hi = err(de)
                    ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK,
                                elinewidth=0.5, capsize=1.0, capthick=0.5, zorder=3)
                if not log and v == 0:
                    ax.text(0.15, yy, "0", va="center", ha="left", fontsize=5.5,
                            color=COLOR[m])
        title(ax, "abc"[i], sigma(n))
        ax.set_xlabel(xlabel)
        no_ticks(ax, "y")
        ax.tick_params(axis="y", right=False)
        if log:
            ax.set_xscale("log")
        for g in GROUPS[1:]:
            first = min(ypos[e] for e in ORDER if groups[e] == g)
            ax.axhline(first - 0.8, color=RULE, lw=0.5, zorder=0)
    axes[0].set_yticks([ypos[e] for e in ORDER])
    axes[0].set_yticklabels(["%s  %s" % (e, FORMULA[e]) for e in ORDER])
    axes[0].set_ylim(max(ypos.values()) + 0.7, -0.7)
    for g in GROUPS:
        first = min(ypos[e] for e in ORDER if groups[e] == g)
        axes[-1].text(1.02, first - 0.45, GROUP_NAME[g],
                      transform=axes[-1].get_yaxis_transform(), ha="left", va="center",
                      fontsize=6.5, color=INK2, style="italic")
    return fig, axes


def fig_per_equation(dfs, out):
    fig, axes = per_equation(dfs, lambda de: de.recovered.sum(),
                             "seeds recovered (of 10)")
    for ax in axes:
        ax.set_xlim(0, 10.5)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.xaxis.set_minor_locator(NullLocator())
    axes[1].legend(handles=legend_handles(), loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 1.06), frameon=False)
    save(fig, out, "fig2_per_equation")


def load():
    dfs = {}
    for n in common.NOISES:
        path = os.path.join(RESULTS_DIR, "summary%s.csv" % common.tag(n))
        if os.path.exists(path):
            dfs[n] = pd.read_csv(path)
    return dfs


def fig_time_per_equation(dfs, out):
    fig, axes = per_equation(
        dfs, lambda de: de.wall_time.median(), "median wall time per run (s)",
        log=True, err=lambda de: (de.wall_time.quantile(0.25), de.wall_time.quantile(0.75)))
    for ax in axes:
        ax.set_xlim(0.2, 60)          # off the decades, so the panels' tick labels keep apart
    axes[1].legend(handles=legend_handles(), loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 1.06), frameon=False)
    save(fig, out, "fig5_time_per_equation")


FRONT_SERIES = [  # label, method, filled, hatch
    ("PySR", "PySR", True, None),
    ("PySR, any model of its hall of fame", "PySR", False, None),
    ("GEP-SBP, noise-floor stop", "GEP-SBP", True, None),
    ("GEP-SBP, whole budget: final model", "GEP-SBP", True, "////"),
    ("GEP-SBP, whole budget: parsimony pick", "GEP-SBP", True, "...."),
    ("GEP-SBP, whole budget: any model of its front", "GEP-SBP", False, None),
]


def fig_front(dfs, out):
    fronts = {}
    for n in sorted(dfs):
        path = os.path.join(RESULTS_DIR, "summary_front%s.csv" % common.tag(n))
        if n > 0 and os.path.exists(path):
            fronts[n] = pd.read_csv(path)
    if not fronts:
        return
    noises = sorted(fronts)
    eqs = [e for e in ORDER if any(e in set(f.equation) for f in fronts.values())]
    fig, axes = plt.subplots(1, len(noises), figsize=(DOUBLE, 2.6), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    width = 0.13
    for i, (ax, n) in enumerate(zip(axes, noises)):
        f, sr = fronts[n], dfs[n][dfs[n].method == "PySR"]
        gep = dfs[n][dfs[n].method == "GEP-SBP"]
        for j, (label, m, filled, hatch) in enumerate(FRONT_SERIES):
            vals = []
            for e in eqs:
                pe, fe, ge = sr[sr.equation == e], f[f.equation == e], gep[gep.equation == e]
                vals.append([pe.recovered.sum(), pe.recovered_any.sum(), ge.recovered.sum(),
                             fe.final.sum(), fe.pick.sum(), fe["any"].sum()][j])
            x = np.arange(len(eqs)) + (j - 2.5) * (width + 0.012)
            ax.bar(x, vals, width, color=COLOR[m] if filled else "white",
                   edgecolor="white" if (filled and hatch) else COLOR[m], hatch=hatch,
                   lw=0 if (filled and hatch) else 0.8, zorder=2, label=label)
            for xi, v in zip(x, vals):
                if v == 0:
                    ax.text(xi, 0.15, "0", ha="center", va="bottom", fontsize=5.5,
                            color=COLOR[m])
        ax.set_xticks(range(len(eqs)))
        ax.set_xticklabels(eqs)
        no_ticks(ax, "x")
        ax.tick_params(axis="x", top=False)
        ax.set_xlim(-0.55, len(eqs) - 0.45)
        ax.set_ylim(0, 10.8)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        title(ax, "ab"[i], sigma(n))
    axes[0].set_ylabel("seeds recovered (of 10)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.09),
                   frameon=False, fontsize=6.5, columnspacing=1.2, handlelength=1.6)
    save(fig, out, "fig4_front")


CANDIDATE_SERIES = [  # label, method colour, filled
    ("GEP-SBP", "GEP-SBP", True),
    ("GEP-SBP without units", "GEP-SBP", False),
    ("PySR", "PySR", True),
]


def fig_candidates(dfs, out):
    """Noise-free runs: (a) candidates used against wall time, with lines of equal
    throughput; (b) runs recovered per group of equations."""
    path = os.path.join(RESULTS_DIR, "summary_gep_nounits.csv")
    if 0.0 not in dfs or not os.path.exists(path):
        return
    d = dfs[0.0]
    runs = {"GEP-SBP": d[d.method == "GEP-SBP"], "GEP-SBP without units": pd.read_csv(path),
            "PySR": d[d.method == "PySR"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9),
                               gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.3))
    # (a) runs stopping at the same candidate count spread a little to the side, so that
    # their number shows; a fixed generator keeps the figure the same on every run
    rng = np.random.default_rng(0)
    for label, m, filled in CANDIDATE_SERIES:
        r = runs[label]
        x = r.evaluations.to_numpy(float) * 10 ** rng.uniform(-0.02, 0.02, len(r))
        a.scatter(x, r.wall_time, s=9, lw=0.6, zorder=3, label=label, alpha=0.75,
                  facecolor=COLOR[m] if filled else "white", edgecolor=COLOR[m])
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1.5e3, 1.5e6)
    a.set_ylim(0.03, 300)
    fig.canvas.draw()                  # the axes' final size, for the labels' angle
    xs = np.array([1.5e3, 1.5e6])
    # lines of equal throughput (candidates per second), each labelled above itself
    # where no run lies: past the data for 30 000, near the top for 1 000
    for rate, x0, text in [(1e3, 6e4, "1 000/s"), (3e4, 7e5, "30 000/s")]:
        a.plot(xs, xs / rate, color=RULE, lw=0.8, zorder=1)
        p, q = a.transData.transform([(x0, x0 / rate), (10 * x0, 10 * x0 / rate)])
        angle = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
        a.text(x0, x0 / rate * 1.25, text, rotation=angle, fontsize=6, color=INK2,
               ha="center", va="bottom", rotation_mode="anchor")
    a.set_xlabel("candidates evaluated in the run")
    a.set_ylabel("wall time per run (s)")
    title(a, "a", r"Candidates and time, $\sigma = 0$")
    a.legend(loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.3)
    # (b)
    width = 0.26
    for j, (label, m, filled) in enumerate(CANDIDATE_SERIES):
        r = runs[label]
        vals = [int(r[r.group == g].recovered.sum()) for g in GROUPS]
        x = np.arange(len(GROUPS)) + (j - 1) * (width + 0.02)
        b.bar(x, vals, width, color=COLOR[m] if filled else "white",
              edgecolor=COLOR[m], lw=0 if filled else 0.8, zorder=2, label=label)
        for xi, v in zip(x, vals):
            b.text(xi, v + 1, "%d" % v, ha="center", va="bottom", fontsize=6, color=INK)
    b.set_xticks(range(len(GROUPS)))
    b.set_xticklabels([GROUP_NAME[g] for g in GROUPS])
    no_ticks(b, "x")
    b.tick_params(axis="x", top=False)
    b.set_ylim(0, 58)
    b.set_yticks([0, 10, 20, 30, 40, 50])
    b.set_ylabel("runs recovered (of 50)")
    title(b, "b", r"Symbolic recovery, $\sigma = 0$")
    save(fig, out, "fig3_candidates")


def main():
    out = os.path.join(RESULTS_DIR, "figures")
    dfs = load()
    fig_benchmark(dfs, out)
    fig_per_equation(dfs, out)
    fig_candidates(dfs, out)
    fig_front(dfs, out)
    fig_time_per_equation(dfs, out)


if __name__ == "__main__":
    main()
