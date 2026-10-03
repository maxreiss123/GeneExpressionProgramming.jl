"""
Turn the raw benchmark JSON into the markdown tables used in README.md.

    python summarize.py [--results results] > results/summary.md
"""

import argparse
import glob
import json
import os

import numpy as np

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
TRUTH = np.array([1.0, -0.5, 1.0, -0.5])
TERMS = ["eps0*E_iE_j", "eps0*E_kE_k*d_ij", "B_iB_j/mu0", "B_kB_k*d_ij/mu0"]


def load(results):
    gep, site, shadow, tensor = [], {}, [], []
    for p in sorted(glob.glob(os.path.join(results, "shadow_*.json"))):
        shadow.append(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(results, "tensor_gep_*.json"))):
        tensor.append(json.load(open(p)))
    batched = []
    for p in sorted(glob.glob(os.path.join(results, "tensor_batched_*.json"))):
        batched.append(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(results, "gep_*.json"))):
        gep.append(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(results, "site_*.json"))):
        site[os.path.basename(p)[5:-5]] = json.load(open(p))
    rey = None
    if os.path.exists(os.path.join(results, "reynolds_gep.json")):
        rey = json.load(open(os.path.join(results, "reynolds_gep.json")))
    pre = None
    if os.path.exists(os.path.join(results, "precompile.json")):
        pre = json.load(open(os.path.join(results, "precompile.json")))
    dsmc = None
    if os.path.exists(os.path.join(results, "dsmc_gep.json")):
        dsmc = json.load(open(os.path.join(results, "dsmc_gep.json")))
    return gep, site, rey, pre, shadow, tensor, dsmc, batched


def basis(datadir):
    r = np.genfromtxt(os.path.join(datadir, "maxwell_clean.csv"), delimiter=",", names=True)
    return np.column_stack([EPS0 * r["EE"], EPS0 * r["E2"] * r["delta"],
                            r["BB"] / MU0, r["B2"] * r["delta"] / MU0]), r["T"]


def coeffs(pred, B):
    w, *_ = np.linalg.lstsq(B, np.asarray(pred, float), rcond=None)
    return w


def sel(gep, data="clean", config="dhc", constopt=False, scaling=False):
    return sorted([r for r in gep if r["data"] == data and r["config"] == config
                   and bool(r.get("constant_optimisation", False)) == constopt
                   and bool(r.get("linear_scaling", False)) == scaling],
                  key=lambda r: r["seed"])


def fmt(v, spec="%.3g"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec % v


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--data", default=os.path.join(here, "data"))
    args = ap.parse_args()
    gep, site, rey, pre, shadow, tensor, dsmc, batched = load(args.results)
    B, T = basis(args.data)

    out = []
    out.append("## Maxwell stress tensor -- per-run results\n")
    out.append("| framework | data | dim. check | lin. scaling | seed | generations | "
               "converged | solve time (s) | final loss |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(gep, key=lambda r: (r["data"], r["config"],
                                        bool(r.get("linear_scaling", False)), r["seed"])):
        out.append("| GEP.jl | %s | %s | %s | %d | %d | %s | %.1f | %.2e |" % (
            r["data"], "yes" if r["dimensional_check"] else "no",
            "yes" if r.get("linear_scaling", False) else "no", r["seed"],
            r["epochs_run"], "yes" if r["converged"] else "no",
            r["solve_time_s"], r["final_loss"]))
    for k, r in sorted(site.items()):
        out.append("| SITE (%s) | %s | yes | TLR | 0 | %d | %s | %.1f | %.2e |" % (
            k, "clean" if r.get("noise") is None else "noise %g" % r["noise"],
            r["generations_run"], "yes" if r["converged"] else "no",
            r["wall_time_s"], r["best_loss"]))

    out.append("\n## Time to solution (same machine, 4 CPU cores)\n")
    out.append("| configuration | generations | wall-clock (s) | reached 1e-6 |")
    out.append("|---|---|---|---|")
    for label, runs in (("GEP.jl, dimensional check", sel(gep)),
                        ("**GEP.jl, dimensional check + linear scaling**",
                         sel(gep, scaling=True)),
                        ("GEP.jl, no dimensional check", sel(gep, config="nodhc"))):
        if not runs:
            continue
        conv = [r for r in runs if r["converged"]]
        gens = [r["converged_epoch"] for r in conv] or [r["epochs_run"] for r in runs]
        secs = [r["solve_time_s"] for r in (conv or runs)]
        out.append("| %s | %s (median of %d) | %s | %d/%d seeds |" % (
            label, fmt(float(np.median(gens)), "%.0f"), len(runs),
            fmt(float(np.median(secs)), "%.1f"), len(conv), len(runs)))
    for k in ("tlr_and_rnc", "only_tlr", "only_rnc"):
        if k in site:
            r = site[k]
            out.append("| SITE, %s | %d | %.1f | %s |" % (
                k.replace("_", " "),
                r["converged_generation"] if r["converged"] else r["generations_run"],
                r["wall_time_s"], "yes" if r["converged"] else "no"))
    for label, runs in (("**GEP.jl tensor, SI units + fitted coefficients**",
                         [r for r in shadow if r["config"] == "dhc"
                          and r.get("si_units", False) and r.get("data") == "clean"]),
                        ("**GEP.jl tensor, order check + fitted coefficients**",
                         [r for r in shadow if r["config"] == "dhc"
                          and r.get("linear_scaling", False)
                          and not r.get("si_units", False)
                          and r.get("data") == "clean"]),
                        ("GEP.jl tensor, order check",
                         [r for r in shadow if r["config"] == "dhc"
                          and not r.get("linear_scaling", False)
                          and not r.get("si_units", False)
                          and r.get("data") == "clean"]),
                        ("GEP.jl tensor, no check",
                         [r for r in shadow if r["config"] == "nodhc"])):
        if not runs:
            continue
        conv = [r for r in runs if r["converged"]]
        gens = [r["converged_epoch"] for r in conv] or [r["epochs_run"] for r in runs]
        secs = [r["solve_time_s"] for r in (conv or runs)]
        out.append("| %s | %s (median of %d) | %s | %d/%d seeds |" % (
            label, fmt(float(np.median(gens)), "%.0f"), len(runs),
            fmt(float(np.median(secs)), "%.1f"), len(conv), len(runs)))
    out.append("| SITE TLR+RNC, *paper* (i9-13900K) | 23 | 20 | yes |")
    out.append("| SITE TLR only, *paper* | 714 | 580 | yes |")
    out.append("| SITE RNC only, *paper* | 2000 | 1666 | no |")

    if pre:
        out.append("\n## Julia runtime decomposition\n")
        out.append("| stage | seconds | paid |")
        out.append("|---|---|---|")
        out.append("| dependency precompilation | %s | once per environment |"
                   % fmt(pre.get("deps_precompile_s"), "%.0f"))
        out.append("| package precompilation | %s | once per source change |"
                   % fmt(pre.get("package_precompile_s"), "%.1f"))
        out.append("| `using GeneExpressionProgramming` | %s | once per process |"
                   % fmt(pre.get("load_s"), "%.1f"))
        out.append("| first `fit!` (JIT) | %s | once per process |" % fmt(pre.get("ttfx_s"), "%.1f"))
        out.append("| second `fit!` (same call) | %s | steady state |"
                   % fmt(pre.get("second_fit_s"), "%.2f"))

    out.append("\n## Identified coefficients (projection on the ground-truth terms)\n")
    out.append("| data | " + " | ".join(TERMS) + " | mean rel. error |")
    out.append("|---|" + "---|" * 5)
    variants = [("clean", "clean, all seeds", False, False),
                ("clean", "clean, converged seeds", True, False),
                ("noise005", "5% noise", False, False),
                ("noise010", "10% noise", False, False),
                ("noise020", "20% noise", False, False),
                ("noise005", "**5% noise, linear scaling**", False, True),
                ("noise010", "**10% noise, linear scaling**", False, True),
                ("noise020", "**20% noise, linear scaling**", False, True)]
    tensor_variants = [("noise005", "**5% noise, tensor + SI units**"),
                       ("noise010", "**10% noise, tensor + SI units**"),
                       ("noise020", "**20% noise, tensor + SI units**")]
    for tag, label, only_conv, scaling in variants:
        runs = [r for r in sel(gep, data=tag, scaling=scaling) if r.get("y_pred_clean")]
        if only_conv:
            runs = [r for r in runs if r["converged"]]
        if not runs:
            continue
        cc = np.array([coeffs(r["y_pred_clean"], B) for r in runs])
        err = 100 * np.mean(np.abs((cc - TRUTH) / TRUTH), axis=1)
        cells = ["%+.4f ± %.4f" % (m, s) for m, s in zip(cc.mean(0), cc.std(0))]
        out.append("| %s (n=%d) | %s | %.2f ± %.2f %% |" % (
            label, len(runs), " | ".join(cells), err.mean(), err.std()))
    for tag, label in tensor_variants:
        runs = [r for r in shadow if r.get("data") == tag and r.get("si_units", False)
                and r.get("y_pred_clean")]
        if not runs:
            continue
        cc = np.array([coeffs(r["y_pred_clean"], B) for r in runs])
        err = 100 * np.mean(np.abs((cc - TRUTH) / TRUTH), axis=1)
        cells = ["%+.4f ± %.4f" % (m, sd) for m, sd in zip(cc.mean(0), cc.std(0))]
        out.append("| %s (n=%d) | %s | %.2f ± %.2f %% |" % (
            label, len(runs), " | ".join(cells), err.mean(), err.std()))
    out.append("| ground truth | +1.0000 | -0.5000 | +1.0000 | -0.5000 | -- |")
    out.append("| SITE, paper Table 2, 5% noise | +0.999 | -0.500 | +1.007 | -0.499 | 0.25 ± 0.27 % |")
    out.append("| SITE, paper Table 2, 10% noise | +0.998 | -0.500 | +1.014 | -0.501 | 0.45 ± 0.55 % |")
    out.append("| SITE, paper Table 2, 20% noise | +0.997 | -0.500 | +1.032 | -0.513 | 1.53 ± 1.40 % |")

    out.append("\n## Evaluation throughput (candidate expressions per second)\n")
    out.append("| implementation | expressions/s |")
    out.append("|---|---|")

    def rate(runs):
        v = [r["population"] * r["epochs_run"] / r["solve_time_s"] for r in runs
             if r.get("solve_time_s", 0) > 0]
        return float(np.median(v)) if v else None
    for label, runs in (("**GEP.jl tensor (batched, preallocated)**",
                         batched or [r for r in shadow if r["config"] == "dhc"
                                     and not r.get("linear_scaling", False)
                                     and not r.get("si_units", False)
                                     and r.get("data") == "clean"]),
                        ("GEP.jl scalar (stacked, batched buffers)", sel(gep)),
                        ("GEP.jl tensor *as released*, one call per sample", tensor)):
        v = rate(runs)
        if v:
            out.append("| %s | %s |" % (label, fmt(v, "%.0f")))
    if "tlr_and_rnc" in site:
        r = site["tlr_and_rnc"]
        out.append("| SITE (geppy + numpy) | %s |"
                   % fmt(1600 * r["generations_run"] / r["wall_time_s"], "%.0f"))

    if dsmc:
        out.append("\n## Constitutive relation from the DSMC cavity data (paper Sec. 4)\n")
        out.append("| case | GEP.jl | paper, Table 4 |")
        out.append("|---|---|---|")
        for case in ("incompressible", "compressible"):
            if case not in dsmc["summary"]:
                continue
            e = dsmc["summary"][case]
            got = ("%+.3f mu S_ij %+.3f mu D_kk d_ij %+.3f p d_ij"
                   % (e["coeff_muS"]["mean"], e["coeff_muDkk"]["mean"], e["coeff_p"]["mean"]))
            out.append("| %s (lid %s m/s) | %s | %s |" % (
                case, "50" if case == "incompressible" else "337", got,
                dsmc["paper_table4"][case]))

    if rey:
        out.append("\n## Reynolds-stress transport, sub-sampling study\n")
        out.append("| data points | GEP.jl coefficient | SITE (paper Table 3) |")
        out.append("|---|---|---|")
        paper = {100: "-0.6617", 75: "-(0.6617 ± 0.0001)", 50: "-(0.6617 ± 0.0002)",
                 25: "-(0.6617 ± 0.0004)"}
        for s in (100, 75, 50, 25):
            e = rey["summary"][str(s)]
            out.append("| %d (%d%%) | %+.4f ± %.1e (n=%d) | %s |"
                       % (s, s, e["mean"], e["std"], e["n"], paper[s]))
        times = [r["solve_time_s"] for r in rey["runs"]]
        gens = [r["epochs_run"] for r in rey["runs"]]
        out.append("\nMedian solve time %.2f s over %d runs (median %d generations); "
                   "reference value -2/3 = -0.6667."
                   % (float(np.median(times)), len(times), int(np.median(gens))))

    out.append("\n## Best expression found on the clean data\n")
    tensor_units = [r for r in shadow if r.get("si_units", False)
                    and r.get("data") == "clean" and r.get("y_pred_clean")]
    for label, pool in (("scalar path, dimensional check",
                         [r for r in sel(gep) if r.get("y_pred_clean")]),
                        ("tensor path, SI units + fitted coefficients", tensor_units)):
        best = min(pool, key=lambda r: r["clean_data_loss"], default=None)
        if not best:
            continue
        out.append("**%s** (seed %d, %d generations, loss %.2e)\n"
                   % (label, best["seed"], best["epochs_run"], best["clean_data_loss"]))
        out.append("```\n%s\n```\n" % best["expression"])
        w = coeffs(best["y_pred_clean"], B)
        out.append("projection on the ground-truth terms: " +
                   ", ".join("%s = %+.6f" % (t, v) for t, v in zip(TERMS, w)) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
