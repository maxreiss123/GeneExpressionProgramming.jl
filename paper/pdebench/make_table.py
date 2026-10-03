"""Summary tables for the PDE-discovery benchmark, as markdown: per (pde, sigma) cell, the
functional-recovery R^2 of each method's fitted right-hand side on the clean test
features (bold = recovered), then each method's wall-clock per cell over the 12 cells,
with every GEP route on 4 threads and on one (results/*_1thread.json).
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDES = ("heat", "burgers", "kdv", "ks")
SIGMAS = (0.0, 0.01, 0.05)
# without units, then with
METHODS = (("GEP.jl", "gep.json"), ("GEP.jl (vector)", "gep_tensor.json"),
           ("SITE", "site.json"), ("PDE-FIND", "pdefind.json"),
           ("GEP.jl (vector+units)", "gep_units.json"), ("SITE (units)", "site_units.json"))
# (label, file, threads); SITE and PDE-FIND run in one process, serially
TIMED = (("GEP.jl", "gep.json", 4), ("GEP.jl", "gep_1thread.json", 1),
         ("GEP.jl (vector)", "gep_tensor.json", 4),
         ("GEP.jl (vector)", "gep_tensor_1thread.json", 1),
         ("GEP.jl (vector+units)", "gep_units.json", 4),
         ("GEP.jl (vector+units)", "gep_units_1thread.json", 1),
         ("SITE", "site.json", 1), ("SITE (units)", "site_units.json", 1),
         ("PDE-FIND", "pdefind.json", 1))


def load(fn):
    p = os.path.join(HERE, "results", fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {(r["pde"], float(r["sigma"])): r for r in d["results"]}


def r2_cell(r):
    if r is None or r.get("r2_test") is None:
        return "—"
    mark = "**" if r["recovered"] else ""
    return f"{mark}{r['r2_test']:.4f}{mark}"


def seconds(t):
    return f"{t:.2f}" if t < 1 else f"{t:.1f}"


def table(methods, fmt):
    print("| PDE | σ | " + " | ".join(methods) + " |")
    print("|---|---|" + "---|" * len(methods))
    for pde in PDES:
        for s in SIGMAS:
            print(f"| {pde} | {s:g} | " + " | ".join(fmt(m.get((pde, s))) for m in methods.values()) + " |")


def main():
    methods = {label: d for label, fn in METHODS if (d := load(fn))}

    print("Functional-recovery R² (bold = R² > 0.99 on the clean right-hand side)\n")
    table(methods, r2_cell)
    print("\nRecovered: " + ", ".join(
        f"{label} {sum(1 for r in m.values() if r.get('recovered'))}/{len(m)}"
        for label, m in methods.items()))

    print("\nWall-clock per cell (s), over the 12 cells\n")
    print("| Method | Threads | Median | Range | Total |")
    print("|---|---|---|---|---|")
    for label, fn, threads in TIMED:
        m = load(fn)
        if m is None:
            continue
        t = [r["time_s"] for r in m.values()]
        print(f"| {label} | {threads} | {seconds(np.median(t))} | "
              f"{seconds(min(t))}–{seconds(max(t))} | {seconds(sum(t))} |")


if __name__ == "__main__":
    main()
