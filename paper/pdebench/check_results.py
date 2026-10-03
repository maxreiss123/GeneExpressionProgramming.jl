"""Independent check of the functional-recovery results in results/*.json.

1. Every method's recorded winner ('expression') is parsed, evaluated on the clean test
   features of its cell (the nu constants set to 1) and scored again against the analytic
   right-hand side; the recomputed R^2 is compared with the recorded one and the
   recovered flag with R^2 > 0.99. The expressions are printed to six significant digits
   (PDE-FIND's to four decimals), so small differences are expected.
2. The admissible-terms least-squares fit -- u*u_x, u_xx, u_xxx and, where the features go
   that far, u_xxxx, fitted on the first 80 % of the training rows like every method's
   coefficients -- is recomputed here, and the two unit-constrained winners, expanded
   into monomials, are compared with it coefficient by coefficient.

Exits non-zero if a recovered flag changes or a unit-constrained winner is not that fit.

    python paper/pdebench/check_results.py
"""

import json
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = (("GEP.jl", "gep.json"), ("GEP.jl (vector)", "gep_tensor.json"),
         ("SITE", "site.json"), ("PDE-FIND", "pdefind.json"),
         ("GEP.jl (vector+units)", "gep_units.json"), ("SITE (units)", "site_units.json"))
SYMS = {n: sp.Symbol(n) for n in ("u", "u_x", "u_xx", "u_xxx", "u_xxxx", "nu2", "nu3", "nu4")}
ADMISSIBLE = ("u*u_x", "u_xx", "u_xxx", "u_xxxx")


def r2(y, p):
    if not np.all(np.isfinite(p)):
        return -np.inf
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def parse(expr):
    model = sp.expand(sp.sympify(expr.replace("^", "**"), locals=SYMS))
    return sp.expand(model.subs({SYMS["nu2"]: 1, SYMS["nu3"]: 1, SYMS["nu4"]: 1}))


def columns(c, X):
    return {n: X[:, i] for i, n in enumerate(c["names"])}


def evaluate(model, cols, n):
    syms = sorted(model.free_symbols, key=str)
    if not syms:
        return float(model) * np.ones(n)
    return sp.lambdify(syms, model, "numpy")(*[cols[str(s)] for s in syms])


def monomial(term, cols):
    v = 1.0
    for f in term.split("*"):
        v = v * cols[f]
    return v


def main():
    conds = {(c["pde"], float(c["sigma"])): c
             for c in json.load(open(os.path.join(HERE, "data", "conditions.json")))}
    ok = True

    print("1. R^2 recomputed from each recorded expression\n")
    print("| method | max abs. difference to the recorded R^2 | recovered flags changed |")
    print("|---|---|---|")
    results = {}
    for label, fn in FILES:
        rows = json.load(open(os.path.join(HERE, "results", fn)))["results"]
        results[label] = {(r["pde"], float(r["sigma"])): r for r in rows}
        worst, flips = 0.0, 0
        for r in rows:
            c = conds[(r["pde"], float(r["sigma"]))]
            X = np.asarray(c["X_test"], dtype=float)
            rr = r2(np.asarray(c["rhs_test"], dtype=float),
                    evaluate(parse(r["expression"]), columns(c, X), len(X)))
            worst = max(worst, abs(rr - r["r2_test"]))
            flips += bool(rr > 0.99) != bool(r["recovered"])
        ok &= flips == 0
        print(f"| {label} | {worst:.1e} | {flips} |")

    print("\n2. The unit-constrained winners against the admissible-terms least-squares fit\n")
    print("| PDE | σ | R² of the fit | largest coefficient difference, GEP.jl / SITE |")
    print("|---|---|---|---|")
    for key, c in conds.items():
        Xtr = np.asarray(c["X_train"], dtype=float)
        ytr = np.asarray(c["y_train"], dtype=float)
        n = int(0.8 * len(ytr))
        terms = [t for t in ADMISSIBLE if all(f in c["names"] for f in t.split("*"))]
        cols = columns(c, Xtr[:n])
        w = np.linalg.lstsq(np.column_stack([monomial(t, cols) for t in terms]), ytr[:n],
                            rcond=None)[0]
        Xte = np.asarray(c["X_test"], dtype=float)
        cte = columns(c, Xte)
        fit = r2(np.asarray(c["rhs_test"], dtype=float),
                 sum(wi * monomial(t, cte) for wi, t in zip(w, terms)))
        diffs = []
        for label in ("GEP.jl (vector+units)", "SITE (units)"):
            model = parse(results[label][key]["expression"])
            mono = [sp.Mul(*[SYMS[f] for f in t.split("*")]) for t in terms]
            coef = [float(model.coeff(m)) for m in mono]
            rest = sp.expand(model - sum(a * m for a, m in zip(coef, mono)))
            d = max(abs(a - b) for a, b in zip(coef, w))
            # six printed digits: allow rounding relative to the largest coefficient
            ok &= rest == 0 and d <= 1e-5 * max(abs(w)) + 1e-7
            diffs.append(f"{d:.1e}" + ("" if rest == 0 else f" (+ {rest})"))
        print(f"| {key[0]} | {key[1]:g} | {fit:.6f} | {diffs[0]} / {diffs[1]} |")

    print("\n" + ("all checks pass" if ok else "CHECK FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
