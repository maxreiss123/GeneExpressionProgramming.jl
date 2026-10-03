"""Term recovery: does each method's model hold the right set of terms?

Functional recovery (R^2 of the fitted right-hand side) scores the operator's values; a
model can score well with spurious terms that cancel on the data, or badly with the
right terms under biased coefficients. This scores the structure instead.

Each recorded winner (results/*.json) is expanded into monomials of u, u_x, u_xx, ...
(the unit constants nu set to 1). A monomial's share is the root mean square of its
term, coefficient included, over the clean test points of its cell, divided by that of
the analytic right-hand side. A term counts as identified if its share is at least
tau; least squares gives every candidate term some weight, so without a threshold
nothing but an exact sparse fit could match. The identified set is compared with the
analytic terms: missing true terms (FN), spurious terms (FP), the exact-set count and
the true positivity ratio TPR = TP / (TP + FN + FP).

    python paper/pdebench/term_recovery.py [--tau 0.01]
"""

import argparse
import json
import os

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = (("GEP.jl", "gep.json"), ("GEP.jl (vector)", "gep_tensor.json"),
         ("SITE", "site.json"), ("PDE-FIND", "pdefind.json"),
         ("GEP.jl (vector+units)", "gep_units.json"), ("SITE (units)", "site_units.json"))
PDES = ("heat", "burgers", "kdv", "ks")
SIGMAS = (0.0, 0.01, 0.05)
TAUS = (0.001, 0.01, 0.05)
SYMS = {n: sp.Symbol(n) for n in ("u", "u_x", "u_xx", "u_xxx", "u_xxxx", "nu2", "nu3", "nu4")}


def monomials(expr):
    m = sp.expand(sp.sympify(expr.replace("^", "**"), locals=SYMS))
    m = sp.expand(m.subs({SYMS["nu2"]: 1, SYMS["nu3"]: 1, SYMS["nu4"]: 1}))
    return {k: float(v) for k, v in m.as_coefficients_dict().items() if float(v) != 0.0}


def shares(expr, cond):
    """Share of each monomial of `expr`: rms of its term over rms of the true RHS."""
    X = np.asarray(cond["X_test"], dtype=float)
    cols = {n: X[:, i] for i, n in enumerate(cond["names"])}
    rms_f = np.sqrt(np.mean(np.asarray(cond["rhs_test"], dtype=float) ** 2))
    out = {}
    for mono, coef in monomials(expr).items():
        syms = sorted(mono.free_symbols, key=str)
        v = sp.lambdify(syms, mono, "numpy")(*[cols[str(s)] for s in syms]) if syms \
            else np.ones(len(X))
        out[mono] = np.sqrt(np.mean((coef * v) ** 2)) / rms_f
    return out


def pretty(mono):
    return str(mono).replace("**", "^").replace("*", "·")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tau", type=float, default=0.01, help="threshold of the cell table")
    a = ap.parse_args()

    conds = {(c["pde"], float(c["sigma"])): c
             for c in json.load(open(os.path.join(HERE, "data", "conditions.json")))}
    meta = json.load(open(os.path.join(HERE, "data", "meta.json")))
    truth = {p: {sp.expand(sp.sympify(t, locals=SYMS)) for t in meta[p]["terms"]}
             for p in PDES}
    share = {}
    for label, fn in FILES:
        for r in json.load(open(os.path.join(HERE, "results", fn)))["results"]:
            key = (r["pde"], float(r["sigma"]))
            share[label, key] = shares(r["expression"], conds[key])

    def score(label, key, tau):
        T = truth[key[0]]
        found = {m for m, s in share[label, key].items() if s >= tau}
        return T - found, found - T, len(found & T)

    print(f"Term recovery per cell, share threshold {100 * a.tau:g} % "
          "(✓ = exactly the true terms; − missing true term; +n = n spurious terms, "
          "largest share in brackets)\n")
    print("| PDE | σ | " + " | ".join(l for l, _ in FILES) + " |")
    print("|---|---|" + "---|" * len(FILES))
    for pde in PDES:
        for s in SIGMAS:
            cells = []
            for label, _ in FILES:
                miss, spur, _ = score(label, (pde, s), a.tau)
                if not miss and not spur:
                    cells.append("✓")
                    continue
                parts = ["−" + pretty(m) for m in sorted(miss, key=str)]
                if spur:
                    top = max(share[label, (pde, s)][m] for m in spur)
                    parts.append(f"+{len(spur)} ({100 * top:.0f} %)" if top >= 0.095
                                 else f"+{len(spur)} ({100 * top:.1f} %)")
                cells.append(" ".join(parts))
            print(f"| {pde} | {s:g} | " + " | ".join(cells) + " |")

    print("\nExact term sets out of 12 cells (mean TPR), by share threshold\n")
    print("| method | " + " | ".join(f"{100 * t:g} %" for t in TAUS) + " |")
    print("|---|" + "---|" * len(TAUS))
    for label, _ in FILES:
        row = []
        for t in TAUS:
            exact, tpr = 0, []
            for key in conds:
                miss, spur, tp = score(label, key, t)
                exact += not miss and not spur
                tpr.append(tp / (tp + len(miss) + len(spur)))
            row.append(f"{exact} ({np.mean(tpr):.2f})")
        print(f"| {label} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
