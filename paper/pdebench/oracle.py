"""Reference fits for the PDE-discovery benchmark: the numbers RESULTS.md quotes.

Two least-squares fits per (pde, sigma) cell, on the noisy training features and target
every method gets -- the first 80 % of the training rows, the part the GEP harnesses fit
their gene weights on -- scored like every method (R^2 of the fitted right-hand side
against the analytic one on the clean test features):

    true terms        the terms of the analytic right-hand side, refitted; what the
                      features allow when the structure is known
    admissible terms  every monomial that has the dimension of u_t under the units of
                      pde_gep_units.jl: u*u_x, nu2*u_xx, nu3*u_xxx and nu4*u_xxxx, as far
                      as the cell's features go; the constants nu are columns of ones

Then the Kuramoto-Sivashinsky diagnostics: term scales, the true-terms coefficients per
noise level and spatial window, and PDE-FIND's sigma = 0.01 model with and without its
u*u_xxx term.

    python oracle.py
"""

import json
import os

import numpy as np

from pde_common import DATA, HERE, SIGMAS, make_condition, r2

PDES = ("heat", "burgers", "kdv", "ks")
ADMISSIBLE = ("u*u_x", "u_xx", "u_xxx", "u_xxxx")
KS_TERMS = ("u_xx", "u_xxxx", "u*u_x")


def terms(names, X, labels):
    """Columns of the feature products named in `labels` ('u_xx', 'u*u_x', ...)."""
    col = {n: X[:, i] for i, n in enumerate(names)}
    return np.stack([np.prod([col[f] for f in lab.split("*")], axis=0) for lab in labels],
                    axis=1)


def ls_fit(c, labels):
    """Least squares on `labels` from the fitting rows; coefficients and clean-test R^2."""
    n = int(0.8 * len(c["y_train"]))
    w = np.linalg.lstsq(terms(c["names"], c["X_train"][:n], labels), c["y_train"][:n],
                        rcond=None)[0]
    return w, r2(c["rhs_test"], terms(c["names"], c["X_test"], labels) @ w)


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    print("| PDE | σ | true terms | admissible terms |")
    print("|---|---|---|---|")
    for pde in PDES:
        for sigma in SIGMAS:
            c = make_condition(pde, sigma)
            adm = [t for t in ADMISSIBLE if t.split("*")[-1] in c["names"]]
            _, r_true = ls_fit(c, list(meta[pde]["terms"]))
            _, r_adm = ls_fit(c, adm)
            print(f"| {pde} | {sigma:g} | {r_true:.4f} | {r_adm:.4f} |")

    print("\nKuramoto-Sivashinsky, true terms " + ", ".join(KS_TERMS))
    clean = make_condition("ks", 0.0)
    sd = terms(clean["names"], clean["X_test"], KS_TERMS).std(axis=0)
    print("std on the clean test features: " +
          ", ".join(f"{lab} {s:.2f}" for lab, s in zip(KS_TERMS, sd)) +
          f"; right-hand side {clean['rhs_test'].std():.2f}")
    for sigma in (0.01, 0.05):
        w, r = ls_fit(make_condition("ks", sigma), KS_TERMS)
        print(f"sigma = {sigma}, shipped windows: w = {np.round(w, 2)}, R^2 = {r:.2f}")
    for win in (15, 21, 25):
        w, r = ls_fit(make_condition("ks", 0.01, spatial_window=win), KS_TERMS)
        print(f"sigma = 0.01, spatial window {win}: w = {np.round(w, 2)}, R^2 = {r:.2f}")

    # PDE-FIND's sigma = 0.01 model as recorded, then without its u*u_xxx term
    res = json.load(open(os.path.join(HERE, "results", "pdefind.json")))["results"]
    rec = next(r for r in res if r["pde"] == "ks" and r["sigma"] == 0.01)
    parts = [p.split("*", 1) for p in rec["expression"].split(" + ")]
    c = make_condition("ks", 0.01)
    for label, keep in (("as recorded", parts),
                        ("without u*u_xxx", [p for p in parts if p[1] != "u*u_xxx"])):
        pred = terms(c["names"], c["X_test"], [p[1] for p in keep]) @ \
            np.array([float(p[0]) for p in keep])
        print(f"PDE-FIND, sigma = 0.01, {label}: R^2 = {r2(c['rhs_test'], pred):.2f}")


if __name__ == "__main__":
    main()
