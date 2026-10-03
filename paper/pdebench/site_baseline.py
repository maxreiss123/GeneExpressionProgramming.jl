"""SITE on the PDE-discovery benchmark.

SITE, Symbolic Identification of Tensor Equations (T. Chen, H. Yang, W. Ma and J. Zhang,
J. Fluid Mech. 1024, A34, 2025; https://github.com/BUAA-MARS-group/SITE), evolves a host
chromosome over tensor terminals whose `p_` nodes take scalar plasmid chromosomes,
gives dimensionally inhomogeneous individuals a loss of 1000, and fits one coefficient
per host gene by tensor linear regression (TLR). This harness runs the released SITE.py,
imported unmodified from a clone of that repository, on the shared conditions
(data/conditions.json) with the protocol of the GEP harnesses: coefficients fitted on
the first 80 % of the training rows, the model picked from the hall of fame by R^2 on
the rest, and the pick scored by functional recovery on the clean test features.

A scalar PDE is SITE's one-component case: each feature enters the host as a 1x1 tensor
(U, Ux, Uxx, ...) and the plasmids as a scalar (u, ux, uxx, ...), next to the identity
`delta` as in SITE's own cases. The host has tensor addition, subtraction, inner product
and `p_`, the plasmids addition, subtraction and multiplication -- the {+, -, *} of the
GEP routes -- and random numerical constants (the authors' TLR + RNC configuration). The
other settings are those of the authors' Maxwell case: head lengths 5 (host) and 10
(plasmid), four host genes, one plasmid gene of 15 constants, tournaments of 200, 100
alien individuals, tolerance 1e-6, the same operator rates; population 1600 over 125
generations, the 200,000 candidates the GEP routes generate (1000 x 200).

With --units the dimensional check runs under the units of pde_gep_units.jl: u [m/s],
x [m], t [s], and the constants nu2 [m^2/s], nu3 [m^3/s], nu4 [m^4/s] as plasmid symbol
terminals of value 1; the target u_t is [m/s^2]. Without it every dimension is zero, the
authors' way of switching the check off.

The tensor operators are defined here, as each of SITE's case scripts defines its own.
The inner product is numpy's batched matmul rather than the scripts' loop over samples:
the same values, without a Python loop over 4000 rows per call. The recorded expression
is the fitted model expanded into monomials of the features, built in the order SITE's
regression evaluates the genes and checked against the scored predictions: SITE's own
printer (`my_compile`) can pair a gene's plasmids with the wrong p_ nodes when the gene
holds more than one.

    python site_baseline.py /path/to/SITE [--units]
"""

import argparse
import json
import operator
import os
import random
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R2_RECOVERED = 0.99
N_POP, N_GEN = 1600, 125
SEED = 1

# dimensions [m, s] for --units
UNIT_TARGET = [1, -2]                                    # u_t = m/s^2
UNIT_CONSTS = {"nu2": [2, -1], "nu3": [3, -1], "nu4": [4, -1]}


def tensor_add(*args):
    total = 0
    for a in args:
        total = total + a
    return total


def tensor_sub(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        return a - b
    return False


def tensor_inner_product(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        return np.matmul(a, b)
    return False


def p_(tensor):
    """Placeholder: SITE's evaluator supplies p_ (tensor times its plasmid scalar)."""


def r2(y_true, y_pred):
    if y_pred is None or not np.all(np.isfinite(y_pred)):
        return -np.inf
    sstot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - np.sum((y_true - y_pred) ** 2) / sstot if sstot > 0 else -np.inf


class Site:
    """SITE's modules and the classes and operators every cell shares."""

    def __init__(self, site_repo):
        sys.path.insert(0, os.path.join(site_repo, "demos"))
        import geppy
        from deap import base, creator, tools
        import SITE
        from simplification import linker_add, p_symbol
        self.gep, self.tools, self.SITE = geppy, tools, SITE
        self.linker_add = linker_add
        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Host_Individual", geppy.Chromosome, fitness=creator.FitnessMin,
                       plasmid=[])
        creator.create("Plasmid_Individual", geppy.Chromosome)
        creator.create("tinder", geppy.Chromosome, plasmid=[])
        self.creator = creator
        self.operators = {"tensor_add": tensor_add, "tensor_sub": tensor_sub,
                          "tensor_inner_product": tensor_inner_product,
                          "add": operator.add, "sub": operator.sub, "mul": operator.mul,
                          "linker_add": linker_add}
        self.symbolic = {"tensor_add": linker_add, "tensor_sub": operator.sub,
                         "tensor_inner_product": operator.mul, "p_": p_symbol,
                         "add": operator.add, "sub": operator.sub, "mul": operator.mul}

    def gene_terms(self, ind, variables):
        """Each host gene of `ind` evaluated on `variables`, as SITE.tlr evaluates them."""
        plasmids = list(self.SITE.extract_expressed_plasmid(ind))

        def p_eval(tensor):
            scalar = eval(str(plasmids.pop(0)), env)
            return scalar[:, None] * tensor if isinstance(scalar, np.ndarray) else tensor * scalar

        env = dict(self.operators)
        env.update(variables)
        env["p_"] = p_eval
        return [eval(str(gene), env) for gene in ind]

    def model(self, ind, weights, nonzero, n_features, units):
        """The fitted model as a sympy polynomial in u, u_x, ... (and nu2, nu3, nu4),
        built in the order SITE.tlr evaluates it. SITE's own printer (`my_compile`) can
        pair a gene's plasmids with the wrong p_ nodes when the gene holds more than one."""
        import sympy as sp
        plasmids = list(self.SITE.extract_expressed_plasmid(ind))

        def p_sym(tensor):
            return eval(str(plasmids.pop(0)), env) * tensor

        def total(*args):
            return sum(args[1:], args[0])

        env = {"tensor_add": total, "tensor_sub": operator.sub,
               "tensor_inner_product": operator.mul, "add": operator.add,
               "sub": operator.sub, "mul": operator.mul, "linker_add": total,
               "p_": p_sym, "delta": sp.Integer(1)}
        for o in range(n_features):
            env[host_name(o)] = env[plasmid_name(o)] = sp.Symbol(feature_name(o))
        if units:
            env.update({c: sp.Symbol(c) for c in UNIT_CONSTS})
        genes = [eval(str(gene), env) for gene in ind]
        return sp.expand(sum(float(weights[i, 0]) * genes[g] for i, g in enumerate(nonzero)))


def host_name(o):
    return "U" + "x" * o


def plasmid_name(o):
    return "u" + "x" * o


def feature_name(o):
    return "u" + ("_" + "x" * o if o else "")


def variables(X, units):
    v = {"delta": np.ones((len(X), 1, 1))}
    for o in range(X.shape[1]):
        v[host_name(o)] = X[:, o][:, None, None]
        v[plasmid_name(o)] = X[:, o][:, None]
    if units:
        v.update({c: 1.0 for c in UNIT_CONSTS})
    return v


def dimensions(n_features, units):
    if not units:
        dims = {"delta": [0]}
        for o in range(n_features):
            dims[host_name(o)] = dims[plasmid_name(o)] = [0]
        return dims, 1, [0]
    dims = {"delta": [0, 0], **UNIT_CONSTS}
    for o in range(n_features):
        dims[host_name(o)] = dims[plasmid_name(o)] = [1 - o, -1]   # d^o u/dx^o: m^(1-o)/s
    return dims, 2, UNIT_TARGET


def make_toolbox(site, n_features, fit_vars, y_fit, units):
    gep, tools, SITE, creator = site.gep, site.tools, site.SITE, site.creator
    host = gep.PrimitiveSet("Host", input_names=["delta"] + [host_name(o) for o in range(n_features)])
    host.add_function(tensor_add, 2)
    host.add_function(tensor_sub, 2)
    host.add_function(tensor_inner_product, 2)
    host.add_function(p_, 1)
    plasmid = gep.PrimitiveSet("Plasmid", input_names=[plasmid_name(o) for o in range(n_features)])
    if units:
        for c in UNIT_CONSTS:
            plasmid.add_symbol_terminal(c, 1.0)
    plasmid.add_rnc_terminal()
    plasmid.add_function(operator.add, 2)
    plasmid.add_function(operator.sub, 2)
    plasmid.add_function(operator.mul, 2)

    dims, n_units, target = dimensions(n_features, units)
    tb = gep.Toolbox()
    tb.register("host_gene_gen", gep.Gene, pset=host, head_length=5)
    tb.register("host_individual", creator.Host_Individual, gene_gen=tb.host_gene_gen,
                n_genes=4, linker=tensor_add)
    tb.register("host_population", tools.initRepeat, list, tb.host_individual)
    tb.register("tinder_individual", creator.tinder, gene_gen=tb.host_gene_gen, n_genes=1,
                linker=tensor_add)
    tb.register("rnc_gen", lambda a, b: round(random.uniform(a, b), 5), a=-10, b=10)
    tb.register("plasmid_gene_gen", gep.GeneDc, pset=plasmid, head_length=10,
                rnc_gen=tb.rnc_gen, rnc_array_length=15)
    tb.register("plasmid_individual", creator.Plasmid_Individual,
                gene_gen=tb.plasmid_gene_gen, n_genes=1, linker=site.linker_add)
    tb.register("plasmid_population", SITE.plasmid_generate, tb.plasmid_individual)
    tb.register("compile", SITE.my_compile, dict_of_operators=site.operators,
                symbolic_function_map=site.symbolic, dict_of_variables=fit_vars, Y=y_fit)
    tb.register("dimensional_verification", SITE.dimensional_verification,
                dict_of_dimension=dims, num_units=n_units, target_dimension=target)
    tb.register("evaluate", SITE.evaluate, tb=tb, dict_of_operators=site.operators,
                dict_of_variables=fit_vars, Y=y_fit)

    # the authors' operators and rates (Maxwell_tlr_and_rnc_test.py)
    tb.register("select", tools.selTournament, tournsize=200)
    tb.register("mut_uniform", SITE.mutate_uniform, host_pset=host,
                func=tb.plasmid_individual, ind_pb=0.2, pb=1)
    tb.register("mut_invert", SITE.invert, pb=0.2)
    tb.register("mut_is_transpose", SITE.is_transpose, pb=0.2)
    tb.register("mut_ris_transpose", SITE.ris_transpose, pb=0.2)
    tb.register("mut_gene_transpose", SITE.gene_transpose, pb=0.2)
    tb.register("cx_1p", SITE.crossover_one_point, pb=0.2)
    tb.register("cx_2p", SITE.crossover_two_point, pb=0.2)
    tb.register("cx_gene", SITE.crossover_gene, pb=0.2)
    tb.register("mut_uniform_plasmid", gep.mutate_uniform, pset=plasmid, ind_pb=0.05, pb=1)
    tb.register("mut_invert_plasmid", gep.invert, pb=0.1)
    tb.register("mut_is_transpose_plasmid", gep.is_transpose, pb=0.1)
    tb.register("mut_ris_transpose_plasmid", gep.ris_transpose, pb=0.1)
    tb.register("mut_gene_transpose_plasmid", gep.gene_transpose, pb=0.1)
    tb.register("mut_dc_plasmid", gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
    tb.register("mut_invert_dc_plasmid", gep.invert_dc, pb=0.1)
    tb.register("mut_transpose_dc_plasmid", gep.transpose_dc, pb=0.1)
    tb.register("mut_rnc_array_dc_plasmid", gep.mutate_rnc_array_dc, rnc_gen=tb.rnc_gen,
                ind_pb="0.5p")
    tb.pbs["mut_rnc_array_dc_plasmid"] = 1
    return tb


def predict(site, ind, weights, nonzero, X_vars):
    terms = site.gene_terms(ind, X_vars)
    return sum(float(weights[i, 0]) * terms[g][:, 0, 0] for i, g in enumerate(nonzero))


def run_condition(site, cond, units, n_pop=N_POP, n_gen=N_GEN, seed=SEED):
    X = np.asarray(cond["X_train"], dtype=float)
    y = np.asarray(cond["y_train"], dtype=float)
    ntr = int(0.8 * len(y))
    fit_vars, val_vars = variables(X[:ntr], units), variables(X[ntr:], units)
    test_vars = variables(np.asarray(cond["X_test"], dtype=float), units)
    rhs = np.asarray(cond["rhs_test"], dtype=float)
    y_fit = y[:ntr][:, None, None]
    tb = make_toolbox(site, X.shape[1], fit_vars, y_fit, units)

    random.seed(seed)
    np.random.seed(seed)
    host_pop = tb.host_population(n=n_pop)
    plasmid_pop = tb.plasmid_population(host_pop)
    for ind, plasmids in zip(host_pop, plasmid_pop):
        ind.plasmid = plasmids
    hof = site.tools.HallOfFame(4)
    stats = site.tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as work:     # gep_simple writes output/ here
        os.chdir(work)
        try:
            t0 = time.perf_counter()
            _, log = site.SITE.gep_simple(host_pop, plasmid_pop, tb, n_generations=n_gen,
                                          n_elites=1, n_alien_inds=100, stats=stats,
                                          hall_of_fame=hof, verbose=False,
                                          tolerance=1e-6, GEP_type="pde")
            t_fit = time.perf_counter() - t0
        finally:
            os.chdir(cwd)

    t0 = time.perf_counter()
    best = None
    for ind in hof:
        if ind.fitness.values[0] >= 1000:          # SITE's loss for a rejected individual
            continue
        weights, nonzero, _ = site.SITE.tlr(ind, site.operators, fit_vars, y_fit)
        if weights is None:
            continue
        rv = r2(y[ntr:], predict(site, ind, weights, nonzero, val_vars))
        if np.isfinite(rv) and (best is None or rv > best[0]):
            best = (rv, ind, weights, nonzero)
    t_sel = time.perf_counter() - t0

    out = {"pde": cond["pde"], "sigma": cond["sigma"], "truth": cond["truth"],
           "generations": len(log) - 1, "time_s": t_fit + t_sel}
    if best is None:
        return {**out, "r2_test": None, "recovered": False, "expression": None}
    _, ind, weights, nonzero = best
    pt = predict(site, ind, weights, nonzero, test_vars)
    rt = r2(rhs, pt)
    model = site.model(ind, weights, nonzero, X.shape[1], units)
    # the recorded expression must be the scored model
    import sympy as sp
    syms = sorted(model.free_symbols, key=str)
    values = {feature_name(o): np.asarray(cond["X_test"], dtype=float)[:, o]
              for o in range(X.shape[1])}
    values.update({c: 1.0 for c in UNIT_CONSTS})
    mv = sp.lambdify(syms, model, "numpy")(*[values[str(s)] for s in syms]) if syms \
        else float(model) * np.ones(len(pt))
    assert np.max(np.abs(mv - pt)) <= 1e-6 * np.max(np.abs(pt)), "expression != model"
    return {**out, "r2_test": float(rt) if np.isfinite(rt) else None,
            "recovered": bool(np.isfinite(rt) and rt > R2_RECOVERED),
            "expression": str(sp.N(model, 6)), "loss": ind.fitness.values[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site_repo", help="clone of https://github.com/BUAA-MARS-group/SITE")
    ap.add_argument("--units", action="store_true", help="run SITE's dimensional check")
    ap.add_argument("--cells", default="", help="e.g. heat:0.01,ks:0 (default: all)")
    ap.add_argument("--generations", type=int, default=N_GEN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    site = Site(os.path.abspath(args.site_repo))
    conds = json.load(open(os.path.join(HERE, "data", "conditions.json")))
    if args.cells:
        want = {(p, float(s)) for p, s in (c.split(":") for c in args.cells.split(","))}
        conds = [c for c in conds if (c["pde"], float(c["sigma"])) in want]

    results = []
    for cond in conds:
        r = run_condition(site, cond, args.units, n_gen=args.generations)
        results.append(r)
        r2t = r["r2_test"] if r["r2_test"] is not None else -np.inf
        print(f"{r['pde']:8s} sigma={r['sigma']:<5g} R2={max(r2t, -9.9999):8.4f} "
              f"{'OK' if r['recovered'] else 'no'} {r['time_s']:6.1f}s gen={r['generations']:3d}  "
              f"{(r['expression'] or '-')[:90]}", flush=True)

    label = "SITE (units)" if args.units else "SITE"
    out = args.out or os.path.join(HERE, "results",
                                   "site_units.json" if args.units else "site.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"method": label, "results": results}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
