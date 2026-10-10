"""Check, and with --fix correct, assets/case_dsc.json.

    python check_case_dsc.py [--fix]

Each entry describes one AI-Feynman equation: its variables (`orig_values`, in the column
order of its data, named x1 ... xn in `vars`), their units (`dims`, SI exponents
`[kg, m, s, K, mol, A, cd]`), the units of the output (`targetdims`), the formula
(`true_model`, and `true_model_in_x_notation` in x1 ... xn), the data set (`dataset`) and
SRSD's difficulty (`level`). The reference is PhySO's copy of the AI-Feynman table: its
variables, formulas and units (physo/benchmark/FeynmanDataset, the units `[m, s, kg, T, V]`
rewritten as SI). Checks per entry:

* balance: the formula, with the entry's units for its variables, has the entry's
  target units (sums add like units; arguments of exp, log, sin, ... are dimensionless);
* physical units: each variable's units, and the output's, equal the table's;
* description: the variables are the table's, in its order; `true_model` is the table's
  formula; the x-notation is that formula in x1 ... xn; `dataset` is feynman_<name>.

`--fix` sets all of these from the table and rewrites the file. Exception to the table's
units: II.11.17's number densities n_0 and n are in m^-3 (the table has them
dimensionless), which is physically right and balances as well.

The file as it was, before the fixes:

* Units: 30 of the 100 entries failed. 12 did not balance:

      I.9.18     z1, z2 dimensionless, so the sum adds m^2 to 1
      I.12.5     target in the units of E instead of N
      I.30.5     d dimensionless, so arcsin's argument is a length; target a length
      I.32.17    c in m s^-1 A; target in J instead of W
      I.34.8     p in J s (angular momentum) instead of kg m s^-1
      I.43.16    mu inverted (kg/s), q in Wb, V in A
      I.43.43    units of A and v swapped
      II.15.4    mu in kg m^-3 s^4 A^-1 and B in C/m^2; their product is not J
      II.34.29a  target [2, 0, 0, 0, 0, 1, 0] instead of A m^2 (exponents swapped)
      II.34.29b  5 variables, 3 dimension vectors
      II.37.1    units of B and chi swapped
      III.8.54   E in kg m^2 instead of J, so sin's argument has units

  18 balanced only with unphysical units, most with electric and magnetic quantities
  swapped consistently (charge in Wb, epsilon in H/m, E in A/m): I.11.19, I.30.3,
  I.32.5, I.37.4, I.50.26, II.3.24, II.8.31, II.11.20, II.11.27, II.13.34, II.27.16,
  II.35.18, II.35.21, II.36.38, III.7.38, III.9.52, III.10.19, III.19.51.

* Formulas: 31 had rounded constants (0.08 for 1/(4 pi) in I.12.2, 0.013 for
  1/(8 pi^2) = 0.0127 in III.15.14, 3.1415 for pi), I.30.5 wrote numpy's arcsin, I.26.2
  had none; the x-notation was rounded further (13.0 for 4 pi in III.13.18) and in I.6.2
  divided by theta instead of sigma.
* II.11.17 had five variables with k_B folded into the formula (SRSD's form) and no
  description; it now has the table's six, k_B among them, as the other entries have
  their constants.
* II.34.2a named the data set of II.34.29a; II.34.2a and II.11.17 had no level, and
  I.30.3 and II.34.2 a level other than SRSD's (configs/datasets/feynman/*_set.yaml of
  github.com/omron-sinicx/srsd-benchmark).
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import sympy

warnings.filterwarnings("ignore")
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn  # noqa: E402

import common  # noqa: E402

PATH = os.path.join(common.HERE, "..", "..", "assets", "case_dsc.json")
PER_M3 = [0, -3, 0, 0, 0, 0, 0]
UNIT_OVERRIDES = {"II.11.17": {"n_0": PER_M3, "target": PER_M3}}
LEVELS = {"II.34.2a": "medium", "II.11.17": "hard", "I.30.3": "hard", "II.34.2": "medium"}
SYMPY_NAMES = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "ln": "log"}
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")


class Unbalanced(Exception):
    pass


def dims_of(e, d):
    if e.is_Number or e in (sympy.pi, sympy.E):
        return np.zeros(7)
    if e.is_Symbol:
        return d[e.name]
    if e.is_Add:
        parts = [dims_of(a, d) for a in e.args]
        if any(not np.allclose(p, parts[0]) for p in parts[1:]):
            raise Unbalanced("adds " + " and ".join(fmt(p) for p in parts))
        return parts[0]
    if e.is_Mul:
        return sum(dims_of(a, d) for a in e.args)
    if e.is_Pow:
        base, ex = e.args
        if ex.is_Number:
            return dims_of(base, d) * float(ex)
        if dims_of(base, d).any() or dims_of(ex, d).any():
            raise Unbalanced("a power with units")
        return np.zeros(7)
    for a in e.args:
        if dims_of(a, d).any():
            raise Unbalanced("%s of %s" % (type(e).__name__, fmt(dims_of(a, d))))
    return np.zeros(7)


def fmt(v):
    return str([int(x) if float(x).is_integer() else float(x) for x in v])


def as_ints(v):
    return [int(x) if float(x).is_integer() else x for x in v]


def table(name):
    """The table's variables, formula (sympy syntax), variable units and output units."""
    pb = Feyn.FeynmanProblem(eq_name=name, original_var_names=True)
    names = [str(n) for n in pb.X_names]
    formula = IDENT.sub(lambda m: SYMPY_NAMES.get(m.group(0), m.group(0)), pb.formula_original)
    over = UNIT_OVERRIDES.get(name, {})
    dims = [over.get(n, as_ints(common.si_units(u))) for n, u in zip(names, pb.X_units)]
    target = over.get("target", as_ints(common.si_units(pb.y_units)))
    return names, formula, dims, target


def x_notation(formula, names):
    xs = {n: "x%d" % (i + 1) for i, n in enumerate(names)}
    return IDENT.sub(lambda m: xs.get(m.group(0), m.group(0)), formula)


def parse(text, names):
    local = {n: sympy.Symbol(n) for n in names}
    local.update(pi=sympy.pi, E=sympy.E)
    return sympy.parsing.sympy_parser.parse_expr(text, local_dict=local)


def balance(entry, names, formula):
    """None if the entry's units balance in the formula, else why not."""
    if not (len(entry["vars"]) == len(entry["dims"]) == len(names)):
        return "%d variables, %d in vars, %d dimension vectors" % (
            len(names), len(entry["vars"]), len(entry["dims"]))
    d = {n: np.array(u, float) for n, u in zip(names, entry["dims"])}
    try:
        out = dims_of(parse(formula, names), d)
    except Unbalanced as err:
        return str(err)
    if not np.allclose(out, entry["targetdims"]):
        return "gives %s, target %s" % (fmt(out), fmt(entry["targetdims"]))
    return None


def same(text, formula, names, xs=None):
    """Whether `text` (in `xs` if given, else in `names`) is the formula."""
    try:
        e = parse(text, xs or names)
        if xs:
            e = e.subs({sympy.Symbol(x): sympy.Symbol(n) for x, n in zip(xs, names)})
        return sympy.simplify(e - parse(formula, names)) == 0
    except Exception:
        return False


def description(name, entry, names, formula):
    """What in the entry's description differs from the table (formulas compared
    symbolically, so only a different formula or rounded constants count)."""
    wrong = []
    if entry.get("orig_values") != names:
        wrong.append("variables")
    if "true_model" not in entry or not same(entry["true_model"], formula, names):
        wrong.append("formula")
    xs = ["x%d" % (i + 1) for i in range(len(names))]
    if ("true_model_in_x_notation" not in entry
            or not same(entry["true_model_in_x_notation"], formula, names, xs)):
        wrong.append("x-notation")
    if entry.get("dataset") != "feynman_" + name.replace(".", "_"):
        wrong.append("data set")
    if "level" not in entry or entry["level"] != LEVELS.get(name, entry["level"]):
        wrong.append("level")
    return wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    case = json.load(open(PATH))
    failed = {"do not balance": [], "have units that are not physical": [],
              "differ from the table in their description": []}
    for name, entry in case.items():
        names, formula, dims, target = table(name)
        why = balance(entry, entry.get("orig_values") or names, formula)
        if why:
            failed["do not balance"].append(name)
            print("%-10s does not balance: %s" % (name, why))
        units = (len(entry["dims"]) != len(dims) or
                 not all(np.allclose(u, p) for u, p in zip(entry["dims"], dims)) or
                 not np.allclose(entry["targetdims"], target))
        if units:
            failed["have units that are not physical"].append(name)
        wrong = description(name, entry, names, formula)
        if wrong:
            failed["differ from the table in their description"].append(name)
            print("%-10s differs in: %s" % (name, ", ".join(wrong)))
        if a.fix:
            level = LEVELS.get(name, entry.get("level"))
            entry.clear()
            entry.update(vars=["x%d" % (i + 1) for i in range(len(names))], dims=dims,
                         targetdims=target, level=level,
                         dataset="feynman_" + name.replace(".", "_"), orig_values=names,
                         true_model=formula, true_model_in_x_notation=x_notation(formula, names))
            assert balance(entry, names, formula) is None, name
            assert not description(name, entry, names, formula), name
    print()
    for what, names in failed.items():
        print("%d %s: %s" % (len(names), what, ", ".join(names)))
    if a.fix:
        with open(PATH, "w") as f:
            json.dump(case, f, indent=4)
            f.write("\n")
        print("rewrote %s" % os.path.normpath(PATH))


if __name__ == "__main__":
    main()
