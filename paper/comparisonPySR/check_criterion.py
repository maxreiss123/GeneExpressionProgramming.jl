"""What the recovery criteria accept: models made from each of the 15 formulas, judged by
judge.check (the symbolic criterion, as reported) and by audit_symbolic.py's two numeric
tests (`numeric`: the symbolic criterion checked on the test points; `equal`: the
formula, its constants included). Writes results/criterion_check.csv.

    python check_criterion.py

The models, with f the formula, x1 its first input and m1 the middle of x1's range:

* f, f * 1.005 (constants 0.5 % off), 2 f, f + 1 (shifted by a constant);
* f (1 + 0.01 x1 / m1) and f (1 + 0.05 x1 / m1): an extra term of about 1 % and 5 % of
  the formula;
* f with x1 set to m1 (an input missing), f with x1 replaced by x1^2 / m1 (an exponent
  off), and 0 * x1 (zero).
"""
import os

import numpy as np
import pandas as pd
import sympy

import audit_symbolic as A
import common
import judge

VARIANTS = [
    ("the formula", lambda f, x, m: f),
    ("constants 0.5 % off", lambda f, x, m: sympy.Float(1.005) * f),
    ("twice the formula", lambda f, x, m: 2 * f),
    ("the formula plus 1", lambda f, x, m: f + 1),
    ("an extra term of 1 %", lambda f, x, m: f * (1 + sympy.Float(0.01) * x / m)),
    ("an extra term of 5 %", lambda f, x, m: f * (1 + sympy.Float(0.05) * x / m)),
    ("an input missing", lambda f, x, m: f.subs(x, m)),
    ("an exponent off", lambda f, x, m: f.subs(x, x ** 2 / m)),
    ("zero", lambda f, x, m: 0 * x),
]


def main():
    rows = []
    for name, group in common.EQUATIONS:
        pb = common.problem(name)
        f = pb.formula_sympy_eval
        x = pb.X_sympy_symbols[0]
        m = sympy.Float((pb.X_lows[0] + pb.X_highs[0]) / 2)
        _, _, Xte, yte = common.make_data(name, 1)
        for label, make in VARIANTS:
            model = make(f, x, m)
            ok = judge.check(pb, model, lambda e, pb: e)[0]
            fn = sympy.lambdify(pb.X_sympy_symbols, model, "numpy")
            p = A.values(lambda: fn(*Xte), yte.shape)
            if p is None:
                numeric = equal = False
            else:
                sr, med, sd = A.criterion(p, yte)
                er, ed = A.equality(p, yte)
                numeric = bool((sr <= A.TOL and 0.005 <= med <= 200) or sd <= A.TOL)
                equal = bool(er <= A.TOL or ed <= A.TOL)
            rows.append(dict(equation=name, group=group, model=label, symbolic=ok,
                             numeric=numeric, equal=equal))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(common.RESULTS_DIR, "criterion_check.csv"), index=False)
    order = [v for v, _ in VARIANTS]
    t = df.groupby("model", sort=False)[["symbolic", "numeric", "equal"]].sum().reindex(order)
    pd.set_option("display.width", 200)
    print("formulas (of 15) for which each criterion accepts the model")
    print(t.to_string())
    for label in ("an extra term of 1 %", "an extra term of 5 %"):
        acc = df[(df.model == label) & df.symbolic].equation.tolist()
        print("%s, accepted by the symbolic criterion: %s" % (label, ", ".join(acc) or "none"))


if __name__ == "__main__":
    main()
