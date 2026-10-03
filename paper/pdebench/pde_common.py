"""Shared protocol for the PDE-discovery benchmark.

Everything that decides comparability lives here, once: the noise model, the derivative
estimation, the train/test sampling and the recovery metric. Every method (PDE-FIND and
the three GEP harnesses) consumes the same feature matrices, so the comparison isolates
the regression stage -- no method gets a better differentiator.

Noise (PDE-FIND convention): u_noisy = u + sigma * std(u) * randn, seeded.

Derivatives: local least-squares polynomial (Savitzky-Golay weights) along each axis.
The spatial axis is periodic, so spatial windows wrap and there are no edge artefacts;
the time axis is not, so a margin of frames is trimmed from both ends. Windows widen with
noise, identically for every method.

Recovery metric: the fitted right-hand side, evaluated on CLEAN derivative features at a
random sample of the space-time grid, compared with the analytic right-hand side -- R^2
of that fit ("functional recovery"), recovered if > 0.99. This scores the discovered
operator itself, not its fit to its own noise. The test points are drawn independently
of the training points, not disjoint from them; at sigma = 0 a shared point carries the
same features in both.
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SIGMAS = (0.0, 0.01, 0.05)
N_TRAIN = 5000
N_TEST = 20000
T_MARGIN = 5            # frames trimmed from each end of t for u_t estimation
R2_RECOVERED = 0.99


def load_field(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    meta = json.load(open(os.path.join(DATA, "meta.json")))[name]
    return d["x"], d["t"], d["u"], meta


def add_noise(u, sigma, seed):
    rng = np.random.RandomState(100003 * seed + 17)
    return u + sigma * np.std(u) * rng.randn(*u.shape)


def sg_weights(window, degree, h, order):
    """Savitzky-Golay derivative weights on a uniform grid: differentiate the
    least-squares polynomial of `degree` over `window` points at its centre."""
    half = window // 2
    s = (np.arange(window) - half) * h
    V = np.vander(s, degree + 1, increasing=True)
    pinv = np.linalg.pinv(V)
    return pinv[order] * float(math.factorial(order))


def periodic_derivative(u, h, order, window, degree):
    """d^order/dx^order along axis 1 (periodic), same SG weights at every point."""
    w = sg_weights(window, degree, h, order)
    half = window // 2
    nx = u.shape[1]
    idx = (np.arange(nx)[:, None] + np.arange(-half, half + 1)[None, :]) % nx
    return np.einsum("txw,w->tx", u[:, idx], w)


def time_derivative(u, h, window, degree):
    """du/dt along axis 0 (not periodic): interior points use the centred window,
    the ends are excluded by the caller via T_MARGIN."""
    w = sg_weights(window, degree, h, 1)
    half = window // 2
    nt = u.shape[0]
    ut = np.full_like(u, np.nan)
    for i in range(half, nt - half):
        ut[i] = np.einsum("wx,w->x", u[i - half:i + half + 1], w)
    return ut


def build_features(x, t, u, sigma, max_order, spatial_window=None):
    """Feature matrix [u, u_x, ..., u_x^(max_order)] and target u_t from a (noisy)
    field. Returns names, features (nt x nx x k) and u_t (nt x nx, NaN at t-ends).
    `spatial_window` replaces the noise-dependent spatial window length."""
    hx = float(x[1] - x[0])
    ht = float(t[1] - t[0])
    if sigma == 0:
        wx, dx = 9, 6
        wt, dt = 7, 4
    else:
        wx, dx = 31, 6
        wt, dt = 13, 4
    if spatial_window is not None:
        wx = spatial_window
    names = ["u"] + [f"u_{'x' * o}" for o in range(1, max_order + 1)]
    feats = [periodic_derivative(u, hx, 0, wx, dx)]
    for o in range(1, max_order + 1):
        feats.append(periodic_derivative(u, hx, o, wx, dx))
    ut = time_derivative(u, ht, wt, dt)
    return names, np.stack(feats, axis=-1), ut


def true_rhs(terms, names, feats):
    """Analytic right-hand side from the ground-truth term dictionary, evaluated on
    a flat (n x k) feature matrix. Term keys: 'u_xx', 'u*u_x', ..."""
    col = {n: feats[:, i] for i, n in enumerate(names)}
    rhs = np.zeros(feats.shape[0])
    for term, c in terms.items():
        v = np.ones(feats.shape[0])
        for f in term.split("*"):
            v = v * col[f]
        rhs += c * v
    return rhs


def make_condition(name, sigma, seed=1, spatial_window=None):
    """Everything a method needs for one (pde, sigma) cell:
    noisy training features/target and clean test features/true-RHS.
    `spatial_window` replaces the spatial window of the training features only."""
    x, t, u, meta = load_field(name)
    max_order = max(3, meta["max_order"])

    un = add_noise(u, sigma, seed)
    names, F, ut = build_features(x, t, un, sigma, max_order, spatial_window)
    m = T_MARGIN
    Ff = F[m:-m].reshape(-1, F.shape[-1])
    utf = ut[m:-m].reshape(-1)
    ok = np.isfinite(utf)
    Ff, utf = Ff[ok], utf[ok]
    rng = np.random.RandomState(7 * seed + 1)
    tr = rng.choice(len(utf), min(N_TRAIN, len(utf)), replace=False)

    namesC, Fc, _ = build_features(x, t, u, 0.0, max_order)
    Fcf = Fc[m:-m].reshape(-1, Fc.shape[-1])
    te = rng.choice(len(Fcf), min(N_TEST, len(Fcf)), replace=False)
    rhs = true_rhs(meta["terms"], namesC, Fcf[te])

    return dict(pde=name, sigma=sigma, seed=seed, names=names, truth=meta["truth"],
                X_train=Ff[tr], y_train=utf[tr], X_test=Fcf[te], rhs_test=rhs)


def r2(y_true, y_pred):
    if not np.all(np.isfinite(y_pred)):
        return -np.inf
    sstot = np.sum((y_true - y_true.mean()) ** 2)
    ssres = np.sum((y_true - y_pred) ** 2)
    return 1 - ssres / sstot if sstot > 0 else -np.inf


def export_for_julia(path):
    """Write every (pde, sigma) condition to one JSON the Julia harness reads."""
    out = []
    for name in ("heat", "burgers", "kdv", "ks"):
        for sigma in SIGMAS:
            c = make_condition(name, sigma)
            out.append(dict(pde=c["pde"], sigma=c["sigma"], seed=c["seed"],
                            names=c["names"], truth=c["truth"],
                            X_train=c["X_train"].tolist(),
                            y_train=c["y_train"].tolist(),
                            X_test=c["X_test"].tolist(),
                            rhs_test=c["rhs_test"].tolist()))
            print(f"exported {name} sigma={sigma}")
    json.dump(out, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    export_for_julia(os.path.join(DATA, "conditions.json"))
