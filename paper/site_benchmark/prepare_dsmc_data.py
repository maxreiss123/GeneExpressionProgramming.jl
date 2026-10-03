"""
Prepare the DSMC lid-driven-cavity data of arXiv:2507.01466v1 Sec. 4 for
GeneExpressionProgramming.jl.

The authors ship the raw DSMC fields
    2 Macroscopic equation discovery based on small scale simulation/
        1_incompressible_flow/data/TecGrid-Cavity_Kn0.005_nloop50000000.dat
        2_compressible_flow/data/TecGrid-Cavity_Kn0.005_Ma1.0_nloop400000000.dat
but not `data/gradients.mat`, i.e. not the velocity gradients their TensorFlow
network produces. Those gradients are rebuilt here with second-order finite
differences on the uniform 400 x 400 grid. Sampling is restricted to the central
20%-80% of the domain, as in the paper (Sec. 4.2), which keeps clear of the boundary
noise the authors cite as their reason for neural-network differentiation.

Everything downstream (column meanings, viscosity law, pressure, terminal library)
follows the authors' `constitutive_equation_*.py`:

    p      = n k T                         static pressure
    mu     = mu_ref (T/T_ref)^0.81         VHS viscosity
    S_ij   = 1/2 (du_i/dx_j + du_j/dx_i)   strain rate
    D_kk   = du/dx + dv/dy                 velocity divergence
    sigma  = (MOMxx MOMxy; MOMxy MOMyy)    DSMC stress, the target

Reference (Newtonian) form
    sigma_ij = -2 mu S_ij + (2/3) mu D_kk delta_ij + p delta_ij

Output: data/dsmc_<case>_s<seed>.csv, one row per (sample, i, j), i,j in {0,1}:
    p_idx,i,j,S,delta,Dkk,pres,mu,sigma

    python prepare_dsmc_data.py <site_repo> [--n 200] [--seeds 1,2,3,4,5] [--outdir data]
"""

import argparse
import os

import numpy as np

MU_REF = 2.117e-5
T_REF = 273.15
K_B = 1.38065e-23
OMEGA = 0.81

CASES = {
    "incompressible": ("1_incompressible_flow", "TecGrid-Cavity_Kn0.005_nloop50000000.dat"),
    "compressible": ("2_compressible_flow", "TecGrid-Cavity_Kn0.005_Ma1.0_nloop400000000.dat"),
}


def load_case(site_repo, case):
    sub, fname = CASES[case]
    path = os.path.join(site_repo, "2 Macroscopic equation discovery based on small scale simulation",
                        sub, "data", fname)
    raw = np.loadtxt(path, skiprows=3)
    x, y = raw[:, 0], raw[:, 1]
    n_rho, u, v, T = raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 6]
    s_xx, s_yy, s_xy = raw[:, 7], raw[:, 8], raw[:, 10]

    nx = len(np.unique(np.round(x, 12)))
    ny = len(np.unique(np.round(y, 12)))
    assert nx * ny == raw.shape[0], "unexpected grid: %d x %d != %d" % (nx, ny, raw.shape[0])

    # this TecPlot point dump runs y fastest, so the natural array is [ix, iy]
    shape = (nx, ny)
    X = x.reshape(shape)
    Y = y.reshape(shape)
    U = u.reshape(shape)
    V = v.reshape(shape)

    xs = X[:, 0]
    ys = Y[0, :]
    u_x, u_y = np.gradient(U, xs, ys, edge_order=2)
    v_x, v_y = np.gradient(V, xs, ys, edge_order=2)

    p = n_rho * K_B * T
    mu = MU_REF * (T / T_REF) ** OMEGA

    return dict(x=x, y=y, u_x=u_x.ravel(), u_y=u_y.ravel(), v_x=v_x.ravel(),
                v_y=v_y.ravel(), p=p, mu=mu, s_xx=s_xx, s_yy=s_yy, s_xy=s_xy,
                lx=xs.max() - xs.min(), ly=ys.max() - ys.min(),
                x0=xs.min(), y0=ys.min())


def sample(d, n, seed):
    # central 20%-80% of the domain, as in the paper
    xn = (d["x"] - d["x0"]) / d["lx"]
    yn = (d["y"] - d["y0"]) / d["ly"]
    mask = (xn >= 0.2) & (xn <= 0.8) & (yn >= 0.2) & (yn <= 0.8)
    idx = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=n, replace=False)

    u_x, u_y = d["u_x"][pick], d["u_y"][pick]
    v_x, v_y = d["v_x"][pick], d["v_y"][pick]
    S = np.empty((n, 2, 2))
    S[:, 0, 0] = u_x
    S[:, 1, 1] = v_y
    S[:, 0, 1] = S[:, 1, 0] = 0.5 * (u_y + v_x)
    D_kk = u_x + v_y
    sigma = np.empty((n, 2, 2))
    sigma[:, 0, 0] = d["s_xx"][pick]
    sigma[:, 1, 1] = d["s_yy"][pick]
    sigma[:, 0, 1] = sigma[:, 1, 0] = d["s_xy"][pick]
    return dict(S=S, D_kk=D_kk, p=d["p"][pick], mu=d["mu"][pick], sigma=sigma)


def write(sam, path):
    n = sam["S"].shape[0]
    with open(path, "w") as f:
        f.write("p_idx,i,j,S,delta,Dkk,pres,mu,sigma\n")
        for k in range(n):
            for i in range(2):
                for j in range(2):
                    f.write("%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g\n" % (
                        k, i, j, sam["S"][k, i, j], 1 if i == j else 0,
                        sam["D_kk"][k], sam["p"][k], sam["mu"][k], sam["sigma"][k, i, j]))


def newtonian_check(sam):
    """Least-squares fit of the Newtonian model on the sample (sanity check)."""
    n = sam["S"].shape[0]
    delta = np.tile(np.eye(2), (n, 1, 1))
    basis = np.stack([(sam["mu"][:, None, None] * sam["S"]).ravel(),
                      (sam["mu"] * sam["D_kk"])[:, None, None].repeat(2, 1).repeat(2, 2).ravel()
                      * delta.ravel(),
                      (sam["p"][:, None, None] * delta).ravel()], axis=1)
    y = sam["sigma"].ravel()
    w, *_ = np.linalg.lstsq(basis, y, rcond=None)
    res = np.linalg.norm(basis @ w - y) / np.linalg.norm(y)
    return w, res


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("site_repo")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--outdir", default=os.path.join(here, "data"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for case in ("incompressible", "compressible"):
        d = load_case(args.site_repo, case)
        for seed in [int(s) for s in args.seeds.split(",")]:
            sam = sample(d, args.n, seed)
            path = os.path.join(args.outdir, "dsmc_%s_s%d.csv" % (case, seed))
            write(sam, path)
            w, res = newtonian_check(sam)
            print("%-15s seed %d -> %s   least-squares Newtonian fit: "
                  "%.3f mu S_ij %+.3f mu D_kk d_ij %+.3f p d_ij   (rel. residual %.2e)"
                  % (case, seed, os.path.basename(path), w[0], w[1], w[2], res))


if __name__ == "__main__":
    main()
