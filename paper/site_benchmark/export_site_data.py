"""
Export the benchmark datasets of

    T. Chen, H. Yang, W. Ma, J. Zhang,
    "Symbolic identification of tensor equations in multidimensional physical fields",
    arXiv:2507.01466v1 (2025)   --  the SITE framework

into plain CSV files (written at full double precision), so that
GeneExpressionProgramming.jl sees the data the reference implementation
(https://github.com/PistilReaper/SITE) generates for itself.

Case 1 -- Maxwell stress tensor (paper Sec. 3.1, Tables 1 and 2)
    The field definitions, the sampling and the noise model follow the authors'
    scripts
        1 Validation on benchmark problems/1_Maxwell_Stress/Maxwell_tlr_and_rnc_test.py
        1 Validation on benchmark problems/1_Maxwell_Stress/Maxwell_noise_test.py
    including the RNG seed (np.random.seed(0)), so the exported points are the 150
    samples those scripts generate.

Case 2 -- Reynolds stress transport (paper Sec. 3.2, Table 3)
    The authors' repository does *not* ship the OpenFOAM data
    (`data/processed_data.mat` is absent), so the decaying-homogeneous-isotropic
    -turbulence dataset is reconstructed from the description in Sec. 3.2:
    argon, T0 = 273.15 K, p0 = 1e5 Pa, k0 = 4265.9 m^2/s^2 (Ma_t = 0.3),
    100 samples uniformly spaced in t = [1e-8, 1e-6] s, with the standard
    power-law decay of homogeneous isotropic turbulence. This is a *surrogate*
    for the paper's data and is flagged as such wherever it is used.

Output (CSV, one row per (sample, i, j) tensor component):

    maxwell_<tag>.csv : p,i,j,EE,BB,delta,E2,B2,T
    reynolds_dhit.csv : p,i,j,R,delta,k,epsilon,dRdt

Usage:  python export_site_data.py [outdir]
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------- constants --
EPSILON_0 = 8.854e-12          # F/m
MU_0 = 4 * np.pi * 1e-7        # H/m
E0 = 1e6                       # V/m
B0 = 1e-3                      # T
K = np.pi                      # 1/m
N_POINTS = 150
SEED = 0


def E_field(x, y, z):
    return np.array([E0 * np.sin(K * x), E0 * np.cos(K * y), E0 * np.sin(K * z)])


def B_field(x, y, z):
    return np.array([B0 * np.cos(K * x), B0 * np.sin(K * y), B0 * np.cos(K * z)])


def maxwell_case(noise=0.0):
    """Reproduce the authors' Maxwell data generation for a given noise level.

    Returns the terminal library (as seen by the algorithm) and the target.
    For noise > 0 the fields are perturbed first, the stress tensor is computed
    from the perturbed fields and is then perturbed again, as in
    Maxwell_noise_test.py.
    """
    np.random.seed(SEED)
    points = np.random.uniform(low=-1, high=1, size=(N_POINTS, 3))

    E = np.zeros((N_POINTS, 3))
    B = np.zeros((N_POINTS, 3))
    for idx, (x, y, z) in enumerate(points):
        E[idx] = E_field(x, y, z)
        B[idx] = B_field(x, y, z)

    if noise > 0.0:
        noise_E = noise * np.array([np.std(E[:, c]) for c in range(3)]) * np.random.randn(N_POINTS, 3)
        noise_B = noise * np.array([np.std(B[:, c]) for c in range(3)]) * np.random.randn(N_POINTS, 3)
        E = E + noise_E
        B = B + noise_B

    delta_ij = np.tile(np.eye(3), (N_POINTS, 1, 1))
    E_iE_j = np.einsum("ni,nj->nij", E, E)
    B_iB_j = np.einsum("ni,nj->nij", B, B)
    E2 = np.einsum("ni,ni->n", E, E).reshape(-1, 1)
    B2 = np.einsum("ni,ni->n", B, B).reshape(-1, 1)

    T = (EPSILON_0 * (E_iE_j - 0.5 * E2[:, None] * delta_ij)
         + (1.0 / MU_0) * (B_iB_j - 0.5 * B2[:, None] * delta_ij))

    if noise > 0.0:
        sigma_T = np.array([np.std(T[:, i, j]) for i in range(3) for j in range(3)]).reshape(3, 3)
        T = T + noise * sigma_T * np.random.randn(N_POINTS, 3, 3)

    return dict(EE=E_iE_j, BB=B_iB_j, delta=delta_ij, E2=E2, B2=B2, T=T, points=points)


def write_maxwell(case, path):
    n = case["T"].shape[0]
    with open(path, "w") as f:
        f.write("p,i,j,EE,BB,delta,E2,B2,T\n")
        for p in range(n):
            for i in range(3):
                for j in range(3):
                    f.write("%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n" % (
                        p, i, j,
                        case["EE"][p, i, j], case["BB"][p, i, j], case["delta"][p, i, j],
                        case["E2"][p, 0], case["B2"][p, 0], case["T"][p, i, j]))


# ------------------------------------------------------ Reynolds surrogate ---
def reynolds_case(n_samples=100):
    """Decaying homogeneous isotropic turbulence surrogate (paper Sec. 3.2).

    In DHIT with zero mean velocity the Reynolds-stress transport equation
    degenerates to  dR_ij/dt = -(2/3) eps delta_ij  with R_ij = (2/3) k delta_ij.
    We use the classical power-law decay  k(t) = k0 (1 + t/t0)^(-n),
    for which  eps = -dk/dt  is consistent by construction, so the data satisfy
    the target equation to machine precision (as the paper's data do -- they are
    produced by a k-eps closure inside OpenFOAM).
    """
    k0 = 4265.9                      # m^2/s^2, paper Sec. 3.2
    n_decay = 1.25                   # classical DHIT decay exponent
    # eps0 = C_eps k0^{3/2} / L with an integral scale L = O(1 cm): the resulting
    # decay time scale is O(1e-4 s), i.e. k drops by a fraction of a percent over
    # the 1e-6 s window sampled in the paper.
    eps0 = 1.5e7                     # m^2/s^3
    t0 = n_decay * k0 / eps0

    t = np.linspace(1e-8, 1e-6, n_samples)
    k = k0 * (1.0 + t / t0) ** (-n_decay)
    eps = (n_decay * k0 / t0) * (1.0 + t / t0) ** (-n_decay - 1.0)   # = -dk/dt

    delta = np.tile(np.eye(3), (n_samples, 1, 1))
    R = (2.0 / 3.0) * k[:, None, None] * delta

    # The paper obtains dR_ij/dt by finite differences on the recorded R_ij; we do
    # the same (2nd order, one-sided at the ends) instead of using the analytic
    # derivative, so the tiny discretisation bias of the reference workflow is
    # part of the surrogate as well.
    dkdt = np.gradient(k, t, edge_order=2)
    dRdt = (2.0 / 3.0) * dkdt[:, None, None] * delta
    return dict(t=t, k=k, eps=eps, R=R, delta=delta, dRdt=dRdt)


def write_reynolds(case, path):
    n = case["R"].shape[0]
    with open(path, "w") as f:
        f.write("p,i,j,R,delta,k,epsilon,dRdt\n")
        for p in range(n):
            for i in range(3):
                for j in range(3):
                    f.write("%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n" % (
                        p, i, j,
                        case["R"][p, i, j], case["delta"][p, i, j],
                        case["k"][p], case["eps"][p], case["dRdt"][p, i, j]))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(outdir, exist_ok=True)

    for tag, noise in (("clean", 0.0), ("noise005", 0.05), ("noise010", 0.10), ("noise020", 0.20)):
        case = maxwell_case(noise)
        path = os.path.join(outdir, "maxwell_%s.csv" % tag)
        write_maxwell(case, path)
        rel = np.abs(case["T"]).mean()
        print("wrote %-34s  n=%d  mean|T|=%.4g" % (path, case["T"].shape[0], rel))

    rey = reynolds_case()
    path = os.path.join(outdir, "reynolds_dhit.csv")
    write_reynolds(rey, path)
    print("wrote %-34s  n=%d  k: %.5g -> %.5g" % (path, rey["R"].shape[0], rey["k"][0], rey["k"][-1]))


if __name__ == "__main__":
    main()
