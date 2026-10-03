"""Reference fields for the PDE-discovery benchmark.

Four canonical 1D PDEs of the kind PDE-FIND (Rudy et al., Sci. Adv. 2017) was validated
on, solved pseudo-spectrally with ETDRK4 (Kassam & Trefethen, SISC 2005) on periodic
domains:

    heat     u_t = 0.1 u_xx
    burgers  u_t = -u u_x + 0.1 u_xx
    kdv      u_t = -6 u u_x - u_xxx
    ks       u_t = -u u_x - u_xx - u_xxxx

Each data/<pde>.npz carries x, t and u (nt x nx); data/meta.json holds the ground-truth
term dictionary the scorer uses. Every field is checked for finiteness and spatial
resolution (energy in the top quarter of wavenumbers < 1e-6).
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def etdrk4(u0, L, nonlin, dt, nsteps, save_every):
    """ETDRK4 in Fourier space. L: linear symbol (len nx, complex ok);
    nonlin(v) -> N(v) in Fourier space, v = fft(u)."""
    v = np.fft.fft(u0)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
    # complex L (KdV): the contour trick needs the complex mean, not the real part
    if np.iscomplexobj(L) and np.max(np.abs(L.imag)) > 0:
        Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
        f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1)
        f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1)
        f3 = dt * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1)

    snaps = [np.real(np.fft.ifft(v))]
    for n in range(1, nsteps + 1):
        Nv = nonlin(v)
        a = E2 * v + Q * Nv
        Na = nonlin(a)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlin(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            snaps.append(np.real(np.fft.ifft(v)))
    return np.array(snaps)


def wavenumbers(nx, Lx):
    return 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)


def dealias(k):
    m = np.ones_like(k)
    m[np.abs(k) > (2 / 3) * np.max(np.abs(k))] = 0
    return m


def make_heat():
    nx, Lx, nu = 256, 16.0, 0.1
    x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
    k = wavenumbers(nx, Lx)
    u0 = np.exp(-(x + 2) ** 2) + 0.5 * np.exp(-2 * (x - 3) ** 2)
    L = -nu * k ** 2
    dt, T, nsave = 1e-3, 8.0, 100
    nsteps = int(T / dt)
    u = etdrk4(u0, L, lambda v: np.zeros_like(v), dt, nsteps, nsteps // nsave)
    t = np.linspace(0, T, u.shape[0])
    return dict(name="heat", x=x, t=t, u=u,
                truth="u_t = 0.1*u_xx", terms={"u_xx": 0.1}, max_order=2)


def make_burgers():
    nx, Lx, nu = 256, 16.0, 0.1
    x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
    k = wavenumbers(nx, Lx)
    da = dealias(k)
    u0 = np.exp(-(x + 2) ** 2)
    L = -nu * k ** 2

    def nonlin(v):
        u = np.real(np.fft.ifft(v))
        return -0.5j * k * da * np.fft.fft(u * u)

    dt, T, nsave = 1e-3, 10.0, 100
    nsteps = int(T / dt)
    u = etdrk4(u0, L, nonlin, dt, nsteps, nsteps // nsave)
    t = np.linspace(0, T, u.shape[0])
    return dict(name="burgers", x=x, t=t, u=u,
                truth="u_t = -u*u_x + 0.1*u_xx", terms={"u*u_x": -1.0, "u_xx": 0.1},
                max_order=2)


def make_kdv():
    nx, Lx = 512, 40.0
    x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
    k = wavenumbers(nx, Lx)
    da = dealias(k)
    # two solitons, u = (c/2) sech^2(sqrt(c)/2 (x - x0)) for u_t = -6uu_x - u_xxx
    def sol(c, x0):
        return c / 2 * np.cosh(np.sqrt(c) / 2 * (x - x0)) ** -2
    u0 = sol(6.0, -12.0) + sol(2.0, -5.0)
    L = 1j * k ** 3          # -u_xxx in Fourier space

    def nonlin(v):
        u = np.real(np.fft.ifft(v))
        return -3j * k * da * np.fft.fft(u * u)   # -6 u u_x = -3 (u^2)_x

    dt, T, nsave = 2e-5, 3.0, 150
    nsteps = int(T / dt)
    u = etdrk4(u0, L, nonlin, dt, nsteps, nsteps // nsave)
    t = np.linspace(0, T, u.shape[0])
    return dict(name="kdv", x=x, t=t, u=u,
                truth="u_t = -6*u*u_x - u_xxx", terms={"u*u_x": -6.0, "u_xxx": -1.0},
                max_order=3)


def make_ks():
    nx, Lx = 512, 32 * np.pi
    x = np.linspace(0, Lx, nx, endpoint=False)
    k = wavenumbers(nx, Lx)
    da = dealias(k)
    u0 = np.cos(x / 16) * (1 + np.sin(x / 16))
    L = k ** 2 - k ** 4      # -u_xx - u_xxxx

    def nonlin(v):
        u = np.real(np.fft.ifft(v))
        return -0.5j * k * da * np.fft.fft(u * u)

    dt = 0.05
    # discard the transient, keep the chaotic attractor
    burn, keep, nsave = 50.0, 100.0, 250
    u_all = etdrk4(u0, L, nonlin, dt, int((burn + keep) / dt),
                   int(keep / dt) // nsave)
    nburn = int(np.ceil((burn / (burn + keep)) * (u_all.shape[0] - 1)))
    u = u_all[nburn:]
    t = np.arange(u.shape[0]) * (keep / nsave)
    return dict(name="ks", x=x, t=t, u=u,
                truth="u_t = -u*u_x - u_xx - u_xxxx",
                terms={"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}, max_order=4)


def spectral_tail(u, frac=0.25):
    """Energy fraction in the top `frac` of wavenumbers -- resolution check."""
    s = np.mean(np.abs(np.fft.fft(u, axis=1)) ** 2, axis=0)
    n = len(s) // 2
    hi = int(n * (1 - frac))
    return float(np.sum(s[hi:n]) / np.sum(s[:n]))


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for maker in (make_heat, make_burgers, make_kdv, make_ks):
        d = maker()
        u = d["u"]
        assert np.all(np.isfinite(u)), d["name"]
        tail = spectral_tail(u)
        np.savez_compressed(os.path.join(OUT, f"{d['name']}.npz"),
                            x=d["x"], t=d["t"], u=u)
        meta[d["name"]] = dict(truth=d["truth"], terms=d["terms"],
                               max_order=d["max_order"],
                               nt=int(u.shape[0]), nx=int(u.shape[1]),
                               umax=float(np.max(np.abs(u))),
                               spectral_tail=tail)
        print(f"{d['name']:8s} nt={u.shape[0]:4d} nx={u.shape[1]:4d} "
              f"max|u|={np.max(np.abs(u)):7.3f}  tail={tail:.2e}  {d['truth']}")
        assert tail < 1e-6, f"{d['name']} under-resolved (tail {tail:.1e})"
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
