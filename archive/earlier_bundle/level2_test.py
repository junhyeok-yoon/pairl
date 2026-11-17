import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
B, K, AM, U, DT, T = 40, 4, 2.0, 3.0, 0.002, 40.0
NS = int(T / DT)
g = lambda v: U * np.tanh(v / U)                      # monotone, 1-Lipschitz output map
phi = lambda x: np.stack([x, np.sin(2 * x), np.ones_like(x)], axis=1)
Gint = lambda x: np.stack([x ** 2 / 2, -np.cos(2 * x) / 2, x], axis=1)   # antiderivative of phi

L = rng.normal(size=(B, 3, 3)) * 0.4
M = np.einsum('bij,bkj->bik', L, L) + 0.3 * np.eye(3)
A = rng.normal(size=(B, K, 3))
w = rng.uniform(0, 1.5, size=(B, K))
gamma = 10 ** rng.uniform(-0.3, 1.7, size=B)
th = rng.uniform(-2, 2, size=(B, 3))
rb, rA, rw, rp = rng.uniform(-1, 1, B), rng.uniform(0.3, 1.2, B), rng.uniform(0.5, 3, B), rng.uniform(0, 6.28, B)
l0, l1, lw = rng.uniform(0.55, 0.7, B), rng.uniform(0.0, 0.25, B), rng.uniform(0.3, 2.0, B)
sw = rng.uniform(size=B) < 0.5                         # half the cases: effectiveness halves at t = 20 s
x0 = rng.uniform(-1, 1, B)

gradH = lambda z: np.einsum('bij,bj->bi', M, z) + np.einsum('bk,bkj->bj', w * np.tanh(np.einsum('bkj,bj->bk', A, z)), A)
Hf = lambda z: 0.5 * np.einsum('bi,bij,bj->b', z, M, z) + np.sum(w * np.log(np.cosh(np.einsum('bkj,bj->bk', A, z))), axis=1)
ref = lambda t: rb + rA * np.sin(rw * t + rp)
lam = lambda t: (l0 + l1 * np.sin(lw * t)) * np.where(sw & (t >= 20.0), 0.5, 1.0)

zs = np.zeros((B, 3))                                  # zeta* with gradH(zeta*) = theta*
for _ in range(100):
    s2 = 1 - np.tanh(np.einsum('bkj,bj->bk', A, zs)) ** 2
    Hess = M + np.einsum('bk,bki,bkj->bij', w * s2, A, A)
    zs = zs - np.linalg.solve(Hess, (gradH(zs) - th)[:, :, None])[:, :, 0]
Dst = lambda zeta: Hf(zeta) - Hf(zs) - np.einsum('bi,bi->b', th, zeta - zs)


def run(mode, sig=0.0, seed=1):
    r2 = np.random.default_rng(seed)
    x, xr = x0.copy(), x0.copy()
    z = gamma[:, None] * Gint(x0) if mode == "ii" else np.zeros((B, 3))
    out_D, out_e = [], []

    def f(t, x, xr, z, n):
        xm = x + n
        ph = phi(xm)
        r = ref(t)
        if mode == "ii":                               # mismatch-driven (I&I): zeta = z - gamma * Gint(x)
            zeta = z - gamma[:, None] * Gint(xm)
            zd = gamma[:, None] * ph * (-AM * xm + AM * r)[:, None]
        else:                                          # tracking-error-driven (Level-1 type)
            zeta = z
            zd = -gamma[:, None] * ph * (xm - xr)[:, None]
        v = np.einsum('bi,bi->b', ph, gradH(zeta))
        xd = -AM * x + AM * r + lam(t) * (g(v) - g(np.einsum('bi,bi->b', phi(x), th)))
        return xd, -AM * xr + AM * r, zd

    for i in range(NS):
        t = i * DT
        n = sig * r2.normal(size=B) if (sig > 0 and i % 5 == 0) else (n if sig > 0 else 0.0)
        if i % 50 == 0:
            zeta = z - gamma[:, None] * Gint(x) if mode == "ii" else z
            out_D.append(Dst(zeta)); out_e.append(x - xr)
        k1 = f(t, x, xr, z, n)
        k2 = f(t + DT / 2, x + DT / 2 * k1[0], xr + DT / 2 * k1[1], z + DT / 2 * k1[2], n)
        k3 = f(t + DT / 2, x + DT / 2 * k2[0], xr + DT / 2 * k2[1], z + DT / 2 * k2[2], n)
        k4 = f(t + DT, x + DT * k3[0], xr + DT * k3[1], z + DT * k3[2], n)
        x = x + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        xr = xr + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return np.array(out_D), np.array(out_e)


res = {}
for mode in ["ii", "l1"]:
    D, e = run(mode)
    res[mode] = (D, e)
    fin = np.isfinite(D).all(axis=0) & np.isfinite(e).all(axis=0)
    inc = np.nanmax(np.diff(D, axis=0) / np.maximum(D[0], 1e-12), axis=0)
    tail = np.sqrt(np.nanmean(e[-50:] ** 2, axis=0))
    print(f"{mode}: finite {fin.sum()}/{B}; cases where D ever increases by >1e-6*D0: {(inc > 1e-6).sum()}; "
          f"max relative step increase {np.nanmax(inc):.2e}; tail |e| median {np.nanmedian(tail):.2e}, worst {np.nanmax(tail):.2e}; "
          f"rms e over run median {np.nanmedian(np.sqrt(np.nanmean(e**2, axis=0))):.3f}")
for mode in ["ii", "l1"]:
    D, e = run(mode, sig=0.02)
    print(f"{mode} with measurement noise 0.02: finite {np.isfinite(e).all(axis=0).sum()}/{B}; rms e (last 10 s) median "
          f"{np.nanmedian(np.sqrt(np.nanmean(e[-100:]**2, axis=0))):.3f}, worst {np.nanmax(np.sqrt(np.nanmean(e[-100:]**2, axis=0))):.3f}")
t = np.arange(res["ii"][0].shape[0]) * 50 * DT
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for b in range(0, B, 4):
    c = plt.cm.viridis(b / B)
    ax[0].semilogy(t, res["ii"][0][:, b] / res["ii"][0][0, b], color=c, lw=1)
    ax[1].semilogy(t, np.clip(res["l1"][0][:, b] / res["l1"][0][0, b], 1e-8, 1e3), color=c, lw=1)
    ax[2].semilogy(t, np.abs(res["ii"][1][:, b]) + 1e-7, color=c, lw=0.8)
ax[0].set_title("mismatch-driven update: D(t)/D(0)", fontsize=9)
ax[1].set_title("tracking-error-driven update: D(t)/D(0)", fontsize=9)
ax[2].set_title("mismatch-driven update: |e|", fontsize=9)
for a in ax:
    a.set_xlabel("time [s]"); a.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("note/fig_level2.png", dpi=150)
