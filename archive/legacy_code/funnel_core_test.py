import numpy as np

# Barrier (funnel) adaptive core on the scalar plant: no prior parameter set, bounded residual and a bounded arbitrary
# "free policy" input act inside the commanded funnel psi(t). Compared with the same law without the barrier.
rng = np.random.default_rng(0)
N, A_P, A_M, DT, T = 64, 1.0, 2.0, 1e-4, 12.0
th = rng.uniform(-8, 8, (N, 3)); lam = rng.uniform(0.3, 1.0, N)
gam = 10 ** rng.uniform(0.0, 1.5, N); kap = 10 ** rng.uniform(-1.0, 0.5, N)
d0, dw = rng.uniform(-1, 1, N), rng.uniform(0.5, 4, N)
f0, fw = rng.uniform(-1, 1, N), rng.uniform(1, 8, N)
rA, rw = rng.uniform(0.3, 1.0, N), rng.uniform(0.4, 1.5, N)
psi = lambda t: 0.1 + 0.9 * np.exp(-0.8 * t) + 0.125 * (1 + np.sin(0.6 * t - 1.5))
fx = lambda x: np.stack([x, np.sin(2 * x), np.ones_like(x)], 1)


def run(barrier):
    x = np.zeros(N); xr = np.zeros(N); ch = np.zeros((N, 4))
    ratio = np.zeros(N); eff = []; alive = np.ones(N, bool)

    def der(t, x, xr, ch):
        r = rA * np.sin(rw * t)
        ubl = -(A_P + A_M) * x + A_M * r
        Phi = np.concatenate([fx(x), ubl[:, None]], 1)
        e = x - xr
        p = psi(t)
        xi = e / np.maximum(p * p - e * e, 1e-12) if barrier else e
        uad = np.einsum('ni,ni->n', Phi, ch) - kap * xi + f0 * np.sin(fw * t)
        xd = A_P * x + lam * (ubl + uad + np.einsum('ni,ni->n', th, fx(x)) + d0 * np.sin(dw * t))
        return xd, -A_M * xr + A_M * r, -gam[:, None] * Phi * xi[:, None], kap * np.abs(xi)

    for i in range(int(T / DT)):
        t = i * DT
        k1 = der(t, x, xr, ch)
        k2 = der(t + DT / 2, x + DT / 2 * k1[0], xr + DT / 2 * k1[1], ch + DT / 2 * k1[2])
        k3 = der(t + DT / 2, x + DT / 2 * k2[0], xr + DT / 2 * k2[1], ch + DT / 2 * k2[2])
        k4 = der(t + DT, x + DT * k3[0], xr + DT * k3[1], ch + DT * k3[2])
        x = x + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        xr = xr + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ch = ch + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        rr = np.abs(x - xr) / psi(t + DT)
        alive &= np.isfinite(rr)
        ratio = np.where(alive, np.maximum(ratio, np.nan_to_num(rr, nan=np.inf)), ratio)
        if i % 100 == 0:
            eff.append(k1[3])
    eff = np.array(eff)
    return ratio, alive, eff


for name, b in [("barrier core", True), ("same law without barrier", False)]:
    ratio, alive, eff = run(b)
    n = eff.shape[0]
    print(f"{name}: finite {alive.sum()}/{N}; cases leaving the funnel {(ratio >= 1).sum()}/{N}; max |e|/psi: median {np.median(ratio):.3f}, worst {ratio.max():.3f}; "
          f"feedback effort kappa|xi| (median): first 2 s {np.nanmedian(eff[: n // 6]):.3f}, last 2 s {np.nanmedian(eff[-n // 6:]):.3f}", flush=True)
