import numpy as np
from scipy.optimize import linprog

# Does tolerance buy information? Barrier core on the scalar plant, certified set from the barrier-storage identity.
# Settings: narrow / wide commanded tolerance, with / without a bounded probing input inside the funnel.
rng = np.random.default_rng(1)
N, A_P, A_M, KAP, DT, T, WIN = 8, 1.0, 2.0, 1.0, 1e-4, 10.0, 0.5
L0, L1 = 0.4, 1.0
Gam = np.array([20.0, 20.0, 20.0, 5.0]); Gi = 1.0 / Gam
th = rng.uniform(-2, 2, (N, 3)); lam = rng.uniform(L0, L1, N)
ctrue = np.concatenate([-th, (1.0 / lam - 1.0)[:, None]], 1)
fx = lambda x: np.stack([x, np.sin(2 * x), np.ones_like(x)], 1)
probe = lambda t: 0.6 * np.sin(3.1 * t) + 0.4 * np.sin(7.3 * t + 1.0)


def run(psi, use_probe, rich=False, lam_known=False):
    x = np.zeros(N); xr = np.zeros(N); ch = np.zeros((N, 4))
    acc = np.zeros((N, 3))                       # int xi^2, int xi*nu, int e*xi
    rows = [[] for _ in range(N)]; lo = [[] for _ in range(N)]; hi = [[] for _ in range(N)]
    w0 = None; ratio = np.zeros(N); eff = []

    def der(t, x, xr, ch):
        r = 0.5 * np.sin(0.7 * t) + (0.8 * np.sin(2.3 * t + 1.0) + 0.6 * np.tanh(5 * np.sin(0.45 * t)) if rich else 0.0)
        ubl = -(A_P + A_M) * x + A_M * r
        Phi = np.concatenate([fx(x), ubl[:, None]], 1)
        e = x - xr
        xi = e / np.maximum(psi * psi - e * e, 1e-12)
        nu = probe(t) if use_probe else 0.0
        uad = np.einsum('ni,ni->n', Phi, ch) - KAP * xi + nu
        xd = A_P * x + lam * (ubl + uad + np.einsum('ni,ni->n', th, fx(x)))
        return xd, -A_M * xr + A_M * r, -Gam * Phi * xi[:, None], np.stack([xi * xi, xi * nu * np.ones(N), e * xi], 1)

    Bf = lambda e: 0.5 * np.log(psi * psi / (psi * psi - e * e))
    nwin = int(WIN / DT)
    for i in range(int(T / DT)):
        t = i * DT
        if w0 is None:
            w0 = (x - xr, ch.copy(), acc.copy())
        k1 = der(t, x, xr, ch)
        k2 = der(t + DT / 2, x + DT / 2 * k1[0], xr + DT / 2 * k1[1], ch + DT / 2 * k1[2])
        k3 = der(t + DT / 2, x + DT / 2 * k2[0], xr + DT / 2 * k2[1], ch + DT / 2 * k2[2])
        k4 = der(t + DT, x + DT * k3[0], xr + DT * k3[1], ch + DT * k3[2])
        acc = acc + DT / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        x = x + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        xr = xr + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ch = ch + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        ratio = np.maximum(ratio, np.abs(x - xr) / psi)
        if i % 200 == 0 and t >= T - 2.0:
            eff.append(KAP * np.abs((x - xr) / (psi * psi - (x - xr) ** 2)))
        if (i + 1) % nwin == 0:
            e0, ch0, a0 = w0
            e1 = x - xr; d = acc - a0
            g = Gi * (ch - ch0)
            b0 = 0.5 * np.sum(ch * Gi * ch, 1) - 0.5 * np.sum(ch0 * Gi * ch0, 1) + KAP * d[:, 0] - d[:, 1]
            E = Bf(e1) - Bf(e0) + A_M * d[:, 2]
            a, b = (b0 + E / lam, b0 + E / lam) if lam_known else (b0 + E / L0, b0 + E / L1)
            tol = 1e-6 * (1.0 + np.abs(b0) + np.abs(E) + np.abs(g).sum(1))
            for n in range(N):
                rows[n].append(g[n]); lo[n].append(min(a[n], b[n]) - tol[n]); hi[n].append(max(a[n], b[n]) + tol[n])
            w0 = None
    bnd = [(-4, 4), (-4, 4), (-4, 4), (0.0, 1.5)]; prior = np.array([8.0, 8.0, 8.0, 1.5])
    widths, inside = [], True
    for n in range(N):
        R = np.array(rows[n]); A = np.vstack([R, -R]); bb = np.concatenate([hi[n], -np.array(lo[n])])
        inside = inside and bool(np.all(R @ ctrue[n] <= np.array(hi[n]) + 1e-6) and np.all(R @ ctrue[n] >= np.array(lo[n]) - 1e-6))
        w = []
        for k in range(4):
            c = np.zeros(4); c[k] = 1.0
            r1 = linprog(c, A_ub=A, b_ub=bb, bounds=bnd, method="highs"); r2 = linprog(-c, A_ub=A, b_ub=bb, bounds=bnd, method="highs")
            w.append((-r2.fun - r1.fun) if (r1.status == 0 and r2.status == 0) else np.nan)
        widths.append(np.array(w) / prior)
    return inside, np.nanmedian(np.array(widths), 0), ratio.max(), np.median(np.array(eff))


for name, kw in [("smooth reference", dict()), ("smooth reference + input probing", dict(use_probe=True)), ("rich reference", dict(rich=True))]:
    for lk in [False, True]:
        a = dict(use_probe=False); a.update(kw)
        inside, w, rmax, eff = run(0.3, a["use_probe"], rich=a.get("rich", False), lam_known=lk)
        print(f"{name}, effectiveness {'known' if lk else 'in an interval'}: truth inside {inside}; set width / prior after 10 s (mean of 4) {np.nanmean(w):.4f}; max |e|/psi {rmax:.2f}; feedback effort last 2 s {eff:.3f}", flush=True)
