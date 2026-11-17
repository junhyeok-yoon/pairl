import numpy as np

rng = np.random.default_rng(1)
B, K, NZ, AM, DT, T = 30, 4, 7, 2.0, 0.002, 30.0
NS = int(T / DT)


def Gf(x):      # features as the Jacobian of a potential map (integrable by construction), shape (B, 7, 2)
    x1, x2 = x[:, 0], x[:, 1]
    o, z = np.ones_like(x1), np.zeros_like(x1)
    return np.stack([np.stack([x1, z], 1), np.stack([z, x2], 1), np.stack([x2, x1], 1), np.stack([np.cos(x1), z], 1),
                     np.stack([z, np.cos(x2)], 1), np.stack([o, z], 1), np.stack([z, o], 1)], axis=1)


def Bf(x):
    x1, x2 = x[:, 0], x[:, 1]
    return np.stack([x1 ** 2 / 2, x2 ** 2 / 2, x1 * x2, np.sin(x1), np.sin(x2), x1, x2], axis=1)


L = rng.normal(size=(B, NZ, NZ)) * 0.3
M = np.einsum('bij,bkj->bik', L, L) + 0.3 * np.eye(NZ)
A = rng.normal(size=(B, K, NZ))
w = rng.uniform(0, 1.5, size=(B, K))
gam = 10 ** rng.uniform(-0.3, 1.3, size=B)
th = rng.uniform(-1.5, 1.5, size=(B, NZ))
Q = np.linalg.qr(rng.normal(size=(B, 2, 2)))[0]
S0 = np.einsum('bij,bj,bkj->bik', Q, rng.uniform(0.3, 1.0, size=(B, 2)), Q)      # symmetric PD part
ks, kw, cw = rng.uniform(0, 0.6, B), rng.uniform(0.3, 2, B), rng.uniform(0.3, 2, B)
SK = np.array([[0.0, 1.0], [-1.0, 0.0]])
Lam = lambda t: S0 * (1 + 0.25 * np.sin(cw * t))[:, None, None] + (ks * np.sin(kw * t))[:, None, None] * SK   # non-symmetric, time-varying
rA, rw, rp = rng.uniform(0.3, 1.0, (B, 2)), rng.uniform(0.5, 2.5, (B, 2)), rng.uniform(0, 6.28, (B, 2))
ref = lambda t: rA * np.sin(rw * t + rp)
gradH = lambda z: np.einsum('bij,bj->bi', M, z) + np.einsum('bk,bkj->bj', w * np.tanh(np.einsum('bkj,bj->bk', A, z)), A)
Hf = lambda z: 0.5 * np.einsum('bi,bij,bj->b', z, M, z) + np.sum(w * np.log(np.cosh(np.einsum('bkj,bj->bk', A, z))), axis=1)
zs = np.zeros((B, NZ))
for _ in range(200):
    s2 = 1 - np.tanh(np.einsum('bkj,bj->bk', A, zs)) ** 2
    zs = zs - np.linalg.solve(M + np.einsum('bk,bki,bkj->bij', w * s2, A, A), (gradH(zs) - th)[:, :, None])[:, :, 0]
print("zeta* residual:", np.abs(gradH(zs) - th).max())
Dst = lambda zeta: Hf(zeta) - Hf(zs) - np.einsum('bi,bi->b', th, zeta - zs)
x0 = rng.uniform(-1, 1, (B, 2))


def run(mode):
    x, xr = x0.copy(), x0.copy()
    z = gam[:, None] * Bf(x0) if mode == "ii" else np.zeros((B, NZ))
    Ds, es = [], []

    def f(t, x, xr, z):
        G, r = Gf(x), ref(t)
        if mode == "ii":
            zeta = z - gam[:, None] * Bf(x)
            zd = gam[:, None] * np.einsum('bij,bj->bi', G, -AM * x + AM * r)
        else:
            zeta = z
            zd = -gam[:, None] * np.einsum('bij,bj->bi', G, x - xr)
        vt = np.einsum('bij,bi->bj', G, gradH(zeta) - th)
        return -AM * x + AM * r + np.einsum('bij,bj->bi', Lam(t), vt), -AM * xr + AM * r, zd

    for i in range(NS):
        t = i * DT
        if i % 50 == 0:
            Ds.append(Dst(z - gam[:, None] * Bf(x) if mode == "ii" else z)); es.append(np.linalg.norm(x - xr, axis=1))
        k1 = f(t, x, xr, z)
        k2 = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip((x, xr, z), k1)])
        k3 = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip((x, xr, z), k2)])
        k4 = f(t + DT, *[a + DT * b for a, b in zip((x, xr, z), k3)])
        x, xr, z = [a + DT / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip((x, xr, z), k1, k2, k3, k4)]
    return np.array(Ds), np.array(es)


for mode in ["ii", "l1"]:
    D, e = run(mode)
    fin = np.isfinite(D).all(0) & np.isfinite(e).all(0)
    inc = np.nanmax(np.diff(D, axis=0) / np.maximum(np.abs(D[0]), 1e-12), axis=0)
    print(f"{mode}: finite {fin.sum()}/{B}; D ever increases (>1e-6 D0): {(inc[fin] > 1e-6).sum()}; max rel. step increase {np.nanmax(inc[fin]) if fin.any() else np.nan:.2e}; "
          f"min D {np.nanmin(D[:, fin]) if fin.any() else np.nan:.2e}; rms |e| median {np.nanmedian(np.sqrt(np.nanmean(e[:, fin]**2, axis=0))) if fin.any() else np.nan:.3f}; tail |e| median {np.nanmedian(e[-50:, fin].mean(0)) if fin.any() else np.nan:.2e}")
