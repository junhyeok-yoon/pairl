import numpy as np

# Soundness beyond convexity: H = (mu/2)|z|^2 + bounded tanh network (non-convex, coercive);
# J, R, G_f, kappa depend on the operator state z as well as on the measured signal x.
rng = np.random.default_rng(3)
B, K, NA, NF, AM, MU, DT, T = 40, 6, 4, 3, 2.0, 0.5, 0.002, 40.0
NZ, NS = NA + NF, int(40.0 / 0.002)
W = rng.uniform(-3, 3, (B, K)); A = rng.normal(size=(B, K, NZ)); bb = rng.normal(size=(B, K))
P1 = rng.normal(size=(B, 1 + NZ)) * 0.7
A0, A1 = rng.normal(size=(B, NF, NF)), rng.normal(size=(B, NF, NF))
L0, L1 = rng.normal(size=(B, NF, NF)) * 0.6, rng.normal(size=(B, NF, NF)) * 0.6
g0, g1 = rng.normal(size=(B, NF)), rng.normal(size=(B, NF))
k0, k1 = rng.normal(size=B), rng.normal(size=B)
th, lam = rng.uniform(-2, 2, (B, 3)), rng.uniform(0.4, 1.0, B)
cst = np.concatenate([-th, (1 / lam - 1)[:, None]], 1)
rb, rA, rw, rp = rng.uniform(-1, 1, B), rng.uniform(0.3, 1, B), rng.uniform(0.5, 3, B), rng.uniform(0, 6.28, B)
x0 = rng.uniform(-1, 1, B)
ref = lambda t: rb + rA * np.sin(rw * t + rp)
fx = lambda x: np.stack([x, np.sin(2 * x), np.ones_like(x)], 1)


def Hfun(z):
    return 0.5 * MU * np.sum(z * z, 1) + np.sum(W * np.tanh(np.einsum('bkj,bj->bk', A, z) + bb), 1)


def gradH(z):
    t = np.tanh(np.einsum('bkj,bj->bk', A, z) + bb)
    return MU * z + np.einsum('bk,bkj->bj', W * (1 - t * t), A)


def minhess(z):
    t = np.tanh(np.einsum('bkj,bj->bk', A, z) + bb)
    Hs = MU * np.eye(NZ) + np.einsum('bk,bki,bkj->bij', W * (-2 * t * (1 - t * t)), A, A)
    return np.linalg.eigvalsh(Hs)[:, 0]


def f(t, x, xr, z):
    r = ref(t)
    ubl = -(1.0 + AM) * x + AM * r
    Ph = np.concatenate([fx(x), ubl[:, None]], 1)
    y = x - xr
    g = gradH(z); ga, gf = g[:, :NA], g[:, NA:]
    s = np.sin(np.einsum('bj,bj->b', P1, np.concatenate([x[:, None], z], 1)))
    Am = A0 + A1 * s[:, None, None]
    J = Am - np.transpose(Am, (0, 2, 1))
    L = L0 + L1 * np.tanh(s)[:, None, None]
    R = np.einsum('bij,bkj->bik', L, L)
    G = g0 + g1 * s[:, None]
    kap = np.logaddexp(0.0, k0 + k1 * np.tanh(s))
    uad = np.einsum('bi,bi->b', Ph, ga) + np.einsum('bi,bi->b', G, gf) - kap * y
    ustar = np.einsum('bi,bi->b', Ph, cst)
    xd = -AM * x + AM * r + lam * (uad - ustar)
    zd = np.concatenate([-Ph * y[:, None], np.einsum('bij,bj->bi', J - R, gf) - G * y[:, None]], 1)
    return xd, -AM * xr + AM * r, zd


x, xr, z = x0.copy(), x0.copy(), np.zeros((B, NZ))
Vs, es, mh, zmax = [], [], np.full(B, np.inf), np.zeros(B)
for i in range(NS):
    t = i * DT
    if i % 50 == 0:
        Vs.append(0.5 * (x - xr) ** 2 + lam * (Hfun(z) - np.einsum('bi,bi->b', cst, z[:, :NA])))
        es.append(x - xr); mh = np.minimum(mh, minhess(z)); zmax = np.maximum(zmax, np.abs(z).max(1))
    st = (x, xr, z)
    k1_ = f(t, *st)
    k2_ = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip(st, k1_)])
    k3_ = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip(st, k2_)])
    k4_ = f(t + DT, *[a + DT * b for a, b in zip(st, k3_)])
    x, xr, z = [a + DT / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(st, k1_, k2_, k3_, k4_)]
V, e = np.array(Vs), np.array(es)
dV = np.diff(V, axis=0)
print("finite:", int(np.isfinite(V).all(0).sum()), "/", B)
print("cases with a non-convex H along the trajectory (min Hessian eigenvalue < 0):", int((mh < 0).sum()), "/", B, "; most negative eigenvalue %.2f" % mh.min())
print("V ever increases: %d / %d ; max step increase %.2e (V range per case: median %.2f)" % (int((dV.max(0) > 1e-9).sum()), B, dV.max(), np.median(V[0] - V[-1])))
print("max |z| over all runs %.2f ; |e| last 5 s: median %.2e, worst %.2e" % (zmax.max(), np.median(np.abs(e[-50:]).max(0)), np.abs(e[-50:]).max()))
