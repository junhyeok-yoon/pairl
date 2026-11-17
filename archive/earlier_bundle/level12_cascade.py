import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1-DOF joint with a joint-torque sensor: link side has unknown inertia/damping/gravity (Level 1, tracking pairing),
# motor side has friction that is nonlinear in its parameters (Level 2, mismatch pairing through the known rotor inertia)
rng = np.random.default_rng(2)
N, KQ, DT, T = 30, 3, 0.001, 30.0
NS = int(T / DT)
BR, LAMS, KD, FB = 0.2, 4.0, 3.0, 2.0                  # rotor inertia (known), sliding slope, damping gain, friction bound (known)
sig = lambda v: FB * np.tanh(v)                        # monotone, FB-Lipschitz
Jl, bv, mg = rng.uniform(0.2, 1.0, N), rng.uniform(0.1, 0.8, N), rng.uniform(2.0, 10.0, N)
a_true = np.stack([BR + Jl, bv, mg], 1)
a_nom = np.array([BR + 0.4, 0.3, 4.0])
th_f = np.stack([rng.uniform(2.0, 15.0, N), rng.uniform(-0.5, 0.5, N)], 1)     # friction sharpness and asymmetry
gam_a, gam_f = 10 ** rng.uniform(0.0, 1.5, N), 10 ** rng.uniform(0.0, 1.5, N)
rA, rw, rp = rng.uniform(0.3, 1.0, (N, 2)), rng.uniform(0.5, 3.0, (N, 2)), rng.uniform(0, 6.28, (N, 2))


def mk(n):
    L = rng.normal(size=(N, n, n)) * 0.4
    M = np.einsum('bij,bkj->bik', L, L) + 0.5 * np.eye(n)
    A, w = rng.normal(size=(N, KQ, n)), rng.uniform(0, 1.0, size=(N, KQ))
    g = lambda z: np.einsum('bij,bj->bi', M, z) + np.einsum('bk,bkj->bj', w * np.tanh(np.einsum('bkj,bj->bk', A, z)), A)
    H = lambda z: 0.5 * np.einsum('bi,bij,bj->b', z, M, z) + np.sum(w * np.log(np.cosh(np.einsum('bkj,bj->bk', A, z))), axis=1)

    def star(c):
        z = np.zeros((N, n))
        for _ in range(200):
            s2 = 1 - np.tanh(np.einsum('bkj,bj->bk', A, z)) ** 2
            z = z - np.linalg.solve(M + np.einsum('bk,bki,bkj->bij', w * s2, A, A), (g(z) - c)[:, :, None])[:, :, 0]
        return z
    return g, H, star


gHa, Ha, stara = mk(3)
gHf, Hf, starf = mk(2)
za_s, zf_s = stara(a_true - a_nom), starf(th_f)
Da = lambda z: Ha(z) - Ha(za_s) - np.einsum('bi,bi->b', a_true - a_nom, z - za_s)
Df = lambda z: Hf(z) - Hf(zf_s) - np.einsum('bi,bi->b', th_f, z - zf_s)
cw = FB / (2 * (KD + bv) * gam_f)                      # weight of the Level-2 storage in the joint certificate


def refs(t):
    qd = np.sum(rA * np.sin(rw * t[..., None] + rp), -1) if np.ndim(t) else np.sum(rA * np.sin(rw * t + rp), 1)
    dqd = np.sum(rA * rw * np.cos(rw * t + rp), 1)
    ddqd = -np.sum(rA * rw ** 2 * np.sin(rw * t + rp), 1)
    return qd, dqd, ddqd


def run(mode, noise=0.0, seed=0):
    r2 = np.random.default_rng(seed)
    q, dq = refs(0.0)[0].copy(), refs(0.0)[1].copy()
    za = np.zeros((N, 3))
    zf = gam_f[:, None] * BR * np.stack([dq ** 2 / 2, dq], 1) if mode == "cascade" else np.zeros((N, 2))
    Vs, es = [], []

    def f(t, q, dq, za, zf, nt):
        qd, dqd, ddqd = refs(t)
        qt, dqt = q - qd, dq - dqd
        s = dqt + LAMS * qt
        Y = np.stack([ddqd - LAMS * dqt, dqd - LAMS * qt, np.sin(q)], 1)
        G = np.stack([dq, np.ones_like(dq)], 1)
        zeta = zf - gam_f[:, None] * BR * np.stack([dq ** 2 / 2, dq], 1) if mode == "cascade" else zf
        tf_hat = sig(np.einsum('bi,bi->b', G, gHf(zeta)))
        tau = np.einsum('bi,bi->b', Y, a_nom + gHa(za)) - KD * s + tf_hat
        tf = sig(np.einsum('bi,bi->b', G, th_f))
        ddq = (tau - tf - bv * dq - mg * np.sin(q)) / a_true[:, 0]
        tauJ = tau - tf - BR * ddq + nt                                        # joint-torque sensor reading
        zad = -gam_a[:, None] * Y * s[:, None]
        zfd = gam_f[:, None] * G * (tau - tauJ - tf_hat)[:, None] if mode == "cascade" else -gam_f[:, None] * G * s[:, None]
        return dq, ddq, zad, zfd

    for i in range(NS):
        t = i * DT
        nt = noise * r2.normal(size=N) if noise > 0 else 0.0
        if i % 100 == 0:
            qd, dqd, _ = refs(t)
            s = (dq - dqd) + LAMS * (q - qd)
            zeta = zf - gam_f[:, None] * BR * np.stack([dq ** 2 / 2, dq], 1) if mode == "cascade" else zf
            Vs.append(0.5 * a_true[:, 0] * s ** 2 + Da(za) / gam_a + cw * Df(zeta)); es.append(q - qd)
        st = (q, dq, za, zf)
        k1 = f(t, *st, nt)
        k2 = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip(st, k1)], nt)
        k3 = f(t + DT / 2, *[a + DT / 2 * b for a, b in zip(st, k2)], nt)
        k4 = f(t + DT, *[a + DT * b for a, b in zip(st, k3)], nt)
        q, dq, za, zf = [a + DT / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(st, k1, k2, k3, k4)]
    return np.array(Vs), np.degrees(np.array(es))


out = {}
for mode in ["cascade", "tracking"]:
    V, e = run(mode)
    out[mode] = (V, e)
    inc = np.nanmax(np.diff(V, axis=0) / V[0], axis=0)
    fin = np.isfinite(V).all(0)
    print(f"{mode}: finite {fin.sum()}/{N}; joint V ever increases (>1e-6 V0): {(inc > 1e-6).sum()}; max rel. step increase {np.nanmax(inc):.2e}; "
          f"rms e {np.nanmedian(np.sqrt(np.nanmean(e**2, 0))):.3f} deg (median); last 5 s rms {np.nanmedian(np.sqrt(np.nanmean(e[-50:]**2, 0))):.4f} deg, worst {np.nanmax(np.sqrt(np.nanmean(e[-50:]**2, 0))):.3f}")
for mode in ["cascade", "tracking"]:
    V, e = run(mode, noise=0.05)
    print(f"{mode}, torque-sensor noise 0.05 N m: finite {np.isfinite(e).all(0).sum()}/{N}; last 5 s rms {np.nanmedian(np.sqrt(np.nanmean(e[-50:]**2, 0))):.4f} deg (median), worst {np.nanmax(np.sqrt(np.nanmean(e[-50:]**2, 0))):.3f}")
t = np.arange(out["cascade"][0].shape[0]) * 100 * DT
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for b in range(0, N, 3):
    c = plt.cm.viridis(b / N)
    ax[0].semilogy(t, out["cascade"][0][:, b] / out["cascade"][0][0, b], color=c, lw=1)
    ax[1].semilogy(t, np.clip(out["tracking"][0][:, b] / out["tracking"][0][0, b], 1e-9, 1e3), color=c, lw=1)
    ax[2].semilogy(t, np.abs(out["cascade"][1][:, b]) + 1e-6, color=c, lw=0.7)
ax[0].set_title("two-layer (Level 1 + Level 2): joint V(t)/V(0)", fontsize=9)
ax[1].set_title("tracking-driven friction update: same V(t)/V(0)", fontsize=9)
ax[2].set_title("two-layer: |q - q_d| [deg]", fontsize=9)
for a in ax:
    a.set_xlabel("time [s]"); a.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("note/fig_cascade.png", dpi=150)
