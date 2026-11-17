import numpy as np
from scipy.optimize import linprog

# Certified parameter set from the closed-loop storage identity (scalar plant, matched case).
# Plant: x' = a_p x + lam (u + th^T f(x)),  u = u_bl + u_ad,  u_bl = -(a_p+a_m) x + a_m r
# Error: e' = -a_m e + lam (u_ad - Phi^T c),  Phi = (f(x), u_bl),  c = (-th, 1/lam - 1)
# Law: chat' = -G Phi e,  u_ad = Phi^T chat - kap e
# Identity on [t1,t2]: c^T G^{-1} dchat = d(chat^T G^{-1} chat / 2) + kap int e^2 + (d(e^2/2) + a_m int e^2) / lam
rng = np.random.default_rng(0)
A_P, A_M, KAP, DT, T, WIN = 1.0, 2.0, 1.0, 1e-3, 40.0, 0.5
L0, L1 = 0.4, 1.0
f = lambda x: np.array([x, np.sin(2 * x), 1.0])
Gam = np.diag([20.0, 20.0, 20.0, 5.0])
Gi = np.linalg.inv(Gam)


def run(th, lam, ref, seed=0):
    c = np.concatenate([-th, [1.0 / lam - 1.0]])
    x = xr = 0.0
    ch = np.zeros(4)
    rows, rhs_lo, rhs_hi, snaps = [], [], [], {}
    w0 = None

    def deriv(t, x, xr, ch):
        r = ref(t)
        ubl = -(A_P + A_M) * x + A_M * r
        Phi = np.concatenate([f(x), [ubl]])
        e = x - xr
        uad = Phi @ ch - KAP * e
        return A_P * x + lam * (ubl + uad + th @ f(x)), -A_M * xr + A_M * r, -Gam @ Phi * e, e

    acc_e2 = 0.0
    for i in range(int(T / DT)):
        t = i * DT
        if w0 is None:
            w0 = (x - xr, ch.copy(), acc_e2)
        k1 = deriv(t, x, xr, ch)
        k2 = deriv(t + DT / 2, x + DT / 2 * k1[0], xr + DT / 2 * k1[1], ch + DT / 2 * k1[2])
        k3 = deriv(t + DT / 2, x + DT / 2 * k2[0], xr + DT / 2 * k2[1], ch + DT / 2 * k2[2])
        k4 = deriv(t + DT, x + DT * k3[0], xr + DT * k3[1], ch + DT * k3[2])
        acc_e2 += DT / 6 * (k1[3] ** 2 + 2 * k2[3] ** 2 + 2 * k3[3] ** 2 + k4[3] ** 2)
        x += DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        xr += DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ch = ch + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if (i + 1) % int(WIN / DT) == 0:
            e0, ch0, a0 = w0
            e1, ie2 = x - xr, acc_e2 - a0
            g = Gi @ (ch - ch0)
            b0 = 0.5 * ch @ Gi @ ch - 0.5 * ch0 @ Gi @ ch0 + KAP * ie2
            E = 0.5 * e1 ** 2 - 0.5 * e0 ** 2 + A_M * ie2
            lo, hi = sorted([b0 + E / L0, b0 + E / L1])
            tol = 1e-6 * (1.0 + abs(b0) + abs(E) + np.abs(g).sum())
            rows.append(g); rhs_lo.append(lo - tol); rhs_hi.append(hi + tol)
            w0 = None
            if abs(t + DT - 10.0) < 1e-6 or abs(t + DT - T) < 1e-6:
                snaps[round(t + DT)] = (np.array(rows), np.array(rhs_lo), np.array(rhs_hi))
    return c, snaps


def box_widths(c, rows, lo, hi):
    # prior set: |th_i| <= 4, 1/lam - 1 in [0, 1.5]
    bnd = [(-4, 4), (-4, 4), (-4, 4), (0.0, 1.5)]
    A = np.vstack([rows, -rows]); b = np.concatenate([hi, -lo])
    inside = bool(np.all(rows @ c <= hi + 1e-6) and np.all(rows @ c >= lo - 1e-6))
    w = []
    for k in range(4):
        e = np.zeros(4); e[k] = 1.0
        r1 = linprog(e, A_ub=A, b_ub=b, bounds=bnd, method="highs"); r2 = linprog(-e, A_ub=A, b_ub=b, bounds=bnd, method="highs")
        if r1.status != 0 or r2.status != 0:
            return False, np.full(4, np.nan)
        mn, mx = r1.fun, -r2.fun
        w.append(mx - mn)
    return inside, np.array(w)


refs = {"smooth reference": lambda t: 0.5 * np.sin(0.7 * t),
        "probing reference": lambda t: 0.5 * np.sin(0.7 * t) + 0.8 * np.sin(2.3 * t + 1.0) + 0.6 * np.sign(np.sin(0.45 * t))}
prior = np.array([8.0, 8.0, 8.0, 1.5])
for name, ref in refs.items():
    W10, WT, ok = [], [], True
    for k in range(8):
        th = rng.uniform(-2, 2, 3); lam = rng.uniform(L0, L1)
        c, snaps = run(th, lam, ref)
        for tt, (R, lo, hi) in snaps.items():
            inside, w = box_widths(c, R, lo, hi)
            ok = ok and inside
            (W10 if tt == 10 else WT).append(w / prior)
    print(f"{name}: true parameter always inside the certified set: {ok}; "
          f"box width / prior width (median over 8 tasks) at 10 s {np.round(np.median(W10, 0), 3)}, at 40 s {np.round(np.median(WT, 0), 3)}", flush=True)
