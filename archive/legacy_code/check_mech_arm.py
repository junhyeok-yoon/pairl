import numpy as np, json, sys
from lib_yield_core import gauge, sat_dir, ramp

# Two-link arm with gravity (unknown inertia, payload and friction), output eta = dq + K1*q of relative degree one.
# Checks: (i) storage identity with the kinetic-energy storage and error-driven fast weights (point evaluation),
# (ii) closed loop with and without adaptation at the same gain, (iii) directional authority over the region.
N, DT, K1, KC, GA, UFF = 32, 5e-4, 4.0, 15.0, 40.0, 50.0
UBAR = np.array([60.0, 25.0]); ALPHA, M_R, C0, G0 = 2.0, 0.5, 0.5, 9.81
QLO, QHI = np.array([-0.2, 0.2]), np.array([1.2, 1.8])          # convex position constraint C_q; C = K1 * C_q

def theta_of(mp, f1, f2):
    m1, m2, l1, lc1 = 4.0, 2.0, 0.4, 0.2
    m2e = m2 + mp; lc2 = (m2 * 0.2 + mp * 0.4) / m2e
    I1 = m1 * l1 ** 2 / 12; I2 = m2 * 0.4 ** 2 / 12 + m2 * (lc2 - 0.2) ** 2 + mp * (0.4 - lc2) ** 2
    a1 = I1 + I2 + m1 * lc1 ** 2 + m2e * (l1 ** 2 + lc2 ** 2); a2 = m2e * l1 * lc2; a3 = I2 + m2e * lc2 ** 2
    return np.stack([a1, a2, a3, (m1 * lc1 + m2e * l1) * G0, m2e * lc2 * G0, f1, f2], -1)

def mass(q, th):
    c = np.cos(q[:, 1]); a1, a2, a3 = th[:, 0], th[:, 1], th[:, 2]
    return np.stack([np.stack([a1 + 2 * a2 * c, a3 + a2 * c], 1), np.stack([a3 + a2 * c, a3], 1)], 1)

def regressor(q, dq, v, a):
    # Y(q, dq, v, a) theta = M(q) a + C(q, dq) v + g(q) + F dq, with dM - 2C skew-symmetric
    c, s = np.cos(q[:, 1]), np.sin(q[:, 1]); c1, c12 = np.cos(q[:, 0]), np.cos(q[:, 0] + q[:, 1]); z = np.zeros(len(q))
    r1 = np.stack([a[:, 0], 2 * c * a[:, 0] + c * a[:, 1] - s * dq[:, 1] * v[:, 0] - s * (dq[:, 0] + dq[:, 1]) * v[:, 1], a[:, 1], c1, c12, dq[:, 0], z], 1)
    r2 = np.stack([z, c * a[:, 0] + s * dq[:, 0] * v[:, 0], a[:, 0] + a[:, 1], z, c12, z, dq[:, 1]], 1)
    return np.stack([r1, r2], 1)                                  # (n, 2, 7)

def field(q, dq, r, psi, ch, vp, psip, th, mode):
    eta = dq + K1 * q; e = eta - r; en = np.linalg.norm(e, axis=1); wn = en / psi
    k = 1.0 / np.maximum(1.0 - wn ** 2, 1e-12); xi = (k / psi ** 2)[:, None] * e; kap = KC * psi ** 2
    m_psi = 1.0 - wn
    clear = np.min(np.minimum(r - K1 * QLO, K1 * QHI - r), axis=1)
    m_c = np.clip((clear - psi) / C0, 0.0, 1.0)
    gr = ramp(np.minimum(m_psi, m_c), M_R); gp = ramp(m_psi, M_R)
    dr = gr[:, None] * vp
    Y = regressor(q, dq, dq - e, dr - K1 * dq)
    uad = np.einsum('nij,nj->ni', Y, ch)
    lff = np.minimum(1.0, UFF / np.maximum(np.linalg.norm(uad, axis=1), 1e-12))
    v = lff[:, None] * uad - kap[:, None] * xi
    u, lam = sat_dir(v, UBAR)
    M = mass(q, th)
    nvec = np.einsum('nij,nj->ni', regressor(q, dq, dq, np.zeros_like(dq)), th)      # C dq + g + F dq
    ddq = np.linalg.solve(M, (u - nvec)[..., None])[..., 0]
    dpsi = -ALPHA * gp * np.maximum(psi - psip, 0.0) + ALPHA * m_c * np.maximum(psip - psi, 0.0)
    sc = lam * lff if mode == 'adapt' else 0.0 * lam
    dch = -GA * sc[:, None] * np.einsum('nij,ni->nj', Y, e)
    ustar = np.einsum('nij,nj->ni', Y, th)
    aux = dict(e=e, wn=wn, xi=xi, kap=kap, lam=lam, lff=lff, ge=gp, u=u, ustar=ustar, M=M, ddq=ddq, dr=dr, dch=dch)
    return dq, ddq, dr, dpsi, dch, aux

def storage(q, dq, r, ch, th):
    e = dq + K1 * q - r
    return 0.5 * np.einsum('ni,nij,nj->n', e, mass(q, th), e) + np.sum((ch - th) ** 2, 1) / (2 * GA)

def identity_check(n=200000, seed=1):
    rng = np.random.default_rng(seed)
    th = theta_of(rng.uniform(0, 2, n), rng.uniform(0.2, 1, n), rng.uniform(0.2, 1, n))
    q = rng.uniform(QLO, QHI, (n, 2)); dq = rng.uniform(-1.5, 1.5, (n, 2)); psi = rng.uniform(0.1, 0.6, n)
    d = rng.normal(size=(n, 2)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = dq + K1 * q - d * (psi * rng.uniform(0, 0.98, n))[:, None]
    ch = th + rng.normal(size=(n, 7)) * rng.uniform(0, 8, n)[:, None]
    vp = rng.normal(size=(n, 2)) * 2.0
    _, ddq, dr, _, dch, a = field(q, dq, r, psi, ch, vp, rng.uniform(0.1, 0.6, n), th, 'adapt')
    e = a['e']; de = ddq + K1 * dq - dr
    sn = np.sin(q[:, 1]); a2 = th[:, 1]                                      # dM/dt = dM/dq2 * dq2
    dM = (-a2 * sn * dq[:, 1])[:, None, None] * np.array([[2.0, 1.0], [1.0, 0.0]])
    lhs = np.einsum('ni,nij,nj->n', e, a['M'], de) + 0.5 * np.einsum('ni,nij,nj->n', e, dM, e) + np.sum((ch - th) * dch, 1) / GA
    rhs = -a['lam'] * a['kap'] * np.einsum('ni,ni->n', e, a['xi']) - (1 - a['lam'] * a['lff']) * np.einsum('ni,ni->n', e, a['ustar'])
    return float(np.max(np.abs(lhs - rhs) / (1 + np.abs(rhs)))), float(np.mean(a['lam'] < 1)), float(np.mean(a['lff'] < 1))

def authority(n=20000, seed=2):
    # scale margin mu on the four edges of the input box, and the conservative class-level form
    rng = np.random.default_rng(seed)
    th = theta_of(rng.uniform(0, 2, n), rng.uniform(0.2, 1, n), rng.uniform(0.2, 1, n))
    q = rng.uniform(QLO, QHI, (n, 2)); dq = rng.uniform(-0.6, 0.6, (n, 2))
    M = mass(q, th); nvec = np.einsum('nij,nj->ni', regressor(q, dq, dq, np.zeros_like(dq)), th)
    td = nvec - K1 * np.einsum('nij,nj->ni', M, dq); A = np.linalg.inv(M); b = np.einsum('nij,nj->ni', A, td)
    g = np.linspace(-1, 1, 801); edges = []
    for i in range(2):
        for sg in (1.0, -1.0):
            v = np.zeros((801, 2)); v[:, i] = sg * UBAR[i]; v[:, 1 - i] = g * UBAR[1 - i]; edges.append(v)
    V = np.concatenate(edges)
    num = np.einsum('ki,nij,kj->nk', V, A, V); den = -np.einsum('ni,ki->nk', b, V)
    mu = np.min(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.inf), axis=1)
    ev = np.linalg.eigvalsh(M)
    rho0 = UBAR.min() / ev[:, 1] - np.linalg.norm(b, axis=1)
    return dict(mu_min=float(mu.min()), mu_med=float(np.median(mu)), mu_pos=float(np.mean(mu > 1)), rho0_pos=float(np.mean(rho0 > 0)),
                lam_min=float(ev[:, 0].min()), lam_max=float(ev[:, 1].max()))

def run(mode, T=40.0, seed=0):
    rng = np.random.default_rng(seed)
    th = theta_of(rng.uniform(0, 2, N), rng.uniform(0.2, 1, N), rng.uniform(0.2, 1, N))
    r = np.tile(np.array([1.0, 3.0]), (N, 1)); rq = r / K1; psi = np.full(N, 0.6)
    d0 = rng.normal(size=(N, 2)); q = rq + d0 * (0.05 / K1 / np.linalg.norm(d0, axis=1))[:, None]; dq = np.zeros((N, 2))
    ch = np.zeros((N, 7))
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    nwin = 5; win = T / nwin; nstep = int(T / DT)
    acc = {k: np.zeros((nwin, N)) for k in ['ge', 'wn', 'eff', 'pos', 'sat']}
    wmax = np.zeros(N); gmax = 0.0; qclear = np.inf; v_steps = v_inc = 0; v_max = -np.inf
    V_first = storage(q, dq, r, ch, th).mean()
    for i in range(nstep):
        t = i * DT
        vp = np.tile(1.0 * dirs[int(t / 2.0) % 4], (N, 1)); psip = np.full(N, 0.2)
        st = (q, dq, r, psi, ch)
        k1 = field(*st, vp, psip, th, mode)
        k2 = field(*[s + DT / 2 * d for s, d in zip(st, k1[:5])], vp, psip, th, mode)
        k3 = field(*[s + DT / 2 * d for s, d in zip(st, k2[:5])], vp, psip, th, mode)
        k4 = field(*[s + DT * d for s, d in zip(st, k3[:5])], vp, psip, th, mode)
        a = k1[5]; j = min(int(t / win), nwin - 1)
        acc['ge'][j] += a['ge'] * DT; acc['wn'][j] += a['wn'] * DT
        acc['eff'][j] += a['kap'] * np.linalg.norm(a['xi'], axis=1) * DT
        acc['pos'][j] += np.linalg.norm(q - rq, axis=1) * DT
        acc['sat'][j] += ((a['lam'] < 1) | (a['lff'] < 1)) * DT
        gmax = max(gmax, float(gauge(a['u'], UBAR).max()))
        V0 = storage(q, dq, r, ch, th)
        drq = r - K1 * rq                                                  # position reference: filtered command
        q, dq, r, psi, ch = [s + DT / 6 * (d1 + 2 * d2 + 2 * d3 + d4) for s, d1, d2, d3, d4 in zip(st, k1[:5], k2[:5], k3[:5], k4[:5])]
        rq = rq + DT * (drq + (r - K1 * (rq + DT * drq))) / 2
        V1 = storage(q, dq, r, ch, th)
        ok = (a['lam'] >= 1) & (a['lff'] >= 1) & (k4[5]['lam'] >= 1) & (k4[5]['lff'] >= 1)
        v_steps += int(ok.sum())
        if ok.any():
            inc = (V1 - V0)[ok]; v_inc += int((inc > 1e-10).sum()); v_max = max(v_max, float(inc.max()))
        wmax = np.maximum(wmax, np.linalg.norm(dq + K1 * q - r, axis=1) / psi)
        qclear = min(qclear, float(np.min(np.minimum(q - QLO, QHI - q))))
    f = lambda x: [float(v) for v in (x / win).mean(1)]
    return dict(law=mode, wmax=float(wmax.max()), gauge_max=gmax, q_clear_min=qclear, yield_err=f(acc['ge']), w_mean=f(acc['wn']),
                effort=f(acc['eff']), pos_err=f(acc['pos']), sat_frac=f(acc['sat']), V0=float(V_first),
                V_end=float(storage(q, dq, r, ch, th).mean()), v_checked=v_steps, v_increases=v_inc, v_max_increase=v_max)

if __name__ == '__main__':
    sel = sys.argv[1:] or ['id', 'none', 'adapt']
    if 'id' in sel:
        err, fs, ff = identity_check(); au = authority()
        print(f'identity: max relative difference {err:.3e} (input saturated {fs:.2f}, feedforward saturated {ff:.2f})')
        print('authority:', {k: round(v, 4) for k, v in au.items()}, flush=True)
        json.dump(dict(identity=err, frac_input_sat=fs, frac_ff_sat=ff, authority=au), open('../records/mech_arm_points.json', 'w'))
    for mode in [m for m in sel if m in ('none', 'adapt')]:
        o = run(mode)
        print(json.dumps({k: ([round(x, 4) for x in v] if isinstance(v, list) else (round(v, 6) if isinstance(v, float) else v)) for k, v in o.items()}), flush=True)
        open('../records/mech_arm.jsonl', 'a').write(json.dumps(o) + '\n')
