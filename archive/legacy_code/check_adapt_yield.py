import numpy as np, json, sys
from lib_yield_core import gauge, sat_dir, ramp

# Isotropic unknown gain, exact representation, fixed normalized feedback gain (kappa = kc * psi_g^2).
# Question: does fast-weight adaptation remove the yielding that feedback alone needs at this gain?
N, DT, GA, KC, UFF = 32, 4e-4, 0.5, 4.0, 1.0
UBAR = np.array([2.0, 3.0]); ALPHA, M_R, M_PSI, C0 = 2.0, 0.5, 0.5, 0.2
B_R = 0.5 * np.log(1.0 / (M_R * (2.0 - M_R)))      # storage level below which the error margin exceeds m_r

def phi(eta):
    return np.stack([np.ones(len(eta)), eta[:, 0], eta[:, 1], np.sin(2 * eta[:, 0]), np.cos(2 * eta[:, 1])], 1)

def field(eta, r, psi, W, vp, psip, gam, Th, scale_mode):
    e = eta - r; en = np.linalg.norm(e, axis=1); wn = en / psi
    k = 1.0 / np.maximum(1.0 - wn ** 2, 1e-12); xi = (k / psi ** 2)[:, None] * e
    kap = KC * psi ** 2
    m_psi = 1.0 - wn
    m_c = np.clip((1.0 - np.max(np.abs(r), axis=1) - psi) / C0, 0.0, 1.0)
    gr = ramp(np.minimum(m_psi, m_c), M_R); gp = ramp(m_psi, M_PSI)
    dr = gr[:, None] * vp
    f = phi(eta); Phi = np.concatenate([f, dr], 1)
    uad = np.einsum('nij,ni->nj', W, Phi)
    lff = np.minimum(1.0, UFF / np.maximum(np.linalg.norm(uad, axis=1), 1e-12)); uff = lff[:, None] * uad
    v = uff - kap[:, None] * xi
    u, lam = sat_dir(v, UBAR)
    ustar = dr / gam[:, None] - np.einsum('nij,ni->nj', Th, f)
    deta = gam[:, None] * (u - ustar) + dr                 # = F + gam*u with F = gam*Theta^T phi
    dpsi = -ALPHA * gp * np.maximum(psi - psip, 0.0) + ALPHA * m_c * np.maximum(psip - psi, 0.0)
    sc = {'none': 0.0 * lam, 'scaled': lam * lff, 'plain': np.ones_like(lam)}[scale_mode]
    dW = -GA * sc[:, None, None] * np.einsum('ni,nj->nij', Phi, xi)
    pump = -(wn ** 2 / np.maximum(1.0 - wn ** 2, 1e-12)) * dpsi / psi      # storage supplied by a contracting tolerance
    aux = dict(wn=wn, xi=xi, kap=kap, lam=lam, lff=lff, gr=gr, ustar=ustar, uff=uff, dr=dr, dpsi=dpsi, ge=ramp(m_psi, M_R), pump=pump)
    return deta, dr, dpsi, dW, aux

def storage(eta, r, psi, W, Wstar, gam):
    wn = np.linalg.norm(eta - r, axis=1) / psi
    return 0.5 * np.log(1.0 / (1.0 - wn ** 2)) + gam * np.sum((W - Wstar) ** 2, axis=(1, 2)) / (2 * GA)

def identity_check(n=200000, seed=1):
    # point evaluation: chain-rule derivative of V along the vector field vs the closed form
    global N
    N0, N = N, n
    rng = np.random.default_rng(seed)
    gam = rng.uniform(0.5, 2.0, n); Th = rng.normal(size=(n, 5, 2)) * 0.1
    Wstar = np.concatenate([-Th, np.tile(np.eye(2)[None], (n, 1, 1)) / gam[:, None, None]], 1)
    W = Wstar + rng.normal(size=(n, 7, 2)) * rng.uniform(0, 1.5, n)[:, None, None]
    r = rng.uniform(-0.5, 0.5, (n, 2)); psi = rng.uniform(0.05, 0.4, n)
    d = rng.normal(size=(n, 2)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    eta = r + d * (psi * rng.uniform(0.0, 0.98, n))[:, None]
    vp = rng.normal(size=(n, 2)) * 1.5; psip = rng.uniform(0.05, 0.4, n)
    deta, dr, dpsi, dW, a = field(eta, r, psi, W, vp, psip, gam, Th, 'scaled')
    wn = a['wn']; pump = -(wn ** 2 / (1 - wn ** 2)) * dpsi / psi
    lhs = np.einsum('ni,ni->n', a['xi'], deta - dr) + pump + gam * np.sum((W - Wstar) * dW, axis=(1, 2)) / GA
    rhs = -gam * a['lam'] * a['kap'] * np.sum(a['xi'] ** 2, 1) - gam * (1 - a['lam'] * a['lff']) * np.einsum('ni,ni->n', a['xi'], a['ustar']) + pump
    N = N0
    return float(np.max(np.abs(lhs - rhs) / (1 + np.abs(rhs)))), float(np.mean(a['lam'] < 1)), float(np.mean(a['lff'] < 1))

def run(scale_mode, vs, seg, T=40.0, seed=0):
    rng = np.random.default_rng(seed)
    gam = rng.uniform(0.5, 2.0, N)
    Th = rng.normal(size=(N, 5, 2)); Th *= (0.3 / np.sqrt(5)) / np.linalg.norm(Th, 2, axis=(1, 2))[:, None, None]
    Wstar = np.concatenate([-Th, np.tile(np.eye(2)[None], (N, 1, 1)) / gam[:, None, None]], 1)
    W = np.zeros((N, 7, 2)); r = np.full((N, 2), -0.3); psi = np.full(N, 0.4)
    e0 = rng.normal(size=(N, 2)); eta = r + e0 * (0.05 / np.linalg.norm(e0, axis=1))[:, None]
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    nwin = 5; win = T / nwin; nstep = int(T / DT)
    acc = {k: np.zeros((nwin, N)) for k in ['gr', 'ge', 'wn', 'eff', 'mis', 'sat', 'real', 'prop']}
    wmax = np.zeros(N); v_inc = 0; v_max_inc = 0.0; v_steps = 0
    Vtrace = []
    for i in range(nstep):
        t = i * DT
        vp = np.tile(vs * dirs[int(t / seg) % 4], (N, 1)); psip = np.full(N, 0.1)
        st = (eta, r, psi, W)
        k1 = field(*st, vp, psip, gam, Th, scale_mode)
        k2 = field(*[s + DT / 2 * d for s, d in zip(st, k1[:4])], vp, psip, gam, Th, scale_mode)
        k3 = field(*[s + DT / 2 * d for s, d in zip(st, k2[:4])], vp, psip, gam, Th, scale_mode)
        k4 = field(*[s + DT * d for s, d in zip(st, k3[:4])], vp, psip, gam, Th, scale_mode)
        a = k1[4]; j = min(int(t / win), nwin - 1)
        acc['gr'][j] += a['gr'] * DT; acc['ge'][j] += a['ge'] * DT; acc['wn'][j] += a['wn'] * DT
        acc['eff'][j] += a['kap'] * np.linalg.norm(a['xi'], axis=1) * DT
        acc['mis'][j] += np.linalg.norm(a['uff'] - a['ustar'], axis=1) * DT
        acc['sat'][j] += ((a['lam'] < 1) | (a['lff'] < 1)) * DT
        acc['real'][j] += np.linalg.norm(a['dr'], axis=1) * DT; acc['prop'][j] += vs * DT
        V0 = storage(eta, r, psi, W, Wstar, gam)
        eta, r, psi, W = [s + DT / 6 * (d1 + 2 * d2 + 2 * d3 + d4) for s, d1, d2, d3, d4 in zip(st, k1[:4], k2[:4], k3[:4], k4[:4])]
        V1 = storage(eta, r, psi, W, Wstar, gam)
        # on steps without any saturation: V1 - V0 - (supply from tolerance contraction) must be <= 0
        ok = (a['lam'] >= 1) & (a['lff'] >= 1) & (k4[4]['lam'] >= 1) & (k4[4]['lff'] >= 1)
        supply = DT / 6 * (a['pump'] + 2 * k2[4]['pump'] + 2 * k3[4]['pump'] + k4[4]['pump'])
        v_steps += int(ok.sum()); inc = (V1 - V0 - supply)[ok]
        if inc.size:
            v_inc += int((inc > 1e-9).sum()); v_max_inc = max(v_max_inc, float(inc.max()))
        wmax = np.maximum(wmax, np.linalg.norm(eta - r, axis=1) / psi)
        if i % 250 == 0:
            Vtrace.append(V1.copy())
    Vtrace = np.array(Vtrace)
    out = dict(law=scale_mode, vs=vs, wmax=float(wmax.max()),
               yield_factor=[float(x) for x in (acc['gr'] / win).mean(1)], yield_err=[float(x) for x in (acc['ge'] / win).mean(1)],
               w_mean=[float(x) for x in (acc['wn'] / win).mean(1)], effort=[float(x) for x in (acc['eff'] / win).mean(1)],
               mismatch=[float(x) for x in (acc['mis'] / win).mean(1)], sat_frac=[float(x) for x in (acc['sat'] / win).mean(1)],
               path_ratio=[float(x) for x in (acc['real'] / acc['prop']).mean(1)],
               psi_end=float(psi.mean()), V0=float(Vtrace[0].mean()), V_end=float(Vtrace[-1].mean()),
               frac_below_Br_end=float(np.mean(Vtrace[-1] < B_R)), v_checked=v_steps, v_increases=v_inc, v_max_increase=v_max_inc)
    return out

if __name__ == '__main__':
    runs = [('none', 0.3, 2.0), ('scaled', 0.3, 2.0), ('plain', 0.3, 2.0), ('none', 1.5, 0.4), ('scaled', 1.5, 0.4), ('plain', 1.5, 0.4)]
    sel = [int(i) for i in sys.argv[1:]] or range(len(runs))
    if 0 in sel:
        err, fs, ff = identity_check()
        print(f'identity: max relative difference {err:.3e} over 200000 points (input saturated {fs:.2f}, feedforward saturated {ff:.2f})', flush=True)
        open('../records/log_identity.txt', 'w').write(f'{err:.6e} {fs:.4f} {ff:.4f}\n')
    for i in sel:
        o = run(*runs[i]); o['run'] = i
        print(json.dumps({k: ([round(x, 4) for x in v] if isinstance(v, list) else (round(v, 6) if isinstance(v, float) else v)) for k, v in o.items()}), flush=True)
        open('../records/adapt_yield.jsonl', 'a').write(json.dumps(o) + '\n')
