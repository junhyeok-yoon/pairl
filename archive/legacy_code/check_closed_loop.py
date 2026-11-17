import numpy as np, pickle, sys
from lib_yield_core import *

# Closed-loop sanity check of the composed loop (plant, core, interface), relative degree one, two inputs.
# Proposals are random and aggressive, the feedforward is adversarial but bounded, plants are random class members.
N, DT, T, TH = 64, 2e-4, 20.0, 0.25
J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])

def make_case(seed, coupled, vbar_prop=None):
    rng = np.random.default_rng(seed)
    if coupled:   # strongly coupled symmetric gain with unequal input limits
        P = dict(g0=0.5, g1=9.5, Fbar=0.6, abar=1.0, kap0=1.0, alpha=2.0, psi_low=0.05, psi_max=0.4,
                 vbar=3.0, m_r=0.5, m_psi=0.5)
        ubar = np.array([2.0, 8.0])
    else:
        P = dict(g0=0.5, g1=2.0, Fbar=0.6, abar=1.0, kap0=1.0, alpha=2.0, psi_low=0.05, psi_max=0.4,
                 vbar=3.0, m_r=0.5, m_psi=0.5)
        ubar = np.array([2.0, 3.0])
    P['umin'] = ubar.min()
    c = dict(phi0=rng.uniform(0, np.pi, N), c1=rng.uniform(-2, 2, N), c2=rng.uniform(-2, 2, N),
             lam_hi=rng.uniform(0.8, 1.7, N), sk=rng.uniform(0, 0.3, N),
             fa=rng.uniform(-4, 4, (N, 2, 2)), fb=rng.uniform(0, 6, (N, 2)),
             dw=rng.uniform(0.5, 5, (N, 2)), aw=rng.uniform(1, 9, (N, 3, 2)), aph=rng.uniform(0, 6, (N, 3, 2)),
             adv_ff=rng.random(N) < 0.5, adv_F=rng.random(N) < 0.5)
    K = int(T / TH) + 1
    vdir = rng.normal(size=(K, N, 2)); vdir /= np.linalg.norm(vdir, axis=2, keepdims=True)
    acts = dict(vp=(P['vbar'] if vbar_prop is None else vbar_prop) * vdir, psip=rng.choice([0.05, 0.1, 0.2, 0.4], size=(K, N)))
    if vbar_prop is not None:   # benign setting: no adversarial drift or feedforward
        c['adv_ff'][:] = False; c['adv_F'][:] = False
    r0 = rng.uniform(-0.5, 0.5, (N, 2))
    e0 = rng.normal(size=(N, 2)); e0 *= (0.9 * rng.random(N) * 0.4 / np.linalg.norm(e0, axis=1))[:, None]
    return P, ubar, c, acts, r0, e0

def gamma_of(eta, c, coupled):
    if coupled:
        return np.broadcast_to(5.0 * np.array([[1.0, 0.9], [0.9, 1.0]]), (N, 2, 2))
    phi = c['phi0'] + c['c1'] * eta[:, 0] + c['c2'] * eta[:, 1]
    cs, sn = np.cos(phi), np.sin(phi)
    R = np.stack([np.stack([cs, -sn], 1), np.stack([sn, cs], 1)], 1)
    l2 = 0.5 + (c['lam_hi'] - 0.5) * (0.5 + 0.5 * np.sin(3 * eta[:, 0] - 2 * eta[:, 1]))
    lam = np.stack([np.full(N, 0.5), l2], 1)
    return np.einsum('nij,nj,nkj->nik', R, lam, R) + c['sk'][:, None, None] * J2

def run(mode, coupled, seed=0, record=None, vbar_prop=None):
    P, ubar, c, acts, r0, e0 = make_case(seed, coupled, vbar_prop)
    ms, rho0 = margin_star(P)
    g1 = P['g1']; fbar = 0.3; dbar = (P['Fbar'] - fbar) / g1
    r = r0.copy(); psi = np.full(N, 0.4); eta = r0 + e0
    alive = np.ones(N, bool)
    stat = dict(wmax=np.zeros(N), gmax=np.zeros(N), eta_clear=np.full(N, np.inf), tube_clear=np.full(N, np.inf),
                sat_time=np.zeros(N), path_real=np.zeros(N), path_prop=np.zeros(N), yield_mean=np.zeros(N), yield_err=np.zeros(N))
    w0 = np.linalg.norm(e0, axis=1) / 0.4
    trace = []

    def der(t, eta, r, psi, vp, psip):
        e = eta - r
        en = np.linalg.norm(e, axis=1)
        wn = en / psi
        what = e / np.maximum(en, 1e-12)[:, None]
        k = 1.0 / np.maximum(1.0 - wn ** 2, 1e-12)
        xi = (k / psi ** 2)[:, None] * e
        osc = np.stack([np.sin(c['aw'][:, :, 0] * t + c['aph'][:, :, 0]).mean(1),
                        np.cos(c['aw'][:, :, 1] * t + c['aph'][:, :, 1]).mean(1)], 1)
        a = np.where(c['adv_ff'][:, None], P['abar'] * what, P['abar'] * osc / np.sqrt(2))
        v = a - P['kap0'] * xi
        if mode == 'clip':
            u = sat_clip(v, ubar); lam = 1.0 / np.maximum(1.0, gauge(v, ubar))
        else:
            u, lam = sat_dir(v, ubar)
        Gam = gamma_of(eta, c, coupled)
        f = fbar * np.stack([np.sin(np.einsum('ni,ni->n', c['fa'][:, 0], eta) + c['fb'][:, 0]),
                             np.cos(np.einsum('ni,ni->n', c['fa'][:, 1], eta) + c['fb'][:, 1])], 1) / np.sqrt(2)
        d = dbar * np.stack([np.sin(c['dw'][:, 0] * t), np.cos(c['dw'][:, 1] * t)], 1) / np.sqrt(2)
        Fn = f + np.einsum('nij,nj->ni', Gam, d)
        Fv = np.where(c['adv_F'][:, None], P['Fbar'] * what, Fn)
        m_psi = 1.0 - wn
        clear = 1.0 - np.max(np.abs(r), axis=1)
        m_c = np.clip((clear - psi) / 0.2, 0.0, 1.0)
        if mode == 'noyield':
            gr = ramp(m_c, P['m_r']); gp = np.ones(N)
        else:
            gr = ramp(np.minimum(m_psi, m_c), P['m_r']); gp = ramp(m_psi, P['m_psi'])
        dr = gr[:, None] * vp
        dpsi = -P['alpha'] * gp * np.maximum(psi - psip, 0.0) + P['alpha'] * m_c * np.maximum(psip - psi, 0.0)
        deta = Fv + np.einsum('nij,nj->ni', Gam, u)
        return deta, dr, dpsi, dict(wn=wn, lam=lam, gr=gr, u=u, clear=clear, v=v, ge=ramp(m_psi, P['m_r']))

    nstep = int(T / DT)
    for i in range(nstep):
        t = i * DT; kk = int(t / TH + 1e-9)
        vp, psip = acts['vp'][kk], acts['psip'][kk]
        k1 = der(t, eta, r, psi, vp, psip)
        k2 = der(t + DT / 2, eta + DT / 2 * k1[0], r + DT / 2 * k1[1], psi + DT / 2 * k1[2], vp, psip)
        k3 = der(t + DT / 2, eta + DT / 2 * k2[0], r + DT / 2 * k2[1], psi + DT / 2 * k2[2], vp, psip)
        k4 = der(t + DT, eta + DT * k3[0], r + DT * k3[1], psi + DT * k3[2], vp, psip)
        s = k1[3]
        stat['gmax'] = np.where(alive, np.maximum(stat['gmax'], gauge(s['u'], ubar)), stat['gmax'])
        stat['sat_time'] += alive * (s['lam'] < 1.0) * DT
        stat['path_real'] += alive * np.linalg.norm(k1[1], axis=1) * DT
        stat['path_prop'] += alive * np.linalg.norm(vp, axis=1) * DT
        stat['yield_mean'] += alive * s['gr'] * DT
        stat['yield_err'] += alive * s['ge'] * DT
        upd = alive[:, None]
        eta = np.where(upd, eta + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), eta)
        r = np.where(upd, r + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), r)
        psi = np.where(alive, psi + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), psi)
        wn = np.linalg.norm(eta - r, axis=1) / psi
        stat['wmax'] = np.where(alive, np.maximum(stat['wmax'], np.nan_to_num(wn, nan=np.inf)), stat['wmax'])
        stat['eta_clear'] = np.where(alive, np.minimum(stat['eta_clear'], 1.0 - np.max(np.abs(eta), axis=1)), stat['eta_clear'])
        stat['tube_clear'] = np.where(alive, np.minimum(stat['tube_clear'], 1.0 - np.max(np.abs(r), axis=1) - psi), stat['tube_clear'])
        alive &= np.isfinite(wn) & (wn < 1.0)
        if record and i % 50 == 0:   # rows: t, ||w||, psi_g, psi_p, yield, sat factor, r_g, eta
            trace.append(np.stack([np.full(N, t), s['wn'], psi, psip, s['gr'], s['lam'],
                                   r[:, 0], r[:, 1], eta[:, 0], eta[:, 1]], 0))
    eps = np.maximum(w0, 1.0 - ms)
    out = dict(mode=mode, coupled=coupled, m_star=ms, rho0=rho0, left=int((stat['wmax'] >= 1.0).sum()),
               above_bound=int((stat['wmax'] > eps + 1e-6).sum()), wmax_med=float(np.median(stat['wmax'])),
               wmax_max=float(stat['wmax'].max()), gauge_max=float(stat['gmax'].max()),
               eta_clear_min=float(stat['eta_clear'].min()), tube_clear_min=float(stat['tube_clear'].min()),
               sat_frac=float(np.mean(stat['sat_time'] / T)), real_ratio=float(np.mean(stat['path_real'] / stat['path_prop'])),
               yield_mean=float(np.mean(stat['yield_mean'] / T)), yield_err=float(np.mean(stat['yield_err'] / T)), w0_max=float(w0.max()), wmax_all=stat['wmax'].copy())
    return out, np.array(trace)

if __name__ == '__main__':
    # usage: python3 check_closed_loop.py [indices]; results are appended to ../records/closed_loop.jsonl
    import json
    runs = [('proposed', False, None), ('noyield', False, None), ('proposed', True, None),
            ('clip', True, None), ('proposed', False, 0.3)]
    sel = [int(i) for i in sys.argv[1:]] or range(len(runs))
    for i in sel:
        mode, coupled, vb = runs[i]
        o, _ = run(mode, coupled, vbar_prop=vb)
        o['vbar_prop'] = 3.0 if vb is None else vb
        o = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in o.items() if k != 'wmax_all'}
        print({k: (float(f'{v:.6g}') if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
        open('../records/closed_loop.jsonl', 'a').write(json.dumps(dict(run=i, **o)) + '\n')
