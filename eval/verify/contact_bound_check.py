"""Numerical checks C1 and C2 of the contact force bound (note_contact: scalar comparison lemma, contact force theorem).
Plan: records/contact_bound_check_plan.md. Usage: python contact_bound_check.py [n=3000] [only=c1|c2]"""
import jax; jax.config.update('jax_enable_x64', True)
import json, sys, time, numpy as np, jax.numpy as jnp
from pathlib import Path
try:
    from pairl.stageb import model as Mo, params as P
    from pairl import paths; RECORDS = paths.RECORDS
except ImportError:
    import model as Mo, params as P
    RECORDS = Path(__file__).resolve().parents[1] / 'records'

OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
UB = jnp.asarray(P.UB); TOL = 1e-3
Q = lambda x: [float(np.percentile(x, p)) for p in (50, 90, 99, 100)]

def scalar_check(n=2000, dt=1e-5, T=0.5, seed=0):
    """C1: m l'' = u - k l max(0, 1 + de l'), l(0) = 0, l'(0) = v0, |u| <= F; one contact interval per case."""
    rng = np.random.default_rng(seed); out = {}
    m = rng.uniform(0.5, 10, n); k = np.exp(rng.uniform(np.log(2e3), np.log(2e5), n)); de0 = rng.uniform(0, 5, n)
    F = rng.uniform(0, 100, n); v0 = rng.uniform(0.01, 1.0, n); K = int(round(T / dt)); hold = int(round(2e-3 / dt))
    noise = rng.uniform(-1, 1, (K // hold + 1, n))
    for fam in ('constant_undamped', 'constant', 'bangbang', 'random'):
        de = np.zeros(n) if fam == 'constant_undamped' else de0
        l = np.zeros(n); ld = v0.copy(); alive = np.ones(n, bool); lmax = np.zeros(n); fmax = np.zeros(n); ldmax = v0.copy()
        acc = lambda l, ld, u: (u - k * l * np.maximum(1 + de * ld, 0.0)) / m
        for i in range(K):
            u = F if fam.startswith('constant') else (F * np.sign(ld) if fam == 'bangbang' else F * noise[i // hold])
            k1l, k1v = ld, acc(l, ld, u); k2l, k2v = ld + 0.5 * dt * k1v, acc(l + 0.5 * dt * k1l, ld + 0.5 * dt * k1v, u)
            k3l, k3v = ld + 0.5 * dt * k2v, acc(l + 0.5 * dt * k2l, ld + 0.5 * dt * k2v, u); k4l, k4v = ld + dt * k3v, acc(l + dt * k3l, ld + dt * k3v, u)
            ln = l + dt / 6 * (k1l + 2 * k2l + 2 * k3l + k4l); ldn = ld + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            alive &= ln > 0; l = np.where(alive, ln, l); ld = np.where(alive, ldn, ld)
            lmax = np.maximum(lmax, l); ldmax = np.maximum(ldmax, np.where(alive, ld, 0.0))
            fmax = np.maximum(fmax, np.where(alive, k * l * np.maximum(1 + de * ld, 0.0), 0.0))
            if not alive.any(): break
        a = F / m; c = k / m; lbar = (a + np.sqrt(a ** 2 + c * v0 ** 2)) / c
        rl = lmax / lbar; rf = fmax / (k * lbar * (1 + de * ldmax))
        out[fam] = dict(n=n, r_l=Q(rl), r_f=Q(rf), still_in_contact=int(alive.sum()))
        print(f'C1 {fam}: r_l median {np.median(rl):.6f} max {rl.max():.6f} | r_f max {rf.max():.6f} | still in contact at the horizon {alive.sum()}', flush=True)
    worst = max(max(v['r_l'][-1], v['r_f'][-1]) for v in out.values())
    out['pass_bound'] = bool(worst <= 1 + TOL); out['pass_tight'] = bool(out['constant_undamped']['r_l'][0] >= 0.999); out['worst_ratio'] = worst
    return out

def sample_trajectories(n, seed=0):
    """Plants, surfaces, initial postures and torque-policy parameters of C2."""
    rng = np.random.default_rng(seed); pars, q0s, pol = [], [], []
    nA = n // 2; n_ood = n // 3; jac = jax.jit(Mo.jac); tip = jax.jit(Mo.tip)
    for i in range(n):
        par = P.make_par(rng, ood=(i % 3 == 2) if n_ood else False)
        ye = rng.uniform(-0.30, 0.30)
        par.update(xe=np.float64(0.0), ye=np.float64(ye), we=np.float64(10.0), ke=np.float64(np.exp(rng.uniform(np.log(2e3), np.log(2e5)))),
                   de=np.float64(rng.uniform(0.5, 5.0)), nu=np.float64(rng.uniform(0.2, 0.8)))
        for _ in range(100000):
            q0 = P.QC + 0.9 * P.QH * rng.uniform(-1, 1, 3); p0 = np.asarray(tip(jnp.asarray(q0), jnp.asarray(par['L'])))
            if 0.2 <= p0[0] <= 0.9 and ye + 0.03 <= p0[1] <= ye + 0.25: break
        else: raise RuntimeError('initial posture sampling failed')
        J = np.asarray(jac(jnp.asarray(q0), jnp.asarray(par['L'])))
        pt = np.array([p0[0] + rng.uniform(-0.1, 0.1), ye - rng.uniform(0.0, 0.05)])
        qt = q0 + np.linalg.pinv(J) @ (pt - p0)
        pars.append(par); q0s.append(q0)
        pol.append(dict(kind=np.float64(0.0 if i < nA else 1.0), xi=np.float64(i % 2), qt=qt, kp=rng.uniform(0.3, 1.0) * np.array([400.0, 200.0, 80.0]),
                        kd=rng.uniform(0.3, 1.5) * np.array([30.0, 12.0, 2.0]), sd=np.float64(rng.uniform(0.0, 10.0)), amp=np.float64(rng.uniform(0.05, 0.3)),
                        fb=np.float64(rng.uniform(0.0, 80.0)), dith=rng.uniform(-1, 1, (41, 3))))
    stack = lambda ds: {k: jnp.asarray(np.stack([d[k] for d in ds])) for k in ds[0]}
    return stack(pars), jnp.asarray(np.stack(q0s)), stack(pol)

def theorem_check(n=3000, dt=1e-4, T=1.2, seed=0):
    """C2: random torque inputs on the Stage B plant against a flat surface; the bound is evaluated per contact interval with its realized constants."""
    pars, q0, pol = sample_trajectories(n, seed); K = int(round(T / dt)); hold = int(round(0.03 / dt))
    def torque(q, dq, i, par, po):
        g = Mo.gravity(q, par, po['xi']); d = po['dith'][i // hold]; J = Mo.jac(q, par['L'])
        ta = po['kp'] * (po['qt'] - q) - po['kd'] * dq + g + po['sd'] * d
        tb = g - po['kd'] * dq + po['amp'] * d * UB + J.T @ jnp.stack([jnp.zeros(()), -po['fb']])
        return jnp.clip(jnp.where(po['kind'] < 0.5, ta, tb), -UB, UB)
    def diag(q, dq, tp, tn, par, xi):
        L = par['L']; J = Mo.jac(q, L); M = Mo.mass(q, par, xi); p = Mo.tip(q, L); v = J @ dq
        fe, _ = Mo.contact(p, v, par); Li = J @ jnp.linalg.solve(M, J.T)
        mu = Li[1, 1] - par['nu'] * jnp.abs(Li[0, 1])
        Jdqd = jax.jvp(lambda qq: Mo.jac(qq, L) @ dq, (q,), (dq,))[1]
        ain = lambda tau: -(J @ Mo.accel(q, dq, tau, xi, par, f_extra=-fe)[0] + Jdqd)[1]     # contact force cancelled
        return par['ye'] - p[1], -v[1], fe[1], mu, jnp.maximum(ain(tp), ain(tn))
    def close(e, par):
        """Bound of the interval held in e; returns the ratios and whether the interval is valid."""
        k = par['ke']; valid = e['mu'] > 0; c = jnp.where(valid, e['mu'], 1.0) * k; a = e['a']
        V0 = jnp.maximum(0.5 * jnp.maximum(e['lda'], 0.0) ** 2 + 0.5 * c * e['la'] ** 2 - a * e['la'], 0.0)
        lbar = (a + jnp.sqrt(a ** 2 + 2 * c * V0)) / c; fbar = k * lbar * (1 + par['de'] * e['ldmax'])
        return valid, e['lmax'] / lbar, e['fmax'] / fbar, lbar, fbar
    def fold(s, e, par, active):
        """Add the closed interval e to the trajectory statistics s if active."""
        valid, rl, rf, lbar, fbar = close(e, par); ok = active & valid; bad = active & ~valid; new = ok & (rf > s['rf'])
        return dict(n_valid=s['n_valid'] + ok, n_neg=s['n_neg'] + bad, rl=jnp.where(ok, jnp.maximum(s['rl'], rl), s['rl']), rf=jnp.where(ok, jnp.maximum(s['rf'], rf), s['rf']),
                    w_f=jnp.where(new, e['fmax'], s['w_f']), w_fbar=jnp.where(new, fbar, s['w_fbar']), w_l=jnp.where(new, e['lmax'], s['w_l']), w_lbar=jnp.where(new, lbar, s['w_lbar']),
                    neg_f=jnp.where(bad, jnp.maximum(s['neg_f'], e['fmax']), s['neg_f']), f_all=jnp.maximum(s['f_all'], jnp.where(active, e['fmax'], 0.0)))
    z = jnp.zeros(())
    e0 = dict(inc=jnp.zeros((), bool), la=z, lda=z, mu=z + 1e30, a=z, ldmax=z, lmax=z, fmax=z)
    s0 = dict(n_valid=z, n_neg=z, rl=z, rf=z, w_f=z, w_fbar=z, w_l=z, w_lbar=z, neg_f=z, f_all=z)
    def one_step(c, i, par, po):
        q, dq, tp, e, s = c; xi = po['xi']; tn = torque(q, dq, i, par, po)
        ls, ld, fn, mu, ain = diag(q, dq, tp, tn, par, xi); inc = ls > 0
        s = fold(s, e, par, e['inc'] & ~inc)                                                   # the interval ended before this node
        start = inc & ~e['inc']; ldp = jnp.maximum(ld, 0.0)
        e = dict(inc=inc, la=jnp.where(start, ls, e['la']), lda=jnp.where(start, ld, e['lda']),
                 mu=jnp.where(inc, jnp.minimum(jnp.where(start, 1e30, e['mu']), mu), e['mu']),
                 a=jnp.where(inc, jnp.maximum(jnp.where(start, 0.0, e['a']), jnp.where(ld > 0, ain, 0.0)), e['a']),
                 ldmax=jnp.where(inc, jnp.maximum(jnp.where(start, 0.0, e['ldmax']), ldp), e['ldmax']),
                 lmax=jnp.where(inc, jnp.maximum(jnp.where(start, 0.0, e['lmax']), ls), e['lmax']),
                 fmax=jnp.where(inc, jnp.maximum(jnp.where(start, 0.0, e['fmax']), fn), e['fmax']))
        f = lambda q, dq: (dq, Mo.accel(q, dq, tn, xi, par)[0])
        k1 = f(q, dq); k2 = f(q + 0.5 * dt * k1[0], dq + 0.5 * dt * k1[1]); k3 = f(q + 0.5 * dt * k2[0], dq + 0.5 * dt * k2[1]); k4 = f(q + dt * k3[0], dq + dt * k3[1])
        qn = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); dqn = dq + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return (qn, dqn, tn, e, s)
    def run_one(par, q0, po):
        c0 = (q0, jnp.zeros(3), torque(q0, jnp.zeros(3), 0, par, po), e0, s0)
        q, dq, tp, e, s = jax.lax.fori_loop(0, K, lambda i, c: one_step(c, i, par, po), c0)
        s = fold(s, e, par, e['inc'])                                                          # interval open at the horizon
        return dict(s, finite=jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(dq)), speed_end=jnp.linalg.norm(dq))
    t0 = time.time(); R = jax.jit(jax.vmap(run_one))(pars, q0, pol); R = {k: np.asarray(v) for k, v in R.items()}
    nv = R['n_valid'].sum(); nn = R['n_neg'].sum(); has = R['n_valid'] > 0
    out = dict(n=n, steps=K, dt=dt, seconds=time.time() - t0, finite=int(R['finite'].sum()), intervals_valid=int(nv), intervals_mu_nonpositive=int(nn),
               share_mu_nonpositive=float(nn / max(nv + nn, 1)), trajectories_with_valid_interval=int(has.sum()),
               r_l=Q(R['rl'][has]), r_f=Q(R['rf'][has]), max_fn_valid=float(R['w_f'][has].max()), max_fn_all=float(R['f_all'].max()),
               max_fn_mu_nonpositive=float(R['neg_f'].max()), kind_A_valid=int(R['n_valid'][: n // 2].sum()), kind_B_valid=int(R['n_valid'][n // 2:].sum()))
    out['pass_bound'] = bool(out['r_l'][-1] <= 1 + TOL and out['r_f'][-1] <= 1 + TOL and out['finite'] == n); out['pass_count'] = bool(nv >= 1000)
    print(f"C2: {n} trajectories, {int(nv)} valid contact intervals, {int(nn)} with mu <= 0 (share {out['share_mu_nonpositive']:.4f}), finite {out['finite']}", flush=True)
    print(f"C2: r_l median/p90/p99/max {out['r_l']} | r_f {out['r_f']} | largest f_n {out['max_fn_all']:.1f} N | {out['seconds']:.0f} s", flush=True)
    np.savez(RECORDS / 'contact_bound_check_c2.npz', f=R['w_f'][has], fbar=R['w_fbar'][has], l=R['w_l'][has], lbar=R['w_lbar'][has], kind=np.asarray(pol['kind'])[has])
    return out

if __name__ == '__main__':
    only = OPT.get('only', ''); f = RECORDS / 'contact_bound_check.json'; REC = json.load(open(f)) if f.exists() else {}
    if only in ('', 'c1'): REC['c1'] = scalar_check()
    if only in ('', 'c2'): REC['c2'] = theorem_check(int(OPT.get('n', 3000)))
    json.dump(REC, open(f, 'w'), indent=1)
    print('PASS' if all(REC[c][p] for c in REC for p in REC[c] if p.startswith('pass')) else 'FAIL', {c: {p: REC[c][p] for p in REC[c] if p.startswith('pass')} for c in REC})
