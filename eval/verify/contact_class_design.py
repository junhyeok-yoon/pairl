"""Constants for the enlarged contact class (plan records/contact_class_plan.md, D1-D4).
Usage: python contact_class_design.py [n=200000]"""
import jax; jax.config.update('jax_enable_x64', True)
import json, sys, numpy as np, jax.numpy as jnp
from pathlib import Path
try:
    from pairl.stageb import model as Mo, params as P; from pairl.tools import design_study as DS
    from pairl import paths; RECORDS = paths.RECORDS
except ImportError:
    import model as Mo, params as P, design_study as DS
    RECORDS = Path(__file__).resolve().parents[1] / 'records'
from pairl.tools.lib_contact_bound import fbar_reaction, speed_for
from contact_bound_constants import gauge_states, REGION, DE_BAR, UB, NV

N = int(dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a).get('n', 200000)); NU = 0.8

def one(q, dq, par, xi):
    M = Mo.mass(q, par, xi); J = Mo.jac(q, par['L']); Y = J @ jnp.linalg.solve(M, J.T)
    a0 = -(NV @ (J @ Mo.accel(q, dq, jnp.zeros(3), xi, par)[0] + jax.jvp(lambda qq: Mo.jac(qq, par['L']) @ dq, (q,), (dq,))[1]))
    mus = jnp.max(jnp.abs(J.T @ NV) / UB)
    return dict(theta=Y[1, 1] - NU * jnp.abs(Y[0, 1]), a_relief=a0 - Y[1, 1] / mus, f_stat=1.0 / mus, kin=0.5 * dq @ M @ dq,
                lam=jnp.linalg.eigvalsh(M)[-1], vc=jnp.linalg.norm(J @ dq), p=Mo.tip(q, par['L']))

if __name__ == '__main__':
    REC = dict(n=N, nu=NU, b3={}, cases={})
    # D1
    for F in (60.0, 100.0, 150.0, 200.0, 300.0):
        REC['b3'][str(int(F))] = {('ood' if ood else 'in'): float(-DS.b3_samples(P.QC, P.QH, P.UB, 2.0, 40000, 1, ood, fbar=F, nu=NU)[1].max()) for ood in (False, True)}
        print('D1 fbar', F, REC['b3'][str(int(F))], flush=True)
    # D2, D3
    rng = np.random.default_rng(1); f = jax.jit(jax.vmap(one)); pool = []
    for xi in (0, 1):
        q, dq = gauge_states(rng, N); pars = [P.make_par(rng, ood=False) for _ in range(N)]
        Pb = {k: jnp.asarray(np.stack([p[k] for p in pars])) for k in pars[0]}
        R = {k: np.asarray(v) for k, v in f(jnp.asarray(q), jnp.asarray(dq), Pb, jnp.full(N, float(xi))).items()}
        reg = (R['p'][:, 0] >= REGION[0][0]) & (R['p'][:, 0] <= REGION[0][1]) & (R['p'][:, 1] >= REGION[1][0]) & (R['p'][:, 1] <= REGION[1][1])
        R = {k: v[reg] for k, v in R.items()}; slow = R['vc'] <= 0.5; q05 = float(np.quantile(R['theta'], 0.05)); pool.append(R)
        REC['cases'][f'in_xi{xi}'] = c = dict(n_region=int(reg.sum()), theta_min=float(R['theta'].min()), theta_q01=float(np.quantile(R['theta'], 0.01)), theta_q05=q05,
            theta_median=float(np.median(R['theta'])), share_nonpositive=float(np.mean(R['theta'] <= 0)), share_below_005=float(np.mean(R['theta'] < 0.05)),
            kin_max=float(R['kin'].max()), lam_max=float(R['lam'].max()), kin_max_slow=float(R['kin'][slow].max()),
            a_relief_max_slow_q05=float(R['a_relief'][slow & (R['theta'] >= q05)].max()))
        print(f'D3 xi {xi}:', {k: round(v, 4) for k, v in c.items()}, flush=True)
    fs = np.concatenate([R['f_stat'] for R in pool]); REC['f_stat'] = dict(min=float(fs.min()), q05=float(np.quantile(fs, 0.05)), median=float(np.median(fs)))
    print('D2 f_stat', REC['f_stat'])
    # D4
    K = json.load(open(RECORDS / 'contact_bound_constants.json'))['class_in']
    sets = {'class_nu05': (K['mu_low'], K['a_r']),
            'q05_nu08': (min(REC['cases'][c]['theta_q05'] for c in REC['cases']), -max(REC['cases'][c]['a_relief_max_slow_q05'] for c in REC['cases']))}
    REC['sets'] = {k: dict(theta=v[0], a_r=v[1]) for k, v in sets.items()}; REC['speed'] = {}; REC['energy'] = {}
    for name, (th, ar) in sets.items():
        for k in (2e4, 2e5, 1e6):
            for fl in (60.0, 100.0, 200.0):
                for td in (0.002, 0.004):
                    REC['speed'][f'{name}_k{int(k)}_f{int(fl)}_td{int(td * 1000)}'] = speed_for(lambda v: fbar_reaction(v, max(ar, 0.0), th, k, td, DE_BAR), fl)
    for k in (2e4, 2e5, 1e6):
        for fl in (60.0, 100.0, 200.0): REC['energy'][f'k{int(k)}_f{int(fl)}'] = fl ** 2 / (2 * k)
    for key, v in REC['speed'].items(): print(key, None if v is None else round(v, 4))
    print('sets', REC['sets']); print('energy', REC['energy'])
    json.dump(REC, open(RECORDS / 'contact_class_design.json', 'w'), indent=1)
