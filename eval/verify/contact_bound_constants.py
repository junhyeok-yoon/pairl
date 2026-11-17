"""Constants of the contact force bound for the Stage B structure (check C3 of records/contact_bound_check_plan.md): sampling on the gauge set,
restricted to the postures whose tip lies in the region of the block tops, and evaluation of the bound formulas of note_contact.
Usage: python contact_bound_constants.py [n=200000]"""
import jax; jax.config.update('jax_enable_x64', True)
import json, sys, numpy as np, jax.numpy as jnp
from pathlib import Path
try:
    from pairl.stageb import model as Mo, params as P
    from pairl import paths; RECORDS = paths.RECORDS
except ImportError:
    import model as Mo, params as P
    RECORDS = Path(__file__).resolve().parents[1] / 'records'
from pairl.tools.lib_contact_bound import fbar, fbar_reaction, speed_for

OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a); N = int(OPT.get('n', 200000))
K1 = 2.0                                    # k_1 of the constraint layer (Stage B table of the specification)
REGION = ((0.20, 0.90), (-0.30, 0.30))      # tip positions on a block top: x_e +- w_e with x_e in [0.35, 0.75]; y_e = p*_y - [0.05, 0.15]
DE_BAR, TD = 5.0, 0.004                     # largest dissipation coefficient of the class [s/m]; reaction delay: two control periods [s]
UB = jnp.asarray(P.UB); NV = jnp.array([0.0, 1.0])

def gauge_states(rng, n):
    """States with rho(w) <= 1 and rho(w_q) <= 1; half of them on rho(w) = 1."""
    wq = rng.uniform(-1, 1, (6 * n, 3)); wq = wq[np.sum(np.abs(wq) ** P.PG, -1) <= 1][:n]
    w = rng.normal(size=(n, 3)); w = w / (np.sum(np.abs(w) ** P.PG, -1, keepdims=True) ** (1.0 / P.PG))
    w = w * np.where(np.arange(n)[:, None] % 2 == 0, 1.0, rng.uniform(0, 1, (n, 1)) ** (1 / 3))
    return P.QC + P.QH * wq, K1 * P.QH * (w - wq)

def one(q, dq, par, xi):
    M = Mo.mass(q, par, xi); J = Mo.jac(q, par['L']); Li = J @ jnp.linalg.solve(M, J.T); cvec = jnp.linalg.solve(M, J.T @ NV)
    a0 = -(NV @ (J @ Mo.accel(q, dq, jnp.zeros(3), xi, par)[0] + jax.jvp(lambda qq: Mo.jac(qq, par['L']) @ dq, (q,), (dq,))[1]))   # inward acceleration at tau = 0
    mus = jnp.max(jnp.abs(J.T @ NV) / UB)                                                    # saturation scale of the direction J^T n
    return dict(lnn=Li[1, 1], lnt=jnp.abs(Li[0, 1]), a_box=a0 + jnp.sum(jnp.abs(cvec) * UB), a_relief=a0 - Li[1, 1] / mus,
                vc=jnp.linalg.norm(J @ dq), Jn=jnp.linalg.norm(J, 2), p=Mo.tip(q, par['L']))

if __name__ == '__main__':
    rng = np.random.default_rng(1); f = jax.jit(jax.vmap(one)); REC = dict(n=N, k1=K1, region=REGION, de_bar=DE_BAR, td=TD, cases={})
    for ood in (False, True):
        for xi in (0, 1):
            q, dq = gauge_states(rng, N); pars = [P.make_par(rng, ood=ood) for _ in range(N)]
            Pb = {k: jnp.asarray(np.stack([p[k] for p in pars])) for k in pars[0]}
            R = {k: np.asarray(v) for k, v in f(jnp.asarray(q), jnp.asarray(dq), Pb, jnp.full(N, float(xi))).items()}
            reg = (R['p'][:, 0] >= REGION[0][0]) & (R['p'][:, 0] <= REGION[0][1]) & (R['p'][:, 1] >= REGION[1][0]) & (R['p'][:, 1] <= REGION[1][1])
            R = {k: v[reg] for k, v in R.items()}; slow = R['vc'] <= 0.5
            c = dict(n_region=int(reg.sum()), lam_nn_min=float(R['lnn'].min()), lam_nn_median=float(np.median(R['lnn'])), nu_crit_min=float((R['lnn'] / np.maximum(R['lnt'], 1e-12)).min()),
                     vc_max=float(R['vc'].max()), a_box_max=float(R['a_box'].max()), a_relief_max=float(R['a_relief'].max()), n_slow=int(slow.sum()),
                     a_relief_max_slow=float(R['a_relief'][slow].max()), a_relief_median_slow=float(np.median(R['a_relief'][slow])), J_max=float(R['Jn'].max()))
            for nu in (0.5, 0.8):
                mu = R['lnn'] - nu * R['lnt']
                c[f'mu_{nu}'] = dict(min=float(mu.min()), median=float(np.median(mu)), share_nonpositive=float(np.mean(mu <= 0)), share_below_005=float(np.mean(mu < 0.05)))
            REC['cases'][f"{'ood' if ood else 'in'}_xi{xi}"] = c
            print(f"{'ood' if ood else 'in '} xi {xi}: region {c['n_region']} | lam_nn min {c['lam_nn_min']:.4f} | nu_crit min {c['nu_crit_min']:.3f} | mu(0.5) min {c['mu_0.5']['min']:.4f} "
                  f"| mu(0.8) min {c['mu_0.8']['min']:.4f} nonpositive {c['mu_0.8']['share_nonpositive']:.4f} | vc max {c['vc_max']:.2f} | a_box max {c['a_box_max']:.0f} "
                  f"| a_relief max {c['a_relief_max']:.1f}, slow {c['a_relief_max_slow']:.1f} (median {c['a_relief_median_slow']:.1f}) | J max {c['J_max']:.3f}", flush=True)
    # class constants in distribution: worst over the two load cases
    ins = [REC['cases']['in_xi0'], REC['cases']['in_xi1']]
    K = dict(mu_low=min(c['mu_0.5']['min'] for c in ins), mu_median=min(c['mu_0.5']['median'] for c in ins), vc=max(c['vc_max'] for c in ins), a_box=max(c['a_box_max'] for c in ins),
             a_r=-max(c['a_relief_max_slow'] for c in ins), a_r_median=-max(c['a_relief_median_slow'] for c in ins), J=max(c['J_max'] for c in ins))
    REC['class_in'] = K
    REC['present'] = {f'k{int(k)}': float(fbar(K['vc'], K['a_box'], K['mu_low'], k, DE_BAR)) for k in (2e3, 2e4, 2e5)}
    REC['present_parts'] = dict(F_a=K['a_box'] / K['mu_low'], impact_k20000=float(K['vc'] * np.sqrt(2e4 / K['mu_low'])))
    REC['speed_for'] = {}
    for mu_name in ('mu_low', 'mu_median'):
        mu = K[mu_name]; ar = K['a_r'] if mu_name == 'mu_low' else K['a_r_median']
        for k in (2e3, 2e4, 2e5):
            for flim in (60.0, 200.0):
                REC['speed_for'][f'{mu_name}_k{int(k)}_f{int(flim)}'] = dict(
                    no_drive=speed_for(lambda v: fbar(v, 0.0, mu, k, DE_BAR), flim), box_drive=speed_for(lambda v: fbar(v, K['a_box'], mu, k, DE_BAR), flim),
                    reaction=speed_for(lambda v: fbar_reaction(v, max(ar, 0.0), mu, k, TD, DE_BAR), flim) if ar > 0 else None)
    for key, v in REC['speed_for'].items(): print(key, {a: (None if b is None else round(b, 4)) for a, b in v.items()})
    print('class constants in distribution', {k: round(v, 4) for k, v in K.items()}); print('present structure', REC['present'], REC['present_parts'])
    json.dump(REC, open(RECORDS / 'contact_bound_constants.json', 'w'), indent=1)
