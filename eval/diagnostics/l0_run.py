"""Confirmatory L0 runs (records/preregistration_l0.md, hash in records/preregistration_l0.hash)."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, json, sys, time
from pairl import paths
from pairl.stageb import env, params as P, controller as C, model as Mo
from pairl.learn import l0
T0 = time.time(); R = {}
def log(*a): print('[%5.0f s]' % (time.time() - T0), *a, flush=True)
def base_cfg(kb):
    c = C.default_cfg(); c.update(law=2.0, lam_s=5.0, v_r=0.3, klo=np.array([50.0, 25.0, 10.0]), khi=np.array([400.0, 200.0, 80.0]), kb=kb, gate_sign=1.0, pm=0.5); return c   # L0 as registered: sign-exception gate
def eps_of(seeds, **kw): return [l0.prep(*env.sample_episode(np.random.default_rng(s), **kw, v=1)) for s in seeds]
def run(eps, cfg, brain, opts, nsteps=None, nsub=4, key=0):
    n = nsteps or int(max(int(np.sum(e[1]['nph'])) for e in eps))
    st, cs, m, _ = l0.batch(eps, jax.random.PRNGKey(key), n, cfg, brain, opts, nsub)
    return {k: np.asarray(v) for k, v in st.items()}, {k: np.asarray(v) for k, v in cs.items()}, {k: np.asarray(v) for k, v in m.items()}
NOM = dict(v_d=0.15, g_f=0.3)
# L0-3 and L0-4: choose k_b
KB = [0.25, 0.5, 1.0, 2.0]; R['L0-3'] = {}; R['L0-4'] = {}
rng = np.random.default_rng(1399)
box_eps = []
for s in range(64):
    par, task = l0.prep(*env.sample_episode(np.random.default_rng(1400 + s), hard=False, v=1)); task['corner'] = rng.choice([-1.0, 1.0], 3); box_eps.append((par, task))
hard_eps = eps_of(range(1600, 1664), hard=True)
for kb in KB:
    st, cs, m = run(box_eps, base_cfg(kb), 'box', dict(extra=0.3), nsteps=2000)
    R['L0-3'][str(kb)] = dict(violations=int(m['viol'].sum()), max_rho_wq=float(m['rho_wq'].max()), max_rho_w=float(m['rho_w'].max()), finite=bool(np.all(np.isfinite(st['q']))))
    c = base_cfg(kb); st, cs, m = run(hard_eps, c, 'hold', dict(K=c['khi']), nsteps=1500)
    err = m['hold_err'] / m['hold_n']
    R['L0-4'][str(kb)] = dict(tip_err_median=float(np.median(err)), tip_err_p90=float(np.percentile(err, 90)), tip_err_max=float(err.max()), rho_wq_median=float(np.median(m['rho_wq'])))
    log('kb', kb, R['L0-3'][str(kb)], R['L0-4'][str(kb)])
ok = [kb for kb in KB if R['L0-3'][str(kb)]['violations'] == 0]
good = [kb for kb in ok if R['L0-4'][str(kb)]['tip_err_median'] <= 0.01]
kb = max(good) if good else (min(ok) if ok else None)
R['kb_choice'] = dict(kb=kb, flagged=not bool(good), passing_violation=ok, passing_error=good)
log('chosen kb', R['kb_choice'])
if kb is None: json.dump(R, open(str(paths.RECORDS / 'l0_results.json'), 'w'), indent=1); sys.exit('no k_b without violations')
cfg = base_cfg(kb)
# L0-1 energy accounting (random brain)
g1 = [(eps_of(range(1000, 1064)), 4, 'in'), (eps_of(range(1064, 1096), ood_body=True), 4, 'ood_body'), (eps_of(range(1096, 1128), ood_stiff=True), 8, 'stiff')]
R['L0-1'] = {}
for eps, nsub, name in g1:
    st, cs, m = run(eps, cfg, 'random', NOM, nsub=nsub, key=11)
    R['L0-1'][name] = dict(Tq_err_max=float(m['Tq_err'].max()), barrier_cum_max=float(m['barrier_cum_max'].max()), violations=int(m['viol'].sum()),
                           finite=bool(np.all(np.isfinite(st['q'])) and np.all(np.isfinite(cs['T']))), gate_steps_total=float(m['gate_steps'].sum()), cons_steps_total=float(m['cons_steps'].sum()),
                           lam_min=float(m['lam_min'].min()), grasps=float(np.mean(st['n_grasp'])))
    log('L0-1', name, R['L0-1'][name])
# L0-2 small budgets
R['L0-2'] = {}
for i, E0 in enumerate((0.5, 1.0, 2.0, 5.0)):
    eps = [(p, dict(t, E0=np.float64(E0))) for p, t in eps_of(range(1200 + 32 * i, 1232 + 32 * i))]
    st, cs, m = run(eps, cfg, 'nominal', NOM, key=12)
    # min T over the run: tank_bound_min = min(T - Tmin + max Ub); also record min T directly via Tq path is not logged -> use bound with Ub_run
    R['L0-2'][str(E0)] = dict(bound_deficit_min=float(m['tank_bound_min'].min()), runs_below_bound=int(np.sum(m['tank_bound_min'] < 0)), sat_below_Tmin_runs=int(m['sat_below'].sum()),
                              gate_steps_median=float(np.median(m['gate_steps'])), Ub_run_max=float(m['Ub_run'].max()), minT_lower=float((m['tank_bound_min'] + cfg['Tmin'] - m['Ub_run']).min()))
    log('L0-2', E0, R['L0-2'][str(E0)])
# L0-5 guard
eps = eps_of(range(1800, 1864))
st, cs, m = run(eps, cfg, 'nominal', dict(NOM, adversarial=True), key=15)
tc = m['press_tc'] / np.maximum(m['press_n'], 1); mis = m['press_mis'] / np.maximum(m['press_n'], 1); bound = cfg['tau_max'] + mis + 1.0
R['L0-5'] = dict(n=64, all_within=bool(np.all(tc <= bound)), contact_torque_max=float(tc.max()), contact_torque_median=float(np.median(tc)), excess_max=float((tc - bound).max()),
                 guard_steps_median=float(np.median(cs['guard_on'])), peak_impact_max=float(m['peak_impact'].max()))
log('L0-5', R['L0-5'])
# L0-6 calibration
eps = eps_of(range(2000, 2064))
st, cs, m = run(eps, cfg, 'calib', dict(hold=1.5, K=cfg['Kn']), nsteps=2500, key=16)
okg = m['m_grasp'] > 0; r6a = m['m_1s'][okg] / m['m_grasp'][okg]
up_fast = float(m['upd_fast'].sum())
R['L0-6a'] = dict(n_grasped=int(okg.sum()), ratio_median=float(np.median(r6a)), ratio_p90=float(np.percentile(r6a, 90)), ratio_max=float(r6a.max()), m_grasp_median=float(np.median(m['m_grasp'][okg])))
eps = eps_of(range(2200, 2264))
st, cs, m = run(eps, cfg, 'nominal', dict(NOM, reset_z_at_contact=True), key=17)
edges = np.stack([np.cumsum(e[1]['nph']) for e in eps]) * P.TS
okc = (m['t_contact'] >= 0) & (m['t_contact'] + 1.0 <= edges[:, 2]) & (m['m_c0'] > 0)
r6b = m['m_c1'][okc] / m['m_c0'][okc]; up_fast += float(m['upd_fast'].sum())
R['L0-6b'] = dict(n_valid=int(okc.sum()), ratio_median=float(np.median(r6b)) if okc.any() else None, ratio_p90=float(np.percentile(r6b, 90)) if okc.any() else None,
                  ratio_max=float(r6b.max()) if okc.any() else None, m_c0_median=float(np.median(m['m_c0'][okc])) if okc.any() else None, peak_impact_median=float(np.median(m['peak_impact'])))
log('L0-6a', R['L0-6a']); log('L0-6b', R['L0-6b'])
# L0-7 generalization (F-class vs F-obj)
eps = eps_of(range(2400, 2464)); R['L0-7'] = {}
for feats, name in ((0.0, 'F-class'), (1.0, 'F-obj')):
    st, cs, m = run(eps, dict(cfg, feats=feats), 'calib', dict(hold=1.5, K=cfg['Kn']), nsteps=2500, key=18)
    up_fast += float(m['upd_fast'].sum())
    prior = []
    for i, (par, task) in enumerate(eps):
        pj = {k: jnp.asarray(v) for k, v in par.items()}; pe = dict(pj, rx=jnp.asarray(st['rx'][i]), ry=jnp.asarray(st['ry'][i])); qs = jnp.asarray(task['qs'])
        ustar = np.asarray(Mo.gravity(qs, pe, 1.0)) - par['d']; ub = np.asarray(C.prior_body(qs, l0.LNOM, l0.MNOM))
        prior.append(np.linalg.norm(ub - ustar))
    prior = np.array(prior); okb = (m['m_B_first'] > 0) & (st['xi'] > 0.5)
    r7 = m['m_B_first'][okb] / prior[okb]
    R['L0-7'][name] = dict(n=int(okb.sum()), ratio_median=float(np.median(r7)), ratio_p90=float(np.percentile(r7, 90)), prior_mis_median=float(np.median(prior[okb])),
                           at_A_after_hold_median=float(np.median(m['m_A_hold'][okb])))
    log('L0-7', name, R['L0-7'][name])
R['L0-6c'] = dict(updates_above_0_35=up_fast)
# L0-8 update law x stiffness
eps = eps_of(range(2600, 2664)); R['L0-8'] = {}
for law, lam, lname in ((0.0, 2.0, 'spec'), (2.0, 5.0, 'N')):
    for Kname in ('klo', 'Kn', 'khi'):
        c = dict(cfg, law=law, lam_s=lam)
        st, cs, m = run(eps, c, 'calib', dict(hold=3.0, K=c[Kname]), nsteps=2500, key=19)
        okg = (m['m_grasp'] > 0) & (m['m_29'] > 0); r1 = m['m_1s'][okg] / m['m_grasp'][okg]; r29 = m['m_29'][okg] / m['m_grasp'][okg]
        R['L0-8'][f'{lname}_{Kname}'] = dict(n=int(okg.sum()), r1_median=float(np.median(r1)), r1_p90=float(np.percentile(r1, 90)), r29_median=float(np.median(r29)),
                                             growth_max=float(np.max(r29 - r1)), vmax_hold_max=float(m['vmax_hold'][okg].max()))
        log('L0-8', lname, Kname, R['L0-8'][f'{lname}_{Kname}'])
# verdicts
V = {}
V['L0-1'] = all(g['Tq_err_max'] <= 0.1 and g['barrier_cum_max'] <= 0.05 and g['violations'] == 0 and g['finite'] for g in R['L0-1'].values())
V['L0-2a'] = all(g['minT_lower'] >= 0.0 for g in R['L0-2'].values())
V['L0-3a'] = R['L0-3'][str(kb)]['violations'] == 0
V['L0-5'] = R['L0-5']['all_within']
V['L0-6a'] = R['L0-6a']['ratio_median'] <= 0.1 and R['L0-6a']['ratio_p90'] <= 0.3
V['L0-6b'] = R['L0-6b']['ratio_median'] is not None and R['L0-6b']['ratio_median'] <= 0.1 and R['L0-6b']['ratio_p90'] <= 0.3
V['L0-6c'] = R['L0-6c']['updates_above_0_35'] == 0
e8 = R['L0-8']
V['L0-8_N'] = e8['N_Kn']['r1_median'] <= 0.1 and e8['N_khi']['r1_median'] <= 0.1 and e8['N_klo']['r1_median'] <= 0.3 and all(e8[f'N_{k}']['growth_max'] <= 0.05 for k in ('klo', 'Kn', 'khi'))
V['prediction_spec_fails_at_Kn'] = e8['spec_Kn']['r1_median'] > 0.1
V['prediction_L0-7'] = R['L0-7']['F-obj']['ratio_median'] <= 0.3 and R['L0-7']['F-obj']['ratio_median'] < R['L0-7']['F-class']['ratio_median']
R['verdicts'] = V; R['all_gating_pass'] = all(V[k] for k in ('L0-1', 'L0-2a', 'L0-3a', 'L0-5', 'L0-6a', 'L0-6b', 'L0-6c', 'L0-8_N'))
json.dump(R, open(str(paths.RECORDS / 'l0_results.json'), 'w'), indent=1)
log('verdicts', V, 'all gating pass:', R['all_gating_pass'])
