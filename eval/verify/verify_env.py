"""V5-V18 of records/stageb_env_criteria.md (cost of spec revision 15)."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, json, sys, time, subprocess, mujoco
from pairl import paths
from pairl.stageb import model as Mo, env, params as P, refs, harness as H
from pairl.stageb.kin_np import best_margin, gauge
R, PASS = {}, {}
rng = np.random.default_rng(7)
J = lambda d: {k: jnp.asarray(v) for k, v in d.items()}
free = lambda st, par, task: jnp.zeros(3)

# V5 integration order and energy drift (free motion, no friction, no disturbance)
def free_case(seed):
    r = np.random.default_rng(seed); par = P.make_par(r, ood=bool(seed % 2)); par['Fv'] = np.zeros(3); par['d'] = np.zeros(3)
    task = dict(c0=np.zeros(2), psi0=np.float64(0.0), q0=r.uniform(-np.pi, np.pi, 3), nph=np.array([500, 500, 500, 500]), po=np.zeros(2), ps=np.zeros(2), fstar=np.float64(1.0), bias=np.zeros(3))
    return par, task, r.uniform(-2, 2, 3)
cases = [free_case(s) for s in range(64)]
par_b, task_b = env.stack([(c[0], c[1]) for c in cases]); dq0 = jnp.asarray(np.stack([c[2] for c in cases]))
def run_free(par, task, dq0, xi, nsub):
    st = dict(env.init_state(par, task), dq=dq0, xi=xi)
    def step(st, _):
        return env.control_step(st, jnp.zeros(3), par, nsub), env.e_rob(st, par)
    st, e = jax.lax.scan(step, st, None, 500)
    return st['q'], st['dq'], e
xi_b = jnp.asarray((np.arange(64) % 4 >= 2).astype(float))
out = {n: jax.jit(jax.vmap(lambda p, t, d, x: run_free(p, t, d, x, n)))(par_b, task_b, dq0, xi_b) for n in (1, 2, 4, 16)}
err = {n: np.max(np.abs(np.concatenate([np.asarray(out[n][0] - out[16][0]), np.asarray(out[n][1] - out[16][1])], 1)), 1) for n in (1, 2, 4)}
order = np.stack([np.log2(err[1] / err[2]), np.log2(err[2] / err[4])])
drift = {n: float(np.max(np.abs(np.asarray(out[n][2]) - np.asarray(out[n][2])[:, :1]) / (1 + np.abs(np.asarray(out[n][2])[:, :1])))) for n in (1, 2, 4, 16)}
R['V5'] = dict(drift=drift, order_median=[float(np.median(order[0])), float(np.median(order[1]))], order_range=[float(order.min()), float(order.max())])
PASS['V5'] = bool(drift[4] <= 1e-7 and 3.5 <= np.median(order[0]) <= 4.5 and 3.5 <= np.median(order[1]) <= 4.5)

# V6 power balance without contact
def tau_rand(st, par, task):
    t = st['k'] * P.TS
    return jnp.asarray([30.0, 15.0, 5.0]) * jnp.sin(task['om'] * t + task['phi'])
cases6 = []
for s in range(64):
    r = np.random.default_rng(100 + s); par = P.make_par(r, ood=bool(s % 2))
    cases6.append((par, dict(c0=np.zeros(2), psi0=np.float64(0.0), q0=r.uniform(-np.pi, np.pi, 3), nph=np.array([500, 500, 500, 500]), po=np.zeros(2), ps=np.zeros(2), fstar=np.float64(1.0), bias=np.zeros(3),
                             om=r.uniform(2, 25, 3), phi=r.uniform(0, 6.28, 3))))
pb, tb = env.stack(cases6); keys = jax.random.split(jax.random.PRNGKey(0), 64)
st6, lg6 = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, 1000, 4, tau_rand, events=False)))(pb, tb, keys)
res6 = np.abs(np.asarray(lg6['e'] - lg6['e'][:, :1] - (lg6['w_act'] - lg6['w_fric']))); sc6 = np.maximum(1.0, np.max(np.abs(np.asarray(lg6['w_act'])), 1, keepdims=True))
R['V6'] = dict(residual_rel=float(np.max(res6 / sc6)), w_act_max=float(np.max(np.abs(np.asarray(lg6['w_act'])))), w_fric_max=float(np.max(np.asarray(lg6['w_fric']))))
PASS['V6'] = bool(R['V6']['residual_rel'] <= 1e-6)

# V7 contact
cp = J(dict(P.make_par(np.random.default_rng(0)), xe=0.5, ye=0.1, we=0.15, ke=1e4, de=2.0, nu=0.5))
tests = [((0.5, 0.11), (0, 0), (0, 0), 0.0), ((0.5, 0.099), (0, 0), (0, 10.0), 0.005), ((0.5, 0.099), (0, -0.1), (0, 12.0), 0.005), ((0.5, 0.099), (0, 1.0), (0, 0), 0.005),
         ((0.5, 0.099), (0.2, 0), (-5.0 * np.tanh(16.0), 10.0), 0.005), ((0.66, 0.05), (0, 0), (0, 0), 0.0), ((0.645, 0.0), (0, 0), (50.0, 0), 0.125), ((0.645, 0.0), (-0.1, 0), (60.0, 0), 0.125),
         ((0.355, 0.0), (0, 0), (-50.0, 0), 0.125)]
e7a = 0.0
for p, v, f, s in tests:
    ff, ss = Mo.contact(jnp.asarray(p, jnp.float64), jnp.asarray(v, jnp.float64), cp); e7a = max(e7a, float(np.max(np.abs(np.asarray(ff) - np.asarray(f)))), abs(float(ss) - s))
r7 = np.random.default_rng(1); n7 = 100000
pp = np.stack([r7.uniform(0.3, 0.7, n7), r7.uniform(-0.1, 0.12, n7)], 1); vv = r7.uniform(-2, 2, (n7, 2))
ff, ss = jax.vmap(lambda p, v: Mo.contact(p, v, cp))(jnp.asarray(pp), jnp.asarray(vv)); ff = np.asarray(ff)
dtop = 0.1 - pp[:, 1]; dside = 0.15 - np.abs(pp[:, 0] - 0.5); ins = (dtop > 0) & (dside > 0); top = dtop <= dside
ddot = np.where(top, -vv[:, 1], -np.sign(pp[:, 0] - 0.5) * vv[:, 0]); sdot = np.where(ins, 1e4 * np.minimum(dtop, dside) * ddot, 0.0)
viol7b = float(np.max(sdot + np.sum(vv * ff, 1)))
def tau_contact(st, par, task):
    q, dq = st['q'], st['dq']; t = st['k'] * P.TS; Jm = Mo.jac(q, par['L']); p = Mo.tip(q, par['L']); v = Jm @ dq
    F = 8.0 + 12.0 * jnp.sin(2 * jnp.pi * 0.7 * t)
    return env.u_star(st, par) + Jm.T @ jnp.stack([300.0 * (par['xe'] + 0.05 * jnp.sin(jnp.pi * t) - p[0]) - 40.0 * v[0], -F - 20.0 * v[1]]) - jnp.asarray([3.0, 2.0, 1.0]) * dq
eps7 = []
for s in range(16):
    par, task = H.add_postures(*env.sample_episode(np.random.default_rng(200 + s), hard=False)); task['q0'] = task['qs']; eps7.append((par, task))
tab7, last = {}, {}
for ke in (2e3, 2e4, 2e5):
    for nsub in (2, 4, 8, 16):
        pb, tb = env.stack([(dict(p, ke=np.float64(ke)), t) for p, t in eps7])
        st, lg = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, 1500, nsub, tau_contact, events=False)))(pb, tb, keys[:16])
        res = np.abs(np.asarray(lg['e'] - lg['e'][:, :1] - (lg['w_act'] - lg['w_fric'] + lg['w_env']))); sc = np.maximum(1.0, np.max(np.abs(np.asarray(lg['w_act'])), 1, keepdims=True))
        tab7[f'{int(ke)}_{nsub}'] = dict(residual_rel=float(np.max(res / sc)), passivity_excess=float(np.max(np.asarray(lg['w_env'] + lg['senv']))), fn_max=float(np.max(np.asarray(lg['fn']))),
                                        contact_frac=float(np.mean(np.asarray(lg['fn']) > 0)))
        last[(ke, nsub)] = (np.asarray(st['q']), np.asarray(lg['fn']))
    for nsub in (2, 4, 8):
        tab7[f'{int(ke)}_{nsub}']['dq_end_vs_16'] = float(np.max(np.abs(last[(ke, nsub)][0] - last[(ke, 16)][0])))
        tab7[f'{int(ke)}_{nsub}']['fn_rms_vs_16'] = float(np.sqrt(np.mean((last[(ke, nsub)][1] - last[(ke, 16)][1]) ** 2)))
R['V7'] = dict(force_values_err=e7a, pointwise_passivity_max=viol7b, runs=tab7)
PASS['V7'] = bool(e7a <= 1e-9 and viol7b <= 1e-9 and all(tab7[f'{k}_4']['residual_rel'] <= 1e-4 for k in (2000, 20000)) and all(v['passivity_excess'] <= 1e-6 for v in tab7.values()))

# V8 gripper events
par, task = H.add_postures(*env.sample_episode(np.random.default_rng(300), hard=False)); pj, tj = J(par), J({k: v for k, v in task.items()})
qo = np.asarray(task['qo']); tipo = np.asarray(Mo.tip(jnp.asarray(qo), pj['L']))
def ev(xi, psi, dist, speed, psi_obj=0.3):
    """Gripper event from the posture qo with the object's grasp point at tip + (dist, 0)."""
    g = tipo + np.array([dist, 0.0]); c = g - par['Lobj'] * np.array([np.cos(psi_obj), np.sin(psi_obj)])
    st = dict(env.init_state(pj, tj), q=jnp.asarray(qo), dq=jnp.asarray([speed, 0.0, 0.0]), xi=jnp.asarray(float(xi)), obj_c=jnp.asarray(c), obj_psi=jnp.asarray(psi_obj))
    s2 = env.gripper_event(st, jnp.asarray(float(psi)), pj, tj)
    return int(s2['xi']), s2, st
table = [((0, 1, 0.0, 0.0), 1), ((0, 1, 0.05, 0.0), 0), ((0, 1, 0.0, 0.5), 0), ((0, 0, 0.0, 0.0), 0), ((1, 1, 0.5, 0.5), 1), ((1, 0, 0.5, 0.5), 0),
         ((1, 0, 0.0, 0.0), 0), ((0, 1, 0.0199, 0.0), 1), ((0, 1, 0.0201, 0.0), 0), ((0, 1, 0.0, 0.099), 1), ((0, 1, 0.0, 0.101), 0)]
tt_ok = all(ev(*a)[0] == b for a, b in table)
# grasp, release, grasp again at the same place
_, s_a, s0 = ev(0, 1, 0.0, 0.0); s_b = env.gripper_event(s_a, jnp.asarray(0.0), pj, tj); s_c = env.gripper_event(s_b, jnp.asarray(1.0), pj, tj)
regrasp_ok = (int(s_a['xi']), int(s_b['xi']), int(s_c['xi']), int(s_c['n_grasp']), int(s_c['n_release'])) == (1, 0, 1, 2, 1)
grasp_point_after_release = float(np.max(np.abs(np.asarray(env.grasp_point(s_b, pj)) - tipo)))
# exact grasp at rest: E_rob jumps by the potential energy of the resting object, center of mass continuous
jump_err = abs(float(s_a['e_jump']) - par['mo'] * Mo.G0 * float(s0['obj_c'][1]))
com_err = float(s_a['com_err_max'])
_, s_r, _ = ev(1, 0, 0.0, 0.07)
rel_unchanged = float(np.max(np.abs(np.asarray(s_r['dq']) - np.array([0.07, 0, 0]))))
# random events: E_rob + S_obj never increases; a release removes exactly the object's kinetic energy; the released center of mass is the attached one
r8b = np.random.default_rng(81); gain_max, rel_err, crel_err = -1e30, 0.0, 0.0
for _ in range(2000):
    prr = P.make_par(r8b, ood=bool(r8b.integers(2))); pjr = J(prr); q = P.QC + P.QH * r8b.uniform(-0.9, 0.9, 3)
    tipq = np.asarray(Mo.tip(jnp.asarray(q), pjr['L'])); ps_ = r8b.uniform(-np.pi, np.pi); g = tipq + r8b.uniform(-0.014, 0.014, 2)
    c = g - prr['Lobj'] * np.array([np.cos(ps_), np.sin(ps_)])
    st = dict(env.init_state(pjr, dict(tj, c0=jnp.asarray(c), psi0=jnp.asarray(ps_))), q=jnp.asarray(q), dq=jnp.asarray(r8b.uniform(-0.1, 0.1, 3)))
    sg = env.gripper_event(st, jnp.asarray(1.0), pjr, tj)
    sg = dict(sg, dq=jnp.asarray(r8b.uniform(-2, 2, 3)))
    ke_obj = 0.5 * prr['mo'] * float(np.sum((np.asarray(jax.jacfwd(lambda qq: Mo.obj_pos(qq, env.pe(pjr, sg)))(jnp.asarray(q))) @ np.asarray(sg['dq'])) ** 2))
    tot0 = float(env.e_rob(sg, pjr) + env.s_obj(sg, pjr)); c_att = np.asarray(Mo.obj_pos(jnp.asarray(q), env.pe(pjr, sg)))
    sr = env.gripper_event(sg, jnp.asarray(0.0), pjr, tj); tot1 = float(env.e_rob(sr, pjr) + env.s_obj(sr, pjr))
    gain_max = max(gain_max, float(sg['ev_gain_max']), float(sr['ev_gain_max'])); rel_err = max(rel_err, abs((tot1 - tot0) + ke_obj)); crel_err = max(crel_err, float(np.max(np.abs(np.asarray(sr['obj_c']) - c_att))))
r8 = np.random.default_rng(8); dw = []
for _ in range(5000):
    pr = P.make_par(r8, ood=bool(r8.integers(2))); q = P.QC + P.QH * r8.uniform(-0.9, 0.9, 3); dq = r8.uniform(-0.1, 0.1, 3)
    dqp = np.asarray(Mo.attach_velocity(jnp.asarray(q), jnp.asarray(dq), J(pr))); dw.append(np.sum(np.abs((dqp - dq) / (2.0 * P.QH)) ** P.PG) ** (1 / P.PG))
R['V8'] = dict(truth_table=tt_ok, regrasp_sequence=regrasp_ok, grasp_point_after_release_err=grasp_point_after_release, grasp_energy_jump_err=float(jump_err), com_continuity=com_err,
               release_velocity_change=rel_unchanged, event_gain_max=gain_max, release_ke_err=rel_err, released_com_err=crel_err, gauge_jump_max=float(np.max(dw)))
PASS['V8'] = bool(tt_ok and regrasp_ok and grasp_point_after_release <= 1e-12 and jump_err <= 1e-9 and com_err <= 1e-12 and rel_unchanged == 0.0 and gain_max <= 1e-9 and rel_err <= 1e-9 and crel_err <= 1e-12)

# V9 sensing
pb, tb = env.stack(eps7[:4]); 
st, lg = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, 1500, 4, tau_contact, events=False)))(pb, tb, keys[:4])
st0 = dict(env.init_state(pj, tj)); 
def obs_seq(par, task, key, noise):
    def step(carry, k):
        st, key = carry; key, ko = jax.random.split(key)
        o = env.observe(st, par, task, ko, noise); tau = tau_contact(st, par, task); fn, p, v = env.top_force(st, par)
        tc = Mo.jac(st['q'], par['L']).T @ Mo.contact(p, v, par)[0]
        return (env.control_step(st, tau, par, 4), key), dict(o, tc=tc, qtrue=st['q'], ptrue=p)
    return jax.lax.scan(step, (env.init_state(par, task), key), jnp.arange(1500))[1]
p1, t1 = J(eps7[0][0]), J(eps7[0][1])
o0 = jax.jit(lambda: obs_seq(p1, t1, keys[0], 0.0))(); o1 = jax.jit(lambda: obs_seq(p1, t1, keys[0], 1.0))()
delay_err = float(np.max(np.abs(np.asarray(o0['tau_ext'][1:]) - np.asarray(t1['bias']) - np.asarray(o0['tc'][:-1]))))
first_err = float(np.max(np.abs(np.asarray(o0['tau_ext'][0]) - np.asarray(t1['bias']))))
contact_seen = float(np.max(np.abs(np.asarray(o0['tc']))))
big = jax.vmap(lambda k: env.observe(st0, pj, tj, k))(jax.random.split(jax.random.PRNGKey(5), 200000))
n_tau = np.asarray(big['tau_ext']) - np.asarray(tj['bias']); n_p = np.asarray(big['p']) - np.asarray(Mo.tip(st0['q'], pj['L']))
from scipy.stats import truncnorm
sd_trunc = P.SIG_TAU * truncnorm.std(-3, 3)
quant = float(np.max(np.abs(np.asarray(o1['q']) - np.asarray(o1['qtrue']))))
grid = float(np.max(np.abs(np.asarray(o1['q']) / P.Q_QUANT - np.round(np.asarray(o1['q']) / P.Q_QUANT))))
st_att = dict(st0, xi=jnp.asarray(1.0)); o_att = env.observe(st_att, pj, tj, keys[0], 0.0)
R['V9'] = dict(delay_err=delay_err, first_sample_err=first_err, contact_torque_max_in_run=contact_seen, noise_mean=float(np.max(np.abs(n_tau.mean(0)))), noise_mean_4se=float(4 * sd_trunc / np.sqrt(200000)),
               noise_sd_ratio=[float(x) for x in n_tau.std(0) / sd_trunc], noise_absmax=float(np.max(np.abs(n_tau))), quant_err=quant, grid_err=grid,
               attached_rest_err=float(np.max(np.abs(np.asarray(o_att['tau_ext']) - np.asarray(tj['bias'])))), p_sd_ratio=[float(x) for x in n_p.std(0) / P.SIG_P])
PASS['V9'] = bool(delay_err <= 1e-12 and first_err <= 1e-12 and contact_seen > 1.0 and R['V9']['noise_mean'] <= R['V9']['noise_mean_4se'] and all(abs(x - 1) <= 0.02 for x in R['V9']['noise_sd_ratio'])
                  and R['V9']['noise_absmax'] <= 0.15 + 1e-12 and quant <= 0.5e-4 + 1e-12 and grid <= 1e-6 and R['V9']['attached_rest_err'] <= 1e-12 and all(abs(x - 1) <= 0.02 for x in R['V9']['p_sd_ratio']))

# V10 sampler
def check_ep(par, task, ood_body, ood_stiff):
    L = par['L']; ok = []
    mg = best_margin(task['ps'], L, P.QC, P.QH)[0]; ok.append((P.MARG_HARD[0] <= mg < P.MARG_HARD[1]) if task['hard'] else (mg >= P.MARG_NORMAL))
    ok.append(best_margin(np.array([par['xe'], par['ye']]), L, P.QC, P.QH)[0] >= P.MARG_HARD[0]); ok.append(best_margin(task['po'], L, P.QC, P.QH)[0] >= P.MARG_NORMAL)
    ok.append(abs(task['po'][0] - task['ps'][0]) >= 0.2); ok.append(0.05 <= task['ps'][1] - par['ye'] <= 0.15 and par['xe'] == task['ps'][0])
    ok.append(all(P.BOX[i][0] <= x[i] <= P.BOX[i][1] for x in (task['po'], task['ps']) for i in (0, 1)))
    ok.append(bool(np.all(np.abs(task['q0'] - P.QC) <= 0.6 * P.QH + 1e-12)))
    th = np.cumsum(task['q0']); ok.append(np.sum(L * np.sin(th)) >= par['ye'] + P.CLEAR and task['po'][1] >= par['ye'] + P.CLEAR)
    lo, hi = np.array([1.5, 2, 2, 1.5]), np.array([2.5, 3, 3, 2.5]); d = task['nph'] * P.TS
    ok.append(bool(np.all(task['nph'] % P.NB == 0) and np.all(d >= lo - P.TH) and np.all(d <= hi + P.TH)))
    fl, fh = task['f_light'], task['f_heavy']; cls = par['mo'] > P.M_TH
    ok.append(3 <= fl <= 15 and 3 <= fh <= 15 and abs(fl - fh) >= P.F_SEP and task['fstar'] == (fh if cls else fl) and task['heavy'] == float(cls) and 0.6 <= task['rho'] <= 1.0
              and (2e4 <= par['ke'] <= 2e5 if ood_stiff else 2e3 <= par['ke'] <= 2e4) and 0.5 <= par['de'] <= 5 and 0.2 <= par['nu'] <= 0.8)
    sc = par['m'] / P.M_NOM; sl = L / P.L_NOM
    ok.append(bool(np.all(((0.7 <= sc) & (sc <= 0.9)) | ((1.1 <= sc) & (sc <= 1.3))) and np.all(((0.85 <= sl) & (sl <= 0.95)) | ((1.05 <= sl) & (sl <= 1.15))) and 1.5 <= par['mo'] <= 2.5) if ood_body
              else bool(np.all((0.9 <= sc) & (sc <= 1.1)) and np.all(sl == 1) and (0.2 <= par['mo'] <= 0.65 or 1.05 <= par['mo'] <= 1.5)))
    ok.append(0 <= par['Lobj'] <= P.LOBJ_MAX and abs(np.linalg.norm(task['c0'] - task['po']) - par['Lobj']) <= 1e-12 and bool(np.all(np.abs(task['bias']) <= P.BIAS_TAU)) and bool(np.all((0.05 <= par['Fv']) & (par['Fv'] <= 0.3))) and bool(np.all(np.abs(par['d']) <= 0.5)))
    return ok
r10 = np.random.default_rng(10); fails = np.zeros(12, int); nh = nb = 0; nheavy = nid = 0
for i in range(3000):
    ob, os_ = (2000 <= i < 2500), (i >= 2500); par, task = env.sample_episode(r10, ood_body=ob, ood_stiff=os_)
    fails += ~np.array(check_ep(par, task, ob, os_)); nh += task['hard']; nb += task['budget']
    if not ob: nheavy += task['heavy']; nid += 1
sd3 = 3 * np.sqrt(0.3 * 0.7 / 3000); sdc = 3 * np.sqrt(0.25 / nid)
R['V10'] = dict(failures_per_check=fails.tolist(), hard_frac=float(nh / 3000), budget_frac=float(nb / 3000), tol=float(sd3), heavy_frac=float(nheavy / nid), heavy_tol=float(sdc))
PASS['V10'] = bool(fails.sum() == 0 and abs(nh / 3000 - 0.3) <= sd3 and abs(nb / 3000 - 0.3) <= sd3 and abs(nheavy / nid - 0.5) <= sdc)

# V11 cost (independent NumPy evaluation of the definition, revision 15)
def cost_np(p, fn, xi, j, po, ps, fs, ye):
    o = p if xi else po; to_obj = np.linalg.norm(p - o) / 0.1; pc = np.array([ps[0], ye])
    press = lambda y, f: min(((f - fs) / fs) ** 2, 1.0) if f > 0 else 1 + np.linalg.norm(y - pc) / 0.1
    if j == 0: om = to_obj
    elif j == 1: om = np.linalg.norm(o - ps) / 0.1 + (0.0 if xi else 1 + to_obj)
    elif j == 2: om = (press(p, fn) + abs(p[0] - ps[0]) / 0.1) if xi else (1 + to_obj + press(o, 0.0) + abs(o[0] - ps[0]) / 0.1)
    else: om = np.linalg.norm(p - ps) / 0.1 + xi
    return om + 5 * max(fn - 2 * fs, 0) / fs
par, task = eps7[1]; pj2, tj2 = J(par), J(task); edges = np.cumsum(task['nph']); e11 = 0.0; n11 = 0
qs = np.asarray(task['qs'])
def q_at(pt): return best_margin(pt, par['L'], P.QC, P.QH)[1]
pts = [(0, task['po'] + np.array([0.03, 0.04]), 0, 0.0), (1, task['ps'] + np.array([0.0, 0.02]), 1, 0.0), (1, task['ps'] + np.array([0.05, 0.0]), 0, 0.0),
       (2, np.array([par['xe'] + 0.02, par['ye'] + 0.03]), 1, 0.0), (2, np.array([par['xe'], par['ye'] - 0.0005]), 1, 0.0), (2, np.array([par['xe'] - 0.03, par['ye'] - 0.004]), 0, 0.0),
       (2, np.array([par['xe'], par['ye'] - 0.0005]), 1, -0.3), (3, task['ps'], 0, 0.0), (3, task['ps'] + np.array([0.02, -0.01]), 1, 0.0), (2, np.array([par['xe'], par['ye'] - 0.01]), 1, 0.0),
       (0, task['po'] + np.array([0.01, 0.0]), 1, 0.0), (2, np.array([par['xe'], par['ye'] - 0.002]), 0, 0.0)]
for j, pt, xi, vy in pts:
    q = q_at(pt); Jq = np.asarray(Mo.jac(jnp.asarray(q), pj2['L'])); dq = np.linalg.pinv(Jq) @ np.array([0.0, vy])
    k = int(edges[j] - 5) if j < 3 else int(edges[2] + 5)
    st = dict(env.init_state(pj2, tj2), q=jnp.asarray(q), dq=jnp.asarray(dq), xi=jnp.asarray(float(xi)), k=jnp.asarray(k))
    fn, p, v = env.top_force(st, pj2)
    fn_np = par['ke'] * (par['ye'] - float(p[1])) * max(1.0 - par['de'] * float(v[1]), 0.0) if (float(p[1]) < par['ye'] and abs(float(p[0]) - par['xe']) < P.WE and par['ye'] - float(p[1]) <= P.WE - abs(float(p[0]) - par['xe'])) else 0.0
    e11 = max(e11, abs(float(env.cost_rate(st, pj2, tj2)) - cost_np(np.asarray(p), fn_np, xi, j, task['po'], task['ps'], task['fstar'], par['ye'])), abs(float(fn) - fn_np)); n11 += 1
R['V11'] = dict(n=n11, max_err=e11); PASS['V11'] = bool(e11 <= 1e-9)

# V12-V14 end to end with the privileged test controller
eps = [H.add_postures(*env.sample_episode(np.random.default_rng(400 + s))) for s in range(64)]
pb, tb = env.stack(eps); nst = int(max(int(np.sum(e[1]['nph'])) for e in eps)); keys64 = jax.random.split(jax.random.PRNGKey(1), 64)
roll = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, nst, 4)))
t0 = time.time(); st, lg = roll(pb, tb, keys64); jax.block_until_ready(st['q']); t1 = time.time(); st_b, lg_b = roll(pb, tb, keys64); jax.block_until_ready(st_b['q']); t2 = time.time()
lgn = {k: np.asarray(v) for k, v in lg.items()}
same = max(float(np.max(np.abs(np.asarray(lg[k]) - np.asarray(lg_b[k])))) for k in lg)
single = jax.jit(lambda p, t, k: H.rollout(p, t, k, nst, 4)); dmax = 0.0
for i in (0, 17, 42):
    s1, l1 = single(J(eps[i][0]), J(eps[i][1]), keys64[i]); dmax = max(dmax, max(float(np.max(np.abs(np.asarray(l1[k]) - lgn[k][i]))) for k in ('q', 'dq', 'fn', 'e', 'cost', 'tau_ext')))
ntot = np.array([int(np.sum(e[1]['nph'])) for e in eps]); valid = np.arange(nst)[None, :] < ntot[:, None]
base = lgn['e'] - lgn['e_jump']                                       # E_rob without the event jumps; reference taken before any event (also at k = 0)
res = np.abs(base - base[:, :1] - (lgn['w_act'] - lgn['w_fric'] + lgn['w_env'])); sc = np.maximum(1.0, np.max(np.abs(lgn['w_act']), 1, keepdims=True))
edges = np.cumsum(np.stack([e[1]['nph'] for e in eps]), 1); kk = np.arange(nst)[None, :]
ph_expect = np.minimum((kk >= edges[:, :1]).astype(int) + (kk >= edges[:, 1:2]) + (kk >= edges[:, 2:3]) + (kk >= edges[:, 3:4]), 3)
dxi = np.diff(lgn['xi'], axis=1) != 0; change_steps = np.nonzero(dxi)[1] + 1
grasped = lgn['xi'][np.arange(64), edges[:, 0] - 1] > 0.5
w3 = (kk >= edges[:, 1:2] + (0.6 * (edges[:, 2:3] - edges[:, 1:2])).astype(int)) & (kk < edges[:, 2:3])
ferr = np.array([np.mean(np.abs(lgn['fn'][i][w3[i]] - eps[i][1]['fstar'])) for i in range(64)])
w2 = (kk >= edges[:, :1] + (0.6 * (edges[:, 1:2] - edges[:, :1])).astype(int)) & (kk < edges[:, 1:2])
perr = np.array([np.mean(np.linalg.norm(lgn['p'][i][w2[i]] - eps[i][1]['ps'], axis=-1)) for i in range(64)])
R['V12'] = dict(batch_vs_single=dmax, rerun_diff=same, steps_per_s=float(64 * nst / (t2 - t1)), compile_s=float(t1 - t0 - (t2 - t1)))
R['V13'] = dict(phase_index_ok=bool(np.all(lgn['ph'] == ph_expect)), xi_change_steps_all_decision=bool(np.all(change_steps % P.NB == 0)), n_xi_changes=int(dxi.sum()),
                episode_steps_range=[int(ntot.min()), int(ntot.max())])
R['V14'] = dict(all_finite=bool(all(np.all(np.isfinite(v)) for v in lgn.values())), audit_residual_rel=float(np.max((res / sc)[valid])), grasp_frac=float(np.mean(grasped)), n_over=float(np.max(np.asarray(st['n_over']))),
                press_force_err_median=float(np.median(ferr)), press_force_err_max=float(np.max(ferr)), hold_pos_err_median=float(np.median(perr)), fn_peak_max=float(lgn['fn'].max()),
                n_grasp=[int(x) for x in np.unique(np.asarray(st['n_grasp']))], n_release=[int(x) for x in np.unique(np.asarray(st['n_release']))], event_gain_max=float(np.max(np.asarray(st['ev_gain_max']))))
PASS['V13'] = bool(R['V13']['phase_index_ok'] and R['V13']['xi_change_steps_all_decision'])
PASS['V14'] = bool(R['V14']['all_finite'] and R['V14']['audit_residual_rel'] <= 1e-4 and R['V14']['grasp_frac'] >= 0.9)

# V16 re-grasp end to end: release while holding at the target, move to another posture with the same tip position, grasp again
eps16 = [H.add_postures(*env.sample_episode(np.random.default_rng(600 + s), hard=False)) for s in range(64)]
pb16, tb16 = env.stack(eps16); nst16 = int(max(int(np.sum(e[1]['nph'])) for e in eps16)); n16 = np.array([int(np.sum(e[1]['nph'])) for e in eps16])
st16, lg16 = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, nst16, 4, H.tau_regrasp, H.psi_regrasp)))(pb16, tb16, jax.random.split(jax.random.PRNGKey(3), 64))
l16 = {k: np.asarray(v) for k, v in lg16.items()}; s16 = {k: np.asarray(v) for k, v in st16.items()}
base16 = l16['e'] - l16['e_jump']
res16 = np.abs(base16 - base16[:, :1] - (l16['w_act'] - l16['w_fric'] + l16['w_env'])); sc16 = np.maximum(1.0, np.max(np.abs(l16['w_act']), 1, keepdims=True))
valid16 = np.arange(nst16)[None, :] < n16[:, None]
two = (s16['n_grasp'] == 2) & (s16['n_release'] == 2)
lobj = np.array([e[0]['Lobj'] for e in eps16])
R['V16'] = dict(frac_two_grasps_two_releases=float(np.mean(two)), n_grasp_values=[int(x) for x in np.unique(s16['n_grasp'])], event_gain_max=float(np.max(s16['ev_gain_max'])),
                com_continuity_max=float(np.max(s16['com_err_max'])), offset_excess_max=float(np.max(s16['r_max'] - lobj - P.GRASP_DIST)), audit_residual_rel=float(np.max((res16 / sc16)[valid16])),
                all_finite=bool(all(np.all(np.isfinite(v)) for v in l16.values())), dtheta3_median=float(np.median([e[1]['dth3'] for e in eps16])), n_over=float(np.max(s16['n_over'])))
PASS['V16'] = bool(R['V16']['frac_two_grasps_two_releases'] >= 0.9 and R['V16']['event_gain_max'] <= 1e-9 and R['V16']['com_continuity_max'] <= 1e-12
                   and R['V16']['offset_excess_max'] <= 1e-12 and R['V16']['audit_residual_rel'] <= 1e-4 and R['V16']['all_finite'])

# V17 cost properties (revision 14): (a) dropping at the tip's pose raises the rate by exactly 1 without top-face force (phases 2, 3); (b) slope with depth along the side wall
par17, task17 = eps7[1]; pj17, tj17 = J(par17), J(task17); s17 = env.init_state(pj17, tj17); ed17 = np.cumsum(task17['nph'])
Q17 = jnp.asarray(np.asarray(P.QC) + np.asarray(P.QH) * np.random.default_rng(17).uniform(-1, 1, (200000, 3)))
def st17(q, k, xi):
    p = Mo.tip(q, pj17['L'])
    return dict(s17, q=q, dq=jnp.zeros(3), k=jnp.asarray(k), xi=jnp.asarray(xi), obj_psi=jnp.asarray(0.0), obj_c=p - pj17['Lobj'] * jnp.array([1.0, 0.0]))
cf17 = jax.jit(jax.vmap(lambda q, k, xi: env.cost_rate(st17(q, k, xi), pj17, tj17), in_axes=(0, None, None)))
P17 = np.asarray(jax.vmap(lambda q: Mo.tip(q, pj17['L']))(Q17)); free17 = np.asarray(jax.vmap(lambda q: env.top_force(st17(q, 0, 1.0), pj17)[0])(Q17)) <= 0
dev17 = float(max(np.max(np.abs((np.asarray(cf17(Q17, int(k), 0.0)) - np.asarray(cf17(Q17, int(k), 1.0)))[free17] - 1.0)) for k in (ed17[0] + 5, ed17[1] + 5)))
side17 = (np.abs(P17[:, 0] - par17['xe']) >= par17['we']) & (np.abs(P17[:, 0] - par17['xe']) <= par17['we'] + 0.01) & (P17[:, 1] < par17['ye']) & (P17[:, 1] > par17['ye'] - 0.3)
slope17 = float(np.polyfit(par17['ye'] - P17[side17, 1], np.asarray(cf17(Q17, int(ed17[1] + 5), 1.0))[side17], 1)[0])
R['V17'] = dict(drop_minus_held_max_dev=dev17, n_free=int(free17.sum()), side_states=int(side17.sum()), side_slope=slope17)
PASS['V17'] = bool(dev17 <= 1e-9 and side17.sum() >= 100 and slope17 >= 1.0)

# V18 cost properties with top-face force (revision 15): tip at the center of the top face, 200 depths with forces 0.05 f* .. 20 f*, phase 3
fs18 = float(task17['fstar']); d18 = np.geomspace(0.05, 20.0, 200) * fs18 / par17['ke']; k18 = int(ed17[1] + 5)
q18 = [best_margin(np.array([par17['xe'], par17['ye'] - d]), par17['L'], P.QC, P.QH)[1] for d in d18]
def r18(q, xi):
    st = dict(st17(jnp.asarray(q), k18, xi)); return float(env.top_force(st, pj17)[0]), float(env.cost_rate(st, pj17, tj17))
F18 = np.array([r18(q, 1.0)[0] for q in q18]); H18 = np.array([r18(q, 1.0)[1] for q in q18]); D18 = np.array([r18(q, 0.0)[1] for q in q18])
i18 = int(np.argmin(np.abs(F18 - fs18))); lo18, hi18 = H18[:i18 + 1], H18[i18:]
R['V18'] = dict(force_range=[float(F18.min() / fs18), float(F18.max() / fs18)], drop_minus_held_min=float((D18 - H18).min()), argmin_force_ratio=float(F18[int(np.argmin(H18))] / fs18),
                nearest_force_ratio=float(F18[i18] / fs18), mono_left_max_increase=float(np.max(np.diff(lo18))) if len(lo18) > 1 else 0.0, mono_right_min_increase=float(np.min(np.diff(hi18))) if len(hi18) > 1 else 0.0)
PASS['V18'] = bool(np.all(F18 > 0) and (D18 - H18).min() >= 1 - 1e-9 and int(np.argmin(H18)) == i18 and R['V18']['mono_left_max_increase'] <= 1e-9 and R['V18']['mono_right_min_increase'] >= -1e-9)
np.savez(str(paths.RECORDS / 'stageb_env_episode.npz'), **{k: lgn[k][:4] for k in ('p', 'fn', 'xi', 'ph', 'q', 'cost')}, res=(res / sc)[:4], ntot=ntot[:4],
         po=np.stack([e[1]['po'] for e in eps[:4]]), ps=np.stack([e[1]['ps'] for e in eps[:4]]), fstar=np.array([e[1]['fstar'] for e in eps[:4]]),
         xe=np.array([e[0]['xe'] for e in eps[:4]]), ye=np.array([e[0]['ye'] for e in eps[:4]]), L=np.stack([e[0]['L'] for e in eps[:4]]))
paths.TMP.mkdir(parents=True, exist_ok=True); np.savez(paths.TMP / 'stageb_x64_ref.npz', q=lgn['q'][:8], fn=lgn['fn'][:8], xi=lgn['xi'][:8])
out = subprocess.run([sys.executable, str(paths.ROOT / 'eval' / 'verify' / 'f32_run.py')], capture_output=True, text=True)
try: R['V12']['float32'] = json.loads(out.stdout.strip().splitlines()[-1])
except Exception as ex: R['V12']['float32'] = dict(error=out.stderr[-300:])
PASS['V12'] = bool(dmax <= 1e-9 and same == 0.0)

# V15 trajectory vs MuJoCo RK4
d15 = 0.0
for s in range(8):
    r = np.random.default_rng(500 + s); par = P.make_par(r, ood=bool(s % 2)); xi = float(s % 4 >= 2); q0 = r.uniform(-1.5, 1.5, 3); om, phi = r.uniform(2, 15, 3), r.uniform(0, 6.28, 3)
    mdl, dat = refs.mujoco_model(par, xi); dat.qpos[:] = q0; dat.qvel[:] = 0; mujoco.mj_forward(mdl, dat)
    st = dict(env.init_state(J(par), dict(q0=jnp.asarray(q0), c0=jnp.zeros(2), psi0=jnp.asarray(0.0))), xi=jnp.asarray(xi)); pj3 = J(par); stepf = jax.jit(lambda s_, t_: env.control_step(s_, t_, pj3, 4))
    for k in range(500):
        tau = np.array([20.0, 10.0, 3.0]) * np.sin(om * k * P.TS + phi)
        dat.qfrc_applied[:] = tau + par['d']
        for _ in range(4): mujoco.mj_step(mdl, dat)
        st = stepf(st, jnp.asarray(tau))
    d15 = max(d15, float(np.max(np.abs(np.asarray(st['q']) - dat.qpos))))
R['V15'] = dict(max_joint_diff=d15); PASS['V15'] = bool(d15 <= 1e-6)
R['pass'] = PASS; R['all_pass'] = bool(all(PASS.values()))
json.dump(R, open(str(paths.RECORDS / 'stageb_env_check.json'), 'w'), indent=1)
for k in sorted(PASS, key=lambda x: int(x[1:])): print(k, 'PASS' if PASS[k] else 'FAIL', json.dumps(R[k])[:420])
print('ALL PASS' if R['all_pass'] else 'SOME FAILED')
