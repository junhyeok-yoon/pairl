"""L1 task-validity runs: Stage B, v3 controller, fixed memoryless brains.
Each brain is a function of the brain observation o_k = (q, dq, tau_ext, task context, ledger) without q_d and K (spec rev. 13); no brain state.
obs_qd=True restores the superseded observation with q_d and K (used only to reproduce the L1 pilot).
Postures from the nominal kinematics are functions of the context and are precomputed per episode."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, env, params as P, controller as C
from pairl.stageb.kin_np import ik_all, gauge

LNOM, MNOM = jnp.asarray(P.L_NOM), jnp.asarray(P.M_NOM)
QLO, QHI = jnp.asarray(P.QC - P.QH), jnp.asarray(P.QC + P.QH)

def pe_arm_np(q, L=P.L_NOM, m=P.M_NOM):
    th = np.cumsum(q, -1); y = np.cumsum(L * np.sin(th), -1) - L / 2 * np.sin(th)
    return Mo.G0 * np.sum(m * y, -1)

def pe_arm(q):
    """Potential energy of the nominal arm without load (JAX)."""
    th = jnp.cumsum(q); y = jnp.cumsum(LNOM * jnp.sin(th)) - LNOM / 2 * jnp.sin(th)
    return Mo.G0 * jnp.sum(MNOM * y)

def postures(p, lim):
    """Best-margin posture and minimum-potential-energy posture among those with gauge <= max(lim, best gauge)."""
    q, ok = ik_all(p, P.L_NOM); rho = np.where(ok, gauge(q, P.QC, P.QH), np.inf)
    ib = int(np.argmin(rho)); pe = pe_arm_np(q)
    il = int(np.argmin(np.where(rho <= max(lim, rho[ib] + 1e-12), pe, np.inf)))
    return q[ib], q[il]

def prep(par, task, lim=0.8):
    task = dict(task)
    task['qo_b'], task['qo_l'] = postures(task['po'], lim)
    task['qs_b'], task['qs_l'] = postures(task['ps'], lim)
    return par, task

def jac_nom(q): return Mo.jac(q, LNOM)
def pinv(J): return J.T @ jnp.linalg.inv(J @ J.T)
def tip_force(q, te):
    J = jac_nom(q); return jnp.linalg.solve(J @ J.T, J @ te)

# brains: (ob, ctx, cfg, opts) -> (q_d^p, K^p, psi). ob: measured signals and ledger (q_d and K only with obs_qd, the superseded observation);
# ctx: task context without the active force target, plus nominal postures computed from it.
CTX_KEYS = ('po', 'ps', 'f_light', 'f_heavy', 'qo_b', 'qo_l', 'qs_b', 'qs_l')

def _goals(ob, ctx, qo, qs, press):
    """Common phase logic: grasp, carry to the target right after the grasp, press, release and retreat; without the object go back to it."""
    j, holding = ob['j'], ob['xi'] > 0.5
    g = jnp.where(holding, jnp.where(j == 2, press, qs), qo)
    qdp = jnp.where(j == 3, qs, g)
    return jnp.clip(qdp, QLO, QHI), jnp.where(j <= 2, 1.0, 0.0)

def _vdown(ob, ctx, opts):
    """Descent speed of the goal: v_fast while the tip is above p*_y - y_fast (no block can be there), v_d below."""
    return jnp.where(ob['p'][1] > ctx['ps'][1] - opts['y_fast'], opts['v_fast'], opts['v_d'])

def f_select(ob, ctx, opts):
    """Force target. f_mode 1: class from the learned load in the ledger (vertical tip load > M_TH g -> f_heavy);
    f_mode 0 (no ledger): the constant minimizing the expected squared relative error for equally likely classes."""
    a, b = ctx['f_light'], ctx['f_heavy']
    heavy = tip_force(ob['q'], ob['uload'])[1] > P.M_TH * Mo.G0
    f_led = jnp.where(heavy, b, a)
    f_const = (1.0 / a + 1.0 / b) / (1.0 / a ** 2 + 1.0 / b ** 2)
    return jnp.where(opts['f_mode'] > 0.5, f_led, f_const)

def press_ff(ob, ctx, cfg, opts):
    """Press without integral action: descend until contact is sensed, then a spring offset from the measured posture that produces
    the selected force (gravity is left to the hand). Without q_d the descent leads the measured posture by v lead_t."""
    q = ob['q']; J = jac_nom(q); Jp = pinv(J); z = 0.0 * q[0]
    dx = opts['k_x'] * (ctx['ps'][0] - ob['p'][0])
    contact = jnp.linalg.norm(ob['te']) > cfg['tau_th']
    if 'qd' in ob:
        base = jnp.where(opts['noqd'] > 0.5, q, ob['qd']); step = jnp.where(opts['noqd'] > 0.5, opts['lead_t'], P.TH)
        Kuse = jnp.where(opts['noqd'] > 0.5, opts['K'], ob['K'])
    else:
        base, step, Kuse = q, opts['lead_t'], opts['K']
    descend = base + Jp @ jnp.stack([dx, -_vdown(ob, ctx, opts) * step])
    press = q + Jp @ jnp.stack([dx, z]) - (J.T @ jnp.stack([z, f_select(ob, ctx, opts)])) / Kuse
    return jnp.where(contact, press, descend)

def press_int(ob, ctx, cfg, opts):
    """Press through the realized goal (needs q_d): descend by moving q_d, then integrate the force and x errors into q_d."""
    q, qd = ob['q'], ob['qd']; J = jac_nom(q); Jp = pinv(J); z = 0.0 * q[0]
    dx = opts['k_x'] * (ctx['ps'][0] - ob['p'][0])
    contact = jnp.linalg.norm(ob['te']) > cfg['tau_th']; f = tip_force(q, ob['te'])
    descend = qd + Jp @ jnp.stack([dx, -_vdown(ob, ctx, opts) * P.TH])
    press = qd + Jp @ jnp.stack([dx, z]) - (J.T @ jnp.stack([z, opts['g_f'] * (f_select(ob, ctx, opts) - f[1])])) / ob['K']
    return jnp.where(contact, press, descend)

def task_corr(ob, pt, opts):
    """Proportional task-space correction of a goal posture from the measured tip (no integral action): k_c J^+ e, |e| clipped to 0.05 m."""
    e = pt - ob['p']; e = e * jnp.minimum(1.0, 0.05 / jnp.maximum(jnp.linalg.norm(e), 1e-9))
    return opts['k_c'] * pinv(jac_nom(ob['q'])) @ e

def brain_ref(ob, ctx, cfg, opts, low=False):
    qo, qs = (ctx['qo_l'], ctx['qs_l']) if low else (ctx['qo_b'], ctx['qs_b'])
    qo = qo + task_corr(ob, ctx['po'], opts); qs = qs + task_corr(ob, ctx['ps'], opts)
    qdp, psi = _goals(ob, ctx, qo, qs, press_ff(ob, ctx, cfg, opts))
    return qdp, opts['K'], psi

def brain_low(ob, ctx, cfg, opts): return brain_ref(ob, ctx, cfg, opts, low=True)

def brain_int(ob, ctx, cfg, opts):
    """Integral action through the observed realized goal (needs q_d): q_d^p = q_d + g_p (q_target - q); force and x errors integrated in the press."""
    q, qd = ob['q'], ob['qd']
    qo = qd + opts['g_p'] * (ctx['qo_b'] - q); qs = qd + opts['g_p'] * (ctx['qs_b'] - q)
    qdp, psi = _goals(ob, ctx, qo, qs, press_int(ob, ctx, cfg, opts))
    return qdp, opts['K'], psi

def brain_intr(ob, ctx, cfg, opts):
    """Integral action near the target, direct jump far from it (needs q_d): q_d^p = q_t + [|q_d - q_t|_inf < c_max](q_d - q_t) + g_p (q_t - q)."""
    q, qd = ob['q'], ob['qd']
    def goal(qt):
        c = qd - qt
        return qt + jnp.where(jnp.max(jnp.abs(c)) < opts['c_max'], c, 0.0 * c) + opts['g_p'] * (qt - q)
    qdp, psi = _goals(ob, ctx, goal(ctx['qo_b']), goal(ctx['qs_b']), press_int(ob, ctx, cfg, opts))
    return qdp, opts['K'], psi

def ik_th3(p, th3):
    """Planar 3R inverse kinematics (nominal lengths) for the tip p and the absolute angle th3 of the last link, branch q2 < 0."""
    L = LNOM; w = p - L[2] * jnp.stack([jnp.cos(th3), jnp.sin(th3)])
    c2 = jnp.clip((w @ w - L[0] ** 2 - L[1] ** 2) / (2 * L[0] * L[1]), -1.0, 1.0); q2 = -jnp.arccos(c2)
    q1 = jnp.arctan2(w[1], w[0]) - jnp.arctan2(L[1] * jnp.sin(q2), L[0] + L[1] * jnp.cos(q2))
    return jnp.stack([q1, q2, th3 - q1 - q2])

def step_goal(ob, pt, th3t, dmax, amax):
    """Goal posture a bounded step ahead of the measured tip on the straight line to pt, with the last-link angle stepped toward th3t."""
    d = pt - ob['p']; n = jnp.linalg.norm(d)
    pv = ob['p'] + d * jnp.minimum(1.0, dmax / jnp.maximum(n, 1e-9))
    th3 = jnp.sum(ob['q']); th3v = th3 + jnp.clip(th3t - th3, -amax, amax)
    return ik_th3(pv, th3v)

def brain_step(ob, ctx, cfg, opts, eco):
    """Task-space stepping brain. eco (0/1 array): shorter steps, and minimum-energy postures if eco_low."""
    dmax = jnp.where(eco > 0.5, opts['d_eco'], opts['d_fast']); low = (eco > 0.5) & (opts['eco_low'] > 0.5)
    th_o = jnp.where(low, jnp.sum(ctx['qo_l']), jnp.sum(ctx['qo_b'])); th_s = jnp.where(low, jnp.sum(ctx['qs_l']), jnp.sum(ctx['qs_b']))
    go = step_goal(ob, ctx['po'], th_o, dmax, opts['a_max']); gs = step_goal(ob, ctx['ps'], th_s, dmax, opts['a_max'])
    qdp, psi = _goals(ob, ctx, go, gs, press_ff(ob, ctx, cfg, dict(opts, noqd=1.0)))
    return qdp, opts['K'], psi

def brain_fast(ob, ctx, cfg, opts): return brain_step(ob, ctx, cfg, opts, jnp.asarray(0.0))
def brain_eco(ob, ctx, cfg, opts): return brain_step(ob, ctx, cfg, opts, jnp.asarray(1.0))
def brain_ledT(ob, ctx, cfg, opts):
    """Ledger brain on the tank level only: eco mode while T - T_min < theta."""
    return brain_step(ob, ctx, cfg, opts, (ob['T'] - cfg['Tmin'] < opts['theta']).astype(ob['q'].dtype))

def brain_ledJ(ob, ctx, cfg, opts):
    """Ledger brain: the reference brain while T - T_min >= theta, slow stepping otherwise."""
    q1, K1, p1 = brain_ref(ob, ctx, cfg, dict(opts, noqd=1.0))
    q2, K2, p2 = brain_step(ob, ctx, cfg, opts, jnp.asarray(1.0))
    use1 = ob['T'] - cfg['Tmin'] >= opts['theta']
    return jnp.where(use1, q1, q2), jnp.where(use1, K1, K2), jnp.where(use1, p1, p2)

def brain_led(ob, ctx, cfg, opts):
    """Ledger brain: best-margin postures while the tank covers their estimated energy, minimum-energy postures otherwise.
    Estimate: change of the nominal arm potential energy to the goal posture plus the learned load times the lift (load from u_load)."""
    q = ob['q']; j, holding = ob['j'], ob['xi'] > 0.5
    tq_b = jnp.where(holding | (j == 3), ctx['qs_b'], ctx['qo_b'])
    ty = jnp.where(holding | (j == 3), ctx['ps'][1], ctx['po'][1])
    w = jnp.maximum(tip_force(q, ob['uload'])[1], 0.0) * holding
    need = pe_arm(tq_b) - pe_arm(q) + w * (ty - ob['p'][1])
    avail = ob['T'] - cfg['Tmin'] - cfg['dT'] - opts['e_marg']
    use_b = (avail >= need) | (j == 2)
    qo = jnp.where(use_b, ctx['qo_b'], ctx['qo_l']); qs = jnp.where(use_b, ctx['qs_b'], ctx['qs_l'])
    qdp, psi = _goals(ob, ctx, qo, qs, press_ff(ob, ctx, cfg, opts))
    return qdp, opts['K'], psi

BRAINS = dict(ref=brain_ref, low=brain_low, int=brain_int, intr=brain_intr, led=brain_led, fast=brain_fast, eco=brain_eco, ledT=brain_ledT, ledJ=brain_ledJ)

def default_opts(cfg):
    return dict(K=jnp.asarray(cfg['Kn']), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, g_p=0.25, g_f=0.3, c_max=0.15, e_marg=0.5, noqd=0.0, lead_t=0.2,
                d_fast=0.15, d_eco=0.05, a_max=0.3, theta=0.8, eco_low=0.0, k_c=0.0, f_mode=1.0)

# rollout: outer scan over decisions, inner scan over control periods
def rollout(par, task, key, nd, cfg, bname, opts, nsub=4, log_every=0, obs_qd=False, memfn=None):
    bfn = BRAINS[bname]; ctx = {k: task[k] for k in CTX_KEYS}
    edges = jnp.cumsum(task['nph']); nwin = jnp.round(0.4 * task['nph']).astype(edges.dtype)
    st0 = env.init_state(par, task); cs0 = C.init_ctrl(task['q0'], task['E0'], cfg, 7)
    big = jnp.asarray(1e30); z0 = jnp.asarray(0.0)
    m0 = dict(cost=z0, cost_ph=jnp.zeros(4), pos_err=jnp.zeros(4), pos_n=jnp.zeros(4), f_err=z0, f_rel=z0, f_n=z0, grasp_end1=z0, t_grasp=z0 - 1.0,
              peak_max=z0, peak_first=z0, t_first=z0 - 1.0, n_onset=z0, t_on=z0 - 1.0, fn_prev=z0, gate=z0, cons=z0, sat=z0, guard=z0, nval=z0, minT=big,
              E_task=-big, mis_hold=z0, mis_n=z0, w_hat=z0, viol=z0, upd=z0, nan=z0, fn_sum=z0,
              fstar=z0 + task['fstar'], fother=z0 + task['f_light'] + task['f_heavy'] - task['fstar'], heavy=z0 + task['heavy'],
              Ub_run=z0, tank_bound=big, sat_below=z0)
    def inner(carry, k):
        st, cs, prop, m, key = carry
        key, ko = jax.random.split(key)
        obs = env.observe(st, par, task, ko)
        tau, cs, dg = C.ctrl_step(cs, obs, prop, cfg, LNOM, MNOM, memfn=memfn)
        val = (k < edges[3]).astype(tau.dtype); j = env.phase(k, task); t = k * P.TS
        fn, p, _ = env.top_force(st, par); om = env.cost_rate(st, par, task)
        ustar = Mo.gravity(st['q'], env.pe(par, st), st['xi']) - par['d']; mis = jnp.linalg.norm(dg['uad'] - ustar)
        what = tip_force(obs['q'], dg['uad'] - C.prior_body(obs['q'], LNOM, MNOM))[1]          # vertical tip load from the learned compensation
        st_n = env.control_step(st, tau, par, nsub)
        Ub_run = jnp.maximum(m['Ub_run'], C.barrier_potential(st['q'], cfg))                      # running max of the barrier storage
        inwin = val * ((k >= edges[j] - nwin[j]) & (k < edges[j])).astype(tau.dtype)
        perr = jnp.linalg.norm(p - task['ps']); onset = (fn > 0) & (m['fn_prev'] <= 0) & (val > 0)
        t_on = jnp.where(onset, t, m['t_on']); near_on = (t_on >= 0) & (t - t_on <= 0.3) & (val > 0)
        t_first = jnp.where(onset & (m['t_first'] < 0) & (j == 2), t, m['t_first']); near_first = (t_first >= 0) & (t - t_first <= 0.3) & (val > 0)
        ph = jax.nn.one_hot(j, 4, dtype=tau.dtype)
        pw = inwin * ph * jnp.asarray([0.0, 1.0, 0.0, 1.0])
        fw = inwin * (j == 2)
        m = dict(m, cost=m['cost'] + val * om * P.TS, cost_ph=m['cost_ph'] + val * om * P.TS * ph,
                 pos_err=m['pos_err'] + pw * perr, pos_n=m['pos_n'] + pw,
                 f_err=m['f_err'] + fw * jnp.abs(fn - task['fstar']), f_rel=m['f_rel'] + fw * jnp.abs(fn - task['fstar']) / task['fstar'], f_n=m['f_n'] + fw, fn_sum=m['fn_sum'] + fw * fn,
                 grasp_end1=jnp.where(k == edges[0] - 1, st['xi'], m['grasp_end1']),
                 t_grasp=jnp.where((m['t_grasp'] < 0) & (st['xi'] > 0.5), t, m['t_grasp']),
                 peak_max=jnp.where(near_on, jnp.maximum(m['peak_max'], fn), m['peak_max']),
                 peak_first=jnp.where(near_first, jnp.maximum(m['peak_first'], fn), m['peak_first']), t_first=t_first, n_onset=m['n_onset'] + onset, t_on=t_on, fn_prev=fn,
                 gate=m['gate'] + val * (dg['alpha'] < 1.0), cons=m['cons'] + val * (jnp.linalg.norm(dg['tau_b']) > 0), sat=m['sat'] + val * (dg['lam'] < 1.0),
                 guard=m['guard'] + val * dg['guard'], nval=m['nval'] + val, minT=jnp.where(val > 0, jnp.minimum(m['minT'], cs['T']), m['minT']),
                 E_task=jnp.where(val > 0, jnp.maximum(m['E_task'], st_n['w_act']), m['E_task']),
                 mis_hold=m['mis_hold'] + inwin * (j == 1) * mis, mis_n=m['mis_n'] + inwin * (j == 1), w_hat=m['w_hat'] + inwin * (j == 1) * what,
                 viol=m['viol'] + val * jnp.any((st_n['q'] < QLO) | (st_n['q'] > QHI)), upd=m['upd'] + val * dg['zeta'],
                 nan=m['nan'] + (~jnp.all(jnp.isfinite(st_n['q']))), Ub_run=Ub_run,
                 tank_bound=jnp.where(val > 0, jnp.minimum(m['tank_bound'], cs['T'] - cfg['Tmin'] + Ub_run), m['tank_bound']),
                 sat_below=jnp.maximum(m['sat_below'], val * ((cs['T'] < cfg['Tmin']) & (dg['lam'] < 1.0))))
        lg = dict(t=t, p=p, fn=fn, xi=st['xi'], T=cs['T'], alpha=dg['alpha'], rho=dg['rho'], mis=mis, j=j, q=st['q'], lam=dg['lam']) if log_every else None
        return (st_n, cs, prop, m, key), lg
    def outer(carry, kd):
        st, cs, m, key = carry
        key, ko, ki = jax.random.split(key, 3)
        k0 = kd * P.NB
        obs = env.observe(st, par, task, ko)
        led, cs = C.ledger(cs, obs['q'], cfg, LNOM, MNOM, memfn=memfn)
        ob = dict(q=obs['q'], dq=obs['dq'], te=obs['tau_ext'], p=obs['p'], xi=obs['xi'], o=obs['o'], j=env.phase(k0, task), T=led['T'], lam=led['lam'], uload=led['uload'])
        if obs_qd: ob.update(qd=cs['qd'], K=cs['K'])
        qdp, Kp, psi = bfn(ob, ctx, cfg, opts)
        prop = dict(qdp=qdp, Kp=Kp * jnp.ones(3))
        st = env.gripper_event(st, psi, par, task)
        (st, cs, _, m, _), lg = jax.lax.scan(inner, (st, cs, prop, m, ki), k0 + jnp.arange(P.NB))
        if log_every: lg = jax.tree_util.tree_map(lambda a: a[::log_every], lg)
        return (st, cs, m, key), lg
    (st, cs, m, _), lg = jax.lax.scan(outer, (st0, cs0, m0, key), jnp.arange(nd))
    return st, cs, m, lg

def summarize(st, m):
    """Per-episode metrics as NumPy arrays."""
    f = lambda a: np.asarray(a)
    pos = f(m['pos_err']) / np.maximum(f(m['pos_n']), 1)
    return dict(cost=f(m['cost']), cost_ph=f(m['cost_ph']), pos2=pos[:, 1], pos4=pos[:, 3], ferr=f(m['f_err']) / np.maximum(f(m['f_n']), 1),
                frel=f(m['f_rel']) / np.maximum(f(m['f_n']), 1), grasp=f(m['grasp_end1']), t_grasp=f(m['t_grasp']), peak_max=f(m['peak_max']),
                peak_first=f(m['peak_first']), t_first=f(m['t_first']), n_onset=f(m['n_onset']), gate=f(m['gate']) / f(m['nval']), cons=f(m['cons']) / f(m['nval']),
                sat=f(m['sat']) / f(m['nval']), guard=f(m['guard']) / f(m['nval']), minT=f(m['minT']), E_task=f(m['E_task']),
                mis_hold=f(m['mis_hold']) / np.maximum(f(m['mis_n']), 1), w_hat=f(m['w_hat']) / np.maximum(f(m['mis_n']), 1), viol=f(m['viol']), nan=f(m['nan']),
                xi_end=f(st['xi']), n_grasp=f(st['n_grasp']), n_release=f(st['n_release']), n_over=f(st['n_over']),
                fbar=f(m['fn_sum']) / np.maximum(f(m['f_n']), 1), heavy=f(m['heavy']), Ub_max=f(m['Ub_run']), tank_bound=f(m['tank_bound']), sat_below=f(m['sat_below']),
                class_ok=(np.abs(f(m['fn_sum']) / np.maximum(f(m['f_n']), 1) - f(m['fstar'])) < np.abs(f(m['fn_sum']) / np.maximum(f(m['f_n']), 1) - f(m['fother']))).astype(float))

def batch(eps, key, cfg, bname, opts, nsub=4, log_every=0, obs_qd=False):
    nd = int(max(int(np.sum(e[1]['nph'])) for e in eps)) // P.NB
    pb, tb = env.stack(eps)
    cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
    optj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in opts.items()}
    fn = jax.jit(jax.vmap(lambda p, t, k: rollout(p, t, k, nd, cfgj, bname, optj, nsub, log_every, obs_qd)))
    st, cs, m, lg = fn(pb, tb, jax.random.split(key, len(eps)))
    return summarize(st, m), lg

def episodes(seed, n, lim=0.8, **kw):
    rng = np.random.default_rng(seed)
    return [prep(*env.sample_episode(rng, **kw), lim=lim) for _ in range(n)]

def budget_E0(eps, E_task):
    """E0 = Delta T + rho E_task in budget episodes (rho drawn by the sampler), E_nom otherwise."""
    return [P.E_NOM if t['budget'] < 0.5 else C.default_cfg()['dT'] + float(t['rho']) * float(e) for (p, t), e in zip(eps, E_task)]

def with_E0(eps, E0):
    out = []
    for (par, task), e in zip(eps, np.broadcast_to(np.asarray(E0, float), (len(eps),))):
        t = dict(task); t['E0'] = np.float64(e); out.append((par, t))
    return out
