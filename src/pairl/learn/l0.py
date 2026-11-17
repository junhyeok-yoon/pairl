"""L0 rollouts: Stage B environment + v3 controller + scripted test brains (test harness only; the brains may use nominal kinematics)."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, env, params as P, controller as C
from pairl.stageb.kin_np import best_margin

LNOM, MNOM = jnp.asarray(P.L_NOM), jnp.asarray(P.M_NOM)

def prep(par, task):
    task = dict(task)
    task['qo'] = best_margin(task['po'], P.L_NOM, P.QC, P.QH)[1]; task['qs'] = best_margin(task['ps'], P.L_NOM, P.QC, P.QH)[1]
    return par, task

def jac_nom(q): return Mo.jac(q, LNOM)
def pinv(J): return J.T @ jnp.linalg.inv(J @ J.T)
def tip_force(q, te):
    J = jac_nom(q); return jnp.linalg.solve(J @ J.T, J @ te)          # force on the tip from the sensed joint torque

# scripted brains: (bs, obs, task, k, cfg, rng) -> (prop, psi, bs)
def brain_nominal(bs, obs, task, k, cfg, key, opts):
    j = env.phase(k, task); qm, te = obs['q'], obs['tau_ext']
    J = jac_nom(qm); contact = jnp.linalg.norm(te) > cfg['tau_th']; f = tip_force(qm, te)
    started = bs['press_on'] > 0.5
    qp = jnp.where((j == 2) & ~started, task['qs'], bs['qp'])
    down = qp + pinv(J) @ jnp.stack([0.0 * qm[0], -opts['v_d'] * P.TH])
    force = qp - (jnp.asarray(1.0) / bs['K']) * (J.T @ jnp.stack([0.0 * qm[0], opts['g_f'] * (task['fstar'] - f[1])])) + pinv(J) @ jnp.stack([0.5 * (task['ps'][0] - obs['p'][0]), 0.0 * qm[0]])
    qp_new = jnp.where(j == 2, jnp.where(contact | (bs['touched'] > 0.5), force, down), qp)
    if opts.get('adversarial', False):
        qp_new = jnp.where((j == 2) & (contact | (bs['touched'] > 0.5)), qp + pinv(J) @ jnp.stack([0.0 * qm[0], -0.025]), qp_new)
    qp_new = jnp.clip(qp_new, jnp.asarray(P.QC - P.QH), jnp.asarray(P.QC + P.QH))
    qdp = jnp.stack([task['qo'], task['qs'], qp_new, task['qs']])[j]
    Kp = jnp.where((j == 2) & opts.get('adversarial', False), jnp.asarray(cfg['khi']), jnp.asarray(cfg['Kn']))
    psi = jnp.where(j <= 2, 1.0, 0.0)
    touched = jnp.where(j == 2, jnp.maximum(bs['touched'], contact.astype(qm.dtype)), 0.0)
    bs = dict(bs, qp=qp_new, press_on=jnp.where(j == 2, 1.0, 0.0), touched=touched, K=jnp.where(j == 2, Kp, bs['K']))
    return dict(qdp=qdp, Kp=Kp), psi, bs

def brain_calib(bs, obs, task, k, cfg, key, opts):
    """Go to the grasp posture, grasp, hold opts['hold'] s after the grasp, then go to the target posture and hold."""
    t = k * P.TS; tg = bs['t_grasp']
    to_B = (tg >= 0) & (t >= tg + opts['hold'])
    qdp = jnp.where(to_B, task['qs'], task['qo'])
    return dict(qdp=qdp, Kp=jnp.asarray(opts['K'])), jnp.asarray(1.0), bs

def brain_box(bs, obs, task, k, cfg, key, opts):
    """Goals alternate between two opposite corners of the joint box every 0.5 s at the highest stiffness."""
    corner = jnp.asarray(task['corner']); s = jnp.where((k // int(0.5 / P.TS)) % 2 == 0, 1.0, -1.0)
    return dict(qdp=jnp.asarray(P.QC) + s * corner * jnp.asarray(P.QH), Kp=jnp.asarray(cfg['khi'])), jnp.asarray(0.0), bs

def brain_hold(bs, obs, task, k, cfg, key, opts):
    return dict(qdp=task['qs'], Kp=jnp.asarray(opts['K'])), jnp.asarray(0.0), bs

def brain_random(bs, obs, task, k, cfg, key, opts):
    """Nominal script with random goal offsets, random stiffness and random gripper toggles (energy-identity test)."""
    prop, psi, bs = brain_nominal(bs, obs, task, k, cfg, key, opts)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    off = jnp.where(jax.random.uniform(k1) < 0.1, jax.random.uniform(k2, (3,), minval=-0.15, maxval=0.15), bs['off'])
    Kr = jnp.where(jax.random.uniform(k3) < 0.1, jax.random.uniform(k4, (3,), minval=jnp.asarray(cfg['klo']), maxval=jnp.asarray(cfg['khi'])), bs['Kr'])
    psi = jnp.where(jax.random.uniform(k5) < 0.05, 1.0 - psi, psi)
    qdp = jnp.clip(prop['qdp'] + off, jnp.asarray(P.QC - P.QH), jnp.asarray(P.QC + P.QH))
    return dict(qdp=qdp, Kp=Kr), psi, dict(bs, off=off, Kr=Kr)

BRAINS = dict(nominal=brain_nominal, calib=brain_calib, box=brain_box, hold=brain_hold, random=brain_random)

# rollout with in-scan metrics
def rollout(par, task, key, nsteps, cfg, brain, opts, nsub=4, log_every=0):
    bfn = BRAINS[brain]
    qc, qh = jnp.asarray(P.QC), jnp.asarray(P.QH)
    def wq_rho(q): return jnp.sum(jnp.abs((q - qc) / qh) ** P.PG) ** (1.0 / P.PG)
    st0 = env.init_state(par, task); E00 = env.e_rob(st0, par)
    cs0 = C.init_ctrl(task['q0'], task['E0'], cfg, 7)
    bs0 = dict(qp=task['qs'], press_on=jnp.asarray(0.0), touched=jnp.asarray(0.0), K=jnp.asarray(cfg['Kn']), off=jnp.zeros(3), Kr=jnp.asarray(cfg['Kn']), t_grasp=jnp.asarray(-1.0))
    bs0['prop'] = dict(qdp=task['q0'], Kp=jnp.asarray(cfg['Kn']) * jnp.ones(3))
    big = jnp.asarray(1e30)
    m0 = dict(Tq_err=0.0, tank_bound_min=big, Ub_run=0.0, sat_below=0.0, barrier_cum=0.0, barrier_cum_max=-big, viol=0.0, rho_w=0.0, rho_wq=0.0,
              m_grasp=-1.0, m_1s=-1.0, m_29=-1.0, m_A_hold=-1.0, m_B_first=-1.0, m_B_prior=-1.0, vmax_hold=0.0, t_contact=-1.0, m_c0=-1.0, m_c1=-1.0,
              press_tc=0.0, press_mis=0.0, press_n=0.0, peak_impact=0.0, upd_fast=0.0, alpha_min=1.0, lam_min=1.0, gate_steps=0.0, cons_steps=0.0,
              hold_err=0.0, hold_n=0.0, reach_B=-1.0, minT=big, tv_tau=0.0, tau_abs_end=0.0, left_A=0.0, upd_transit=0.0)
    m0 = {k_: jnp.asarray(v) for k_, v in m0.items()}
    def step(carry, k):
        st, cs, bs, m, key = carry
        key, ko, kb, ko2 = jax.random.split(key, 4)
        dec = (k % P.NB) == 0
        obs = env.observe(st, par, task, ko)
        # brain at decision times (ledger available to brains that use it)
        prop_new, psi, bs_new = bfn(bs, obs, task, k, cfg, kb, opts)
        prop_new = dict(qdp=prop_new['qdp'], Kp=prop_new['Kp'] * jnp.ones(3))
        prop = jax.tree_util.tree_map(lambda a, b: jnp.where(dec, a, b), prop_new, bs['prop'])
        bs = jax.tree_util.tree_map(lambda a, b: jnp.where(dec, a, b), bs_new, bs); bs = dict(bs, prop=prop)
        xi_before = st['xi']
        st_ev = env.gripper_event(st, psi, par, task); st = jax.tree_util.tree_map(lambda a, b: jnp.where(dec, a, b), st_ev, st)
        grasped_now = (st['xi'] > 0.5) & (xi_before < 0.5)
        bs = dict(bs, t_grasp=jnp.where(grasped_now & (bs['t_grasp'] < 0), k * P.TS, bs['t_grasp']))
        obs = env.observe(st, par, task, ko2)
        extra = None
        if opts.get('extra', 0.0) > 0:
            extra = opts['extra'] * jnp.asarray(P.UB) * jnp.sign(prop['qdp'] - obs['q'])
        if opts.get('reset_z_at_contact', False):
            first_c = (m['t_contact'] < 0) & (jnp.linalg.norm(obs['tau_ext']) > cfg['tau_th']) & (env.phase(k, task) == 2)
            cs = dict(cs, z=jnp.where(first_c, 0.0, cs['z']))
        cs_prev_tau = cs['tau_prev']
        tau, cs, dg = C.ctrl_step(cs, obs, prop, cfg, LNOM, MNOM, extra)
        # true quantities for the metrics
        pe_ = env.pe(par, st); ustar = Mo.gravity(st['q'], pe_, st['xi']) - par['d']; mis = jnp.linalg.norm(dg['uad'] - ustar)
        fn, p, v = env.top_force(st, par); fe, _ = Mo.contact(p, v, par); tc = jnp.linalg.norm(Mo.jac(st['q'], par['L']).T @ fe)
        st_next = env.control_step(st, tau, par, nsub)
        Ub0, Ub1 = C.barrier_potential(st['q'], cfg), C.barrier_potential(st_next['q'], cfg)
        bstep = dg['tau_b'] @ (st_next['q'] - st['q']) + Ub1 - Ub0
        Ttrue = task['E0'] + cfg['Tmin'] - st_next['w_act']
        Ub_run = jnp.maximum(m['Ub_run'], Ub1)
        j = env.phase(k, task); t = k * P.TS
        speed = jnp.max(jnp.abs(st['dq']))
        # calibration bookkeeping (brain_calib)
        tg = bs['t_grasp']; in_hold = (tg >= 0) & (t >= tg) & (t < tg + opts.get('hold', 1e9))
        m_new = dict(m,
            Tq_err=jnp.maximum(m['Tq_err'], jnp.abs(cs['T'] - (task['E0'] + cfg['Tmin'] - st['w_act']))),
            Ub_run=Ub_run, tank_bound_min=jnp.minimum(m['tank_bound_min'], cs['T'] - cfg['Tmin'] + Ub_run),
            sat_below=jnp.maximum(m['sat_below'], ((cs['T'] < cfg['Tmin']) & (dg['lam'] < 1.0)).astype(Ub1.dtype)),
            barrier_cum=m['barrier_cum'] + bstep, barrier_cum_max=jnp.maximum(m['barrier_cum_max'], m['barrier_cum'] + bstep),
            viol=m['viol'] + jnp.any((st_next['q'] < qc - qh) | (st_next['q'] > qc + qh)).astype(Ub1.dtype),
            rho_w=jnp.maximum(m['rho_w'], wq_rho(st['q'] + st['dq'] / cfg['k1'])), rho_wq=jnp.maximum(m['rho_wq'], wq_rho(st['q'])),
            m_grasp=jnp.where(grasped_now, mis, m['m_grasp']),
            m_1s=jnp.where((tg >= 0) & (jnp.abs(t - tg - 1.0) < 0.5 * P.TS), mis, m['m_1s']),
            m_29=jnp.where((tg >= 0) & (jnp.abs(t - tg - 2.9) < 0.5 * P.TS), mis, m['m_29']),
            m_A_hold=jnp.where((tg >= 0) & (jnp.abs(t - tg - opts.get('hold', 1e9) + P.TS) < 0.5 * P.TS), mis, m['m_A_hold']),
            vmax_hold=jnp.where(in_hold & (t > tg + 0.2), jnp.maximum(m['vmax_hold'], speed), m['vmax_hold']),
            left_A=jnp.where((tg >= 0) & (t >= tg + opts.get('hold', 1e9)) & (speed > 0.3), 1.0, m['left_A']),
            upd_transit=m['upd_transit'] + ((m['left_A'] > 0.5) & (m['reach_B'] < 0) & (dg['zeta'] > 0.5)).astype(Ub1.dtype),
            reach_B=jnp.where((m['reach_B'] < 0) & (m['left_A'] > 0.5) & (dg['zeta'] > 0.5), t, m['reach_B']),
            m_B_first=jnp.where((m['reach_B'] < 0) & (m['left_A'] > 0.5) & (dg['zeta'] > 0.5), mis, m['m_B_first']),
            t_contact=jnp.where((m['t_contact'] < 0) & (j == 2) & (dg['contact'] > 0.5), t, m['t_contact']),
            m_c0=jnp.where((m['t_contact'] < 0) & (j == 2) & (dg['contact'] > 0.5), mis, m['m_c0']),
            m_c1=jnp.where((m['t_contact'] >= 0) & (jnp.abs(t - m['t_contact'] - 1.0) < 0.5 * P.TS), mis, m['m_c1']),
            peak_impact=jnp.where((m['t_contact'] >= 0) & (t <= m['t_contact'] + 0.3), jnp.maximum(m['peak_impact'], tc), m['peak_impact']),
            upd_fast=m['upd_fast'] + ((dg['zeta'] > 0.5) & (speed > 0.35)).astype(Ub1.dtype),
            alpha_min=jnp.minimum(m['alpha_min'], dg['alpha']), lam_min=jnp.minimum(m['lam_min'], dg['lam']),
            gate_steps=m['gate_steps'] + (dg['alpha'] < 1.0).astype(Ub1.dtype), cons_steps=m['cons_steps'] + (jnp.linalg.norm(dg['tau_b']) > 0).astype(Ub1.dtype))
        # press-phase steady window: last 0.5 s of phase 2
        edges = jnp.cumsum(task['nph']); in_win = (j == 2) & (k >= edges[2] - int(0.5 / P.TS))
        m_new = dict(m_new, press_tc=m_new['press_tc'] + jnp.where(in_win, tc, 0.0), press_mis=m_new['press_mis'] + jnp.where(in_win, mis, 0.0), press_n=m_new['press_n'] + in_win.astype(Ub1.dtype))
        hold_win = (k >= nsteps - int(0.5 / P.TS))
        m_new = dict(m_new, hold_err=m_new['hold_err'] + jnp.where(hold_win, jnp.linalg.norm(p - task['ps']), 0.0), hold_n=m_new['hold_n'] + hold_win.astype(Ub1.dtype))
        m_new = dict(m_new, minT=jnp.minimum(m['minT'], cs['T']), tv_tau=m['tv_tau'] + jnp.sum(jnp.abs(tau - cs_prev_tau)), tau_abs_end=jnp.sum(jnp.abs(tau)))
        if log_every:
            lg = dict(q=st['q'], p=p, fn=fn, xi=st['xi'], T=cs['T'], mis=mis, zeta=dg['zeta'], alpha=dg['alpha'], lam=dg['lam'], rho=dg['rho'], tc=tc, ph=j)
        else:
            lg = None
        return (st_next, cs, bs, m_new, key), lg
    (st, cs, bs, m, _), lg = jax.lax.scan(step, (st0, cs0, bs0, m0, key), jnp.arange(nsteps))
    # energy identity (true tank) for completeness
    return st, cs, m, lg

def batch(eps, key, nsteps, cfg, brain, opts, nsub=4, log_every=0):
    pb, tb = env.stack(eps); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
    f = jax.jit(jax.vmap(lambda p, t, k: rollout(p, t, k, nsteps, cfgj, brain, opts, nsub, log_every)))
    return f(pb, tb, jax.random.split(key, len(eps)))
