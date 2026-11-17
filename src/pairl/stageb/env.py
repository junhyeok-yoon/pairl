"""Stage B environment: episode sampler (numpy), control step, gripper events, sensing, cost (JAX, one episode; vmap for batches)."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, params as P
from pairl.stageb.kin_np import best_margin

# episode sampler (numpy, float64)
def sample_episode(rng, ood_body=False, ood_stiff=False, hard=None, budget=None, v=2):
    """One episode. v=2: two force targets by payload class and the budget factor rho (drawn after all v=1 draws); v=1: task of L0 and the L1 pilot."""
    par = P.make_par(rng, ood=ood_body, v=v)
    hard = (rng.random() < P.P_HARD) if hard is None else hard
    budget = (rng.random() < P.P_BUDGET) if budget is None else budget
    unif = lambda: np.array([rng.uniform(*P.BOX[0]), rng.uniform(*P.BOX[1])])
    mgn = lambda pt: best_margin(pt, par['L'], P.QC, P.QH)[0]
    for _ in range(100000):
        ps = unif(); drop = rng.uniform(0.05, 0.15); ye = ps[1] - drop; mg = mgn(ps)
        if not (((P.MARG_HARD[0] <= mg < P.MARG_HARD[1]) if hard else (mg >= P.MARG_NORMAL)) and mgn(np.array([ps[0], ye])) >= P.MARG_HARD[0]): continue
        po = None
        for _ in range(200):
            c = unif()
            if abs(c[0] - ps[0]) >= 0.2 and c[1] >= ye + P.CLEAR and mgn(c) >= P.MARG_NORMAL: po = c; break
        if po is not None: break
    else: raise RuntimeError('target sampling failed')
    par.update(xe=np.float64(ps[0]), ye=np.float64(ye), we=np.float64(P.WE),
               ke=np.float64(np.exp(rng.uniform(np.log(2e4), np.log(2e5))) if ood_stiff else np.exp(rng.uniform(np.log(2e3), np.log(2e4)))),
               de=np.float64(rng.uniform(0.5, 5.0)), nu=np.float64(rng.uniform(0.2, 0.8)))
    for _ in range(100000):
        q0 = P.QC + 0.6 * P.QH * rng.uniform(-1, 1, 3); th = np.cumsum(q0)
        if np.sum(par['L'] * np.sin(th)) >= ye + P.CLEAR: break
    else: raise RuntimeError('initial posture sampling failed')
    dur = np.array([rng.uniform(1.5, 2.5), rng.uniform(2, 3), rng.uniform(2, 3), rng.uniform(1.5, 2.5)])
    nph = np.maximum(1, np.round(dur / P.TH)).astype(np.int64) * P.NB                     # phase lengths in control steps, multiples of NB
    psi0 = rng.uniform(-np.pi, np.pi)                                                     # initial orientation of the object
    c0 = po - par['Lobj'] * np.array([np.cos(psi0), np.sin(psi0)])                        # its center of mass; grasp point = c0 + Lobj e(psi0) = po
    task = dict(po=po, ps=ps, fstar=np.float64(rng.uniform(3, 15)), nph=nph, hard=np.float64(hard), budget=np.float64(budget),
                E0=np.float64(P.E_NOM), bias=rng.uniform(-P.BIAS_TAU, P.BIAS_TAU, 3), q0=q0, c0=c0, psi0=np.float64(psi0))
    heavy = np.float64(par['mo'] > P.M_TH)
    if v == 1:
        task.update(f_light=task['fstar'], f_heavy=task['fstar'], heavy=heavy, rho=np.float64(1.0))
        return par, task
    f1 = task['fstar']                                                                    # pair uniform on {|f1 - f2| >= F_SEP} by rejection
    while True:
        f2 = rng.uniform(*P.F_RANGE)
        if abs(f1 - f2) >= P.F_SEP: break
        f1 = rng.uniform(*P.F_RANGE)
    task.update(f_light=np.float64(f1), f_heavy=np.float64(f2), heavy=heavy, fstar=np.float64(f2 if heavy else f1), rho=np.float64(rng.uniform(*P.RHO_BUDGET)))
    return par, task

def stack(eps):
    """List of (par, task) -> batched dicts of jnp arrays."""
    return ({k: jnp.asarray(np.stack([e[0][k] for e in eps])) for k in eps[0][0]}, {k: jnp.asarray(np.stack([e[1][k] for e in eps])) for k in eps[0][1]})

# state and stepping (JAX)
def init_state(par, task):
    z = jnp.zeros(())
    att = 'xi0' in task                                                                        # attached start of the start-state distribution (restart below)
    return dict(q=task['q0'], dq=jnp.zeros(3), xi=task['xi0'] if att else z, k=jnp.zeros((), jnp.int64 if jax.config.jax_enable_x64 else jnp.int32),
                obj_c=task['c0'], obj_psi=task['psi0'], rx=task['rx0'] if att else par['rx'], ry=task['ry0'] if att else par['ry'],
                odelta=task['odelta0'] if att else z,                                          # object pose (detached) and attachment (attached)
                tauc_prev=jnp.zeros(3), w_act=z, w_fric=z, w_env=z, e_jump=z, n_over=z,
                n_grasp=z, n_release=z, ev_gain_max=z - 1e30, com_err_max=z, r_max=z)           # event diagnostics

def restart(par, task, kind):
    """Start-state distribution of the restart pilot. kind 0: the episode unchanged (attached-start fields at their detached values, for mixed batches);
    2: phase 2 starts at the grasp posture q_o (best-margin inverse kinematics of the grasp point) with the object attached at the tip; 3: phase 3 starts at the
    posture q_s of p* with the object attached as if grasped at q_o. Velocity zero; the phases before the start get length zero; numpy, float64."""
    t = dict(task)
    if kind == 0:
        t.update(xi0=np.float64(0.0), rx0=np.float64(par['rx']), ry0=np.float64(par['ry']), odelta0=np.float64(0.0)); return par, t
    qo = np.asarray(task['qo_b'], np.float64); th = np.cumsum(qo); L = np.asarray(par['L'])
    po = np.array([np.sum(L * np.cos(th)), np.sum(L * np.sin(th))])                           # tip at q_o with the episode's link lengths
    c = po - par['Lobj'] * np.array([np.cos(task['psi0']), np.sin(task['psi0'])])                # object grasped exactly at the tip, initial orientation
    v, t3 = c - po, th[-1]
    nph = np.asarray(task['nph']).copy(); nph[:kind - 1] = 0
    t.update(q0=qo if kind == 2 else np.asarray(task['qs_b'], np.float64), c0=c, xi0=np.float64(1.0), odelta0=np.float64(task['psi0'] - t3), nph=nph,
             rx0=np.float64(np.cos(t3) * v[0] + np.sin(t3) * v[1]), ry0=np.float64(-np.sin(t3) * v[0] + np.cos(t3) * v[1]))
    return par, t

def near_start(par, task, rng):
    """Phase-1 start near the object (pilot G, arm S2): at rest with the tip 2-8 cm from the grasp point, posture by best-margin inverse kinematics inside
    the joint box and above the block top; the object, targets and phase lengths unchanged. Falls back to the episode's own start if no point is found in 20
    draws. numpy, float64; returns (par, task, found)."""
    par, t = restart(par, task, 0); po = np.asarray(task['po']); L = np.asarray(par['L'])
    for _ in range(20):
        r, a = rng.uniform(0.02, 0.08), rng.uniform(-np.pi, np.pi); x = po + r * np.array([np.cos(a), np.sin(a)])
        mg, qn = best_margin(x, L, P.QC, P.QH)
        if mg > 0 and x[1] > par['ye'] + P.CLEAR:
            t = dict(t); t['q0'] = np.asarray(qn, np.float64); return par, t, True
    return par, t, False

def pe(par, st):
    """Plant parameters with the current attachment offset of the object."""
    return dict(par, rx=st['rx'], ry=st['ry'])

def grasp_point(st, par):
    """Grasp point of the object: on the tip while attached, c + Lobj e(psi) while detached."""
    g = st['obj_c'] + par['Lobj'] * jnp.stack([jnp.cos(st['obj_psi']), jnp.sin(st['obj_psi'])])
    return jnp.where(st['xi'] > 0.5, Mo.tip(st['q'], par['L']), g)

def s_obj(st, par):
    """Potential energy of the object while it rests detached (0 while attached: then part of E_rob)."""
    return (1.0 - st['xi']) * par['mo'] * Mo.G0 * st['obj_c'][1]

def phase(k, task):
    """Index 0..3 of the phase containing control step k (3 after the end)."""
    edges = jnp.cumsum(task['nph'])
    return jnp.minimum(jnp.sum(k >= edges), 3)

def _deriv(q, dq, tau, xi, par):
    ddq, fe = Mo.accel(q, dq, tau, xi, par)
    v = Mo.jac(q, par['L']) @ dq
    return ddq, jnp.sum(par['Fv'] * dq * dq), v @ fe

def control_step(st, tau, par, nsub=4):
    """Advance one control period with constant torque (RK4, nsub substeps). Returns the new state."""
    q, dq, wf, we = st['q'], st['dq'], st['w_fric'], st['w_env']
    ub = jnp.asarray(P.UB, q.dtype)
    over = jnp.any(jnp.abs(tau) > ub + 1e-9).astype(q.dtype); tau = jnp.clip(tau, -ub, ub)      # physical actuator limit; exceedances are counted
    par = pe(par, st)
    tauc_now = Mo.jac(q, par['L']).T @ Mo.accel(q, dq, tau, st['xi'], par)[1]
    h = P.TS / nsub; q0 = q
    for _ in range(nsub):
        a1, f1, e1 = _deriv(q, dq, tau, st['xi'], par)
        a2, f2, e2 = _deriv(q + 0.5 * h * dq, dq + 0.5 * h * a1, tau, st['xi'], par)
        a3, f3, e3 = _deriv(q + 0.5 * h * (dq + 0.5 * h * a1), dq + 0.5 * h * a2, tau, st['xi'], par)
        a4, f4, e4 = _deriv(q + h * (dq + 0.5 * h * a2), dq + h * a3, tau, st['xi'], par)
        q, dq = q + h * dq + h * h / 6 * (a1 + a2 + a3), dq + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        wf = wf + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4); we = we + h / 6 * (e1 + 2 * e2 + 2 * e3 + e4)
    return dict(st, q=q, dq=dq, k=st['k'] + 1, tauc_prev=tauc_now, w_act=st['w_act'] + tau @ (q - q0), w_fric=wf, w_env=we, n_over=st['n_over'] + over)

def gripper_event(st, psi, par, task):
    """Attachment and release at a decision time. psi in {0,1}.
    Grasp: the object's center of mass and orientation stay where they are; their offset in the last-link frame is fixed from the current geometry.
    Release: the object rests at its current pose. Re-grasping is allowed."""
    q, dq, xi = st['q'], st['dq'], st['xi']
    p = Mo.tip(q, par['L']); t3 = jnp.sum(q)
    near = jnp.linalg.norm(p - grasp_point(st, par)) <= P.GRASP_DIST
    slow = jnp.max(jnp.abs(dq)) <= P.GRASP_SPEED
    grasp = (xi < 0.5) & (psi > 0.5) & near & slow
    release = (xi > 0.5) & (psi < 0.5)
    r_new = Mo.offset_of(q, st['obj_c'], par['L'])
    par_g = dict(par, rx=r_new[0], ry=r_new[1])
    c_rel = Mo.obj_pos(q, pe(par, st))
    new = dict(st, xi=jnp.where(grasp, 1.0, jnp.where(release, 0.0, xi)),
               dq=jnp.where(grasp, Mo.attach_velocity(q, dq, par_g), dq),
               rx=jnp.where(grasp, r_new[0], st['rx']), ry=jnp.where(grasp, r_new[1], st['ry']),
               odelta=jnp.where(grasp, st['obj_psi'] - t3, st['odelta']),
               obj_c=jnp.where(release, c_rel, st['obj_c']), obj_psi=jnp.where(release, t3 + st['odelta'], st['obj_psi']))
    e0, e1 = e_rob(st, par), e_rob(new, par)
    gain = e1 + s_obj(new, par) - e0 - s_obj(st, par)
    event = grasp | release
    return dict(new, e_jump=st['e_jump'] + (e1 - e0), n_grasp=st['n_grasp'] + grasp, n_release=st['n_release'] + release,
                ev_gain_max=jnp.where(event, jnp.maximum(st['ev_gain_max'], gain), st['ev_gain_max']),
                com_err_max=jnp.where(grasp, jnp.maximum(st['com_err_max'], jnp.linalg.norm(Mo.obj_pos(q, par_g) - st['obj_c'])), st['com_err_max']),
                r_max=jnp.where(grasp, jnp.maximum(st['r_max'], jnp.linalg.norm(r_new)), st['r_max']))

def e_rob(st, par):
    ke, pot = Mo.energy(st['q'], st['dq'], pe(par, st), st['xi'])
    return ke + pot - par['d'] @ st['q']

def top_force(st, par):
    """Normal force on the top face (0 when not in top-face contact) and tip position, velocity."""
    q, dq = st['q'], st['dq']
    p = Mo.tip(q, par['L']); v = Mo.jac(q, par['L']) @ dq
    fe, _ = Mo.contact(p, v, par)
    return fe[1], p, v

def observe(st, par, task, key, noise=1.0):
    """Measured signals at the current control instant."""
    k1, k2, k3 = jax.random.split(key, 3)
    n = noise * jnp.clip(jax.random.normal(k3, (3,)), -P.NOISE_CLIP, P.NOISE_CLIP) * P.SIG_TAU
    return dict(q=jnp.round(st['q'] / P.Q_QUANT) * P.Q_QUANT, dq=st['dq'] + noise * P.SIG_DQ * jax.random.normal(k1, (3,)),
                tau_ext=st['tauc_prev'] + task['bias'] + n, p=Mo.tip(st['q'], par['L']) + noise * P.SIG_P * jax.random.normal(k2, (2,)), xi=st['xi'],
                o=grasp_point(st, par))                                                            # context: grasp point of the object (tip while attached)

SPEED_KAPPA, SPEED_R, SPEED_CAP = 1.0 / 6.0, 0.05, 3.0                                      # rev. 17 (COST_V 5): records/l2_grasp_design.md
COST_V = 4                                                                                     # 4: spec rev. 15 (default); 3: rev. 14 (L2 attempt 2); 1: rev. 13 (L2 attempt 1); 2: pilots P1, P2. Read at tracing time

def cost_rate(st, par, task):
    """Cost rate of the task: go to the object, carry it, press with it, release it and retreat.
    Rev. 15 (COST_V 4): as rev. 14 with the press term bounded, min(((f - f*)/f*)^2, 1) under top-face force, so that holding stays cheaper than dropping at every force.
    Rev. 14 (COST_V 3): in phases 2 and 3 the rate is the progress cost of the object at its pose (the tip while held, zero force while not held) plus
    (1 - xi)(1 + |p - o|/0.1); without top-face force the press term is 1 + |o - p_c|/0.1 with p_c the center of the top face.
    COST_V 2: as 3 with the press term 1 + max(0, o_y - y_e)/0.1. COST_V 1 (rev. 13): the object's progress cost is absent while it is not held."""
    fn, p, _ = top_force(st, par); j = phase(st['k'], task); xi = st['xi']; ps, fs = task['ps'], task['fstar']
    o = grasp_point(st, par); to_obj = jnp.linalg.norm(p - o) / 0.1                            # 0 while attached
    pc = jnp.stack([ps[0], par['ye']])                                                         # center of the top face
    near = (lambda y: jnp.linalg.norm(y - pc) / 0.1) if COST_V >= 3 else (lambda y: jnp.maximum(y[1] - par['ye'], 0.0) / 0.1)
    sq = ((fn - fs) / fs) ** 2
    press = jnp.where(fn > 0, jnp.minimum(sq, 1.0) if COST_V >= 4 else sq, 1.0 + near(p))
    if COST_V >= 2: d2, d3 = jnp.linalg.norm(o - ps) / 0.1, 1.0 + near(o) + jnp.abs(o[0] - ps[0]) / 0.1
    else: d2 = d3 = 0.0
    if COST_V >= 5:                                                                            # rev. 17: phase-1 speed term near the object
        vex = jnp.minimum(jnp.maximum(jnp.max(jnp.abs(st['dq'])) - P.GRASP_SPEED, 0.0) / P.GRASP_SPEED, SPEED_CAP)
        to_obj1 = to_obj + (1 - xi) * SPEED_KAPPA * jnp.maximum(0.0, 1.0 - to_obj * 0.1 / SPEED_R) * vex
    else: to_obj1 = to_obj
    om = jnp.stack([to_obj1,
                    xi * jnp.linalg.norm(p - ps) / 0.1 + (1 - xi) * (1.0 + to_obj + d2),
                    xi * (press + jnp.abs(p[0] - ps[0]) / 0.1) + (1 - xi) * (1.0 + to_obj + d3),
                    jnp.linalg.norm(p - ps) / 0.1 + xi])[j]
    return om + 5.0 * jnp.maximum(fn - 2 * fs, 0.0) / fs

def u_star(st, par):
    """Privileged compensation torque g(q) - d (with the attached object)."""
    return Mo.gravity(st['q'], pe(par, st), st['xi']) - par['d']
