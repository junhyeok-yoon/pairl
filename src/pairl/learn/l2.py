"""L2: coupled learning on Stage B (Definition coupled learning, spec rev. 13).
Brain: memoryless MLP policy of o_k = (q, dq, tau_ext, context, ledger), trained by PPO on the task return (exploration on the brain's action only).
Hand: learned memory features G_W(q) (7 x 3, entries in [-1, 1]) and prior a_bar, u_ad = G_W^T (a_bar + z); trained on the role loss
mean |u_ad - u*|^2, replayed on the recorded rollouts (c_r = 0). Tank, gate, guard, constraint layer and saturation are fixed."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, env, params as P, controller as C
from pairl.learn import l1

LNOM, MNOM, QLO, QHI = l1.LNOM, l1.MNOM, l1.QLO, l1.QHI
QC, QH = jnp.asarray(P.QC), jnp.asarray(P.QH)
NA, OBS_DIM, PRIV_DIM, ACT_DIM = 7, 31, 6, 6
OBS_DIM_V1 = 27                            # observation of the first phase of attempt 1 (without the relative positions)
LS_LO = jnp.log(jnp.asarray([0.005] * 3 + [0.05] * 3)); LS_HI = jnp.log(jnp.asarray([0.3] * 3 + [0.5] * 3))
ND = 220                                   # decisions per episode slot (longest episode: 11 s)
ABAR_MAX = 50.0

# networks and optimizer
def mlp_init(key, sizes, out_scale=1.0):
    ks = jax.random.split(key, len(sizes) - 1)
    return [dict(W=(out_scale if i == len(sizes) - 2 else 1.0) / np.sqrt(a) * jax.random.normal(k, (a, b)), b=jnp.zeros(b))
            for i, (k, a, b) in enumerate(zip(ks, sizes[:-1], sizes[1:]))]

def mlp(ps, x):
    for l in ps[:-1]: x = jnp.tanh(x @ l['W'] + l['b'])
    return x @ ps[-1]['W'] + ps[-1]['b']

def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    actor = mlp_init(k1, [OBS_DIM, 128, 128, 2 * ACT_DIM + 1], 0.01)
    actor[-1]['b'] = jnp.concatenate([jnp.asarray([0.0, 0.0, 0.0, np.log(2.0), np.log(2.0), np.log(2.0)]), std_raw(jnp.asarray([0.04] * 3 + [0.3] * 3)), jnp.asarray([6.0])])
    bp = dict(actor=actor, critic=mlp_init(k2, [OBS_DIM + PRIV_DIM, 128, 128, 1], 1.0))
    hp = dict(G=mlp_init(k3, [6, 64, 64, NA * 3], 1.0), abar=jnp.zeros(NA))
    return bp, hp

def std_raw(sd):
    """Pre-sigmoid value giving the standard deviation sd within [exp(LS_LO), exp(LS_HI)]."""
    a = (jnp.log(sd) - LS_LO) / (LS_HI - LS_LO); return jnp.log(a / (1 - a))

def upgrade_v1(bp1):
    """Brain of the first phase (27 inputs, state-independent standard deviation) -> state-dependent standard deviation and 31 inputs, same function:
    zero weights for the new inputs and for the new standard-deviation outputs, whose biases reproduce the old standard deviations."""
    a = [dict(l) for l in bp1['actor']]; c = [dict(l) for l in bp1['critic']]
    a[0]['W'] = jnp.concatenate([a[0]['W'], jnp.zeros((OBS_DIM - OBS_DIM_V1, a[0]['W'].shape[1]))])
    c[0]['W'] = jnp.concatenate([c[0]['W'][:OBS_DIM_V1], jnp.zeros((OBS_DIM - OBS_DIM_V1, c[0]['W'].shape[1])), c[0]['W'][OBS_DIM_V1:]])   # critic input: observation, then privileged entries
    W, b = a[-1]['W'], a[-1]['b']; sd = jnp.exp(bp1['log_std'])
    a[-1] = dict(W=jnp.concatenate([W[:, :ACT_DIM], jnp.zeros((W.shape[0], ACT_DIM)), W[:, ACT_DIM:]], 1), b=jnp.concatenate([b[:ACT_DIM], std_raw(sd), b[ACT_DIM:]]))
    return dict(actor=a, critic=c)

def adam_init(p): return dict(m=jax.tree_util.tree_map(jnp.zeros_like, p), v=jax.tree_util.tree_map(jnp.zeros_like, p), t=jnp.zeros(()))
def adam_step(p, g, s, lr, clip=None, b1=0.9, b2=0.999, eps=1e-8):
    if clip is not None:
        gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))); g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, clip / (gn + 1e-12)), g)
    t = s['t'] + 1
    m = jax.tree_util.tree_map(lambda m, x: b1 * m + (1 - b1) * x, s['m'], g); v = jax.tree_util.tree_map(lambda v, x: b2 * v + (1 - b2) * x ** 2, s['v'], g)
    p = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), p, m, v)
    return p, dict(m=m, v=v, t=t)

# hand: learned memory
def hand_G(hp, q):
    return jnp.tanh(mlp(hp['G'], jnp.concatenate([jnp.sin(q), jnp.cos(q)]))).reshape(NA, 3)
def abar_of(hp): return ABAR_MAX * jnp.tanh(hp['abar'])          # raw parameter of order one
def memfn_of(hp):
    def f(q, z):
        G = hand_G(hp, q); return G.T @ (abar_of(hp) + z), G
    return f

# brain: observation, policy, action map
KLO, KHI = jnp.asarray(C.default_cfg()['klo']), jnp.asarray(C.default_cfg()['khi'])
CEN = jnp.asarray([0.55, 0.1])
def obs_vec(ob, task, cfg):
    return jnp.concatenate([(ob['q'] - QC) / QH, ob['dq'] / 2, ob['te'] / 10, jax.nn.one_hot(ob['j'], 4), (ob['o'] - CEN) / 0.3, (task['ps'] - CEN) / 0.3,
                            (jnp.stack([task['f_light'], task['f_heavy']]) - 9) / 6, (ob['p'] - CEN) / 0.3, ob['xi'][None],
                            (jnp.log1p(jnp.maximum(ob['T'] - cfg['Tmin'], 0.0)) / 4)[None], ((1 - ob['lam']) * 5)[None], ob['uload'] / 10,
                            jnp.clip((ob['o'] - ob['p']) / 0.05, -5, 5), jnp.clip((task['ps'] - ob['p']) / 0.05, -5, 5)]
                           + ([l1.tip_force(ob['q'], ob['uload']) / 10] if OBS_TIP else []))   # relative positions: functions of the entries above; tip load (arm B)
def priv_vec(task, par, k0, edges, j):
    t_rem = (edges[j] - k0) * P.TS
    return jnp.stack([t_rem / 3, par['mo'] / 2, (task['fstar'] - 9) / 6, (task['ps'][1] - par['ye']) / 0.1, task['heavy'], task['budget']])
def policy(bp, x):
    out = mlp(bp['actor'], x)
    return out[:ACT_DIM], LS_LO + (LS_HI - LS_LO) * jax.nn.sigmoid(out[ACT_DIM:2 * ACT_DIM]), out[2 * ACT_DIM]
def value(bp, xc):
    """Critic in return units: the network predicts the return divided by the scale bp['vscale'] (1 if absent), which is set from data and not trained."""
    return jax.lax.stop_gradient(bp.get('vscale', 1.0)) * mlp(bp['critic'], xc)[0]
ACT_MODE = 'joint'                         # 'joint': absolute goal posture in the joint box; 'task' (pilot T): box projection of q + J_nom(q)^+ dp + n v_N. Read at tracing time
DP_MAX, N_MAX, DLS = 0.3, 0.3, 0.02        # task mode: largest tip displacement [m], largest null-space step [rad], damping of the pseudo-inverse [m]
OBS_TIP = False                                     # pilot G arm B: ledger load as a tip load (nominal kinematics) appended to the observation
def set_mode(ck):
    """Action mode of a checkpoint dict (key 'act'; checkpoints without it are joint mode) and its observation layout (key 'obs': 'tipload' or absent)."""
    global ACT_MODE, OBS_TIP; ACT_MODE = ck.get('act', 'joint'); OBS_TIP = ck.get('obs') == 'tipload'
def add_tip_inputs(bp):
    """Two zero-initialized inputs at the end of the observation (actor first layer) and before the privileged entries (critic first layer)."""
    a, c = list(bp['actor']), list(bp['critic']); z = lambda n: jnp.zeros((2, n))
    a[0] = dict(a[0], W=jnp.concatenate([a[0]['W'], z(a[0]['W'].shape[1])]))
    c[0] = dict(c[0], W=jnp.concatenate([c[0]['W'][:OBS_DIM], z(c[0]['W'].shape[1]), c[0]['W'][OBS_DIM:]]))
    return dict(bp, actor=a, critic=c)
def goal_of(u, q):
    """Goal posture of the brain's action at the measured posture q. Task mode: the tip displacement dp = DP_MAX tanh(u_1:2) through the damped pseudo-inverse of
    the nominal Jacobian (class-level kinematics) plus a null-space step N_MAX tanh(u_3) along the unit kernel direction, projected onto the joint box."""
    if ACT_MODE == 'task':
        J = Mo.jac(q, LNOM); Jp = J.T @ jnp.linalg.inv(J @ J.T + DLS ** 2 * jnp.eye(2))
        vn = jnp.cross(J[0], J[1]); vn = vn / (jnp.linalg.norm(vn) + 1e-12)
        return jnp.clip(q + Jp @ (DP_MAX * jnp.tanh(u[:2])) + N_MAX * jnp.tanh(u[2]) * vn, QC - QH, QC + QH)
    return QC + QH * jnp.tanh(u[:3])
def act_map(u, q):
    """Goal posture (goal_of) and stiffness on a log scale between k_lo and k_hi."""
    return goal_of(u, q), KLO * (KHI / KLO) ** jax.nn.sigmoid(u[3:6])
def logp_of(mu, log_std, logit, u, psi):
    lg = -0.5 * jnp.sum(((u - mu) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    return lg + psi * jax.nn.log_sigmoid(logit) + (1 - psi) * jax.nn.log_sigmoid(-logit)
GRIP_EPS = 0.0                             # attempt 4 arm I: training gripper sampled from p' = (1 - eps) sigmoid(logit) + eps / 2 (mode 0, phase 4 only; amendment 1). Read at tracing time
PH4 = 12                                   # observation entry of the phase-4 indicator (one-hot phase at entries 9-12)
def grip_prob(logit, eps):
    """Probability of closing (psi = 1) under the gripper exploration mixture."""
    return (1 - eps) * jax.nn.sigmoid(logit) + 0.5 * eps
def logp_mix(mu, log_std, logit, u, psi, eps):
    """logp_of with the gripper term taken under grip_prob(logit, eps)."""
    lg = -0.5 * jnp.sum(((u - mu) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)); pp = grip_prob(logit, eps)
    return lg + psi * jnp.log(pp) + (1 - psi) * jnp.log(1 - pp)
def actor_hidden(bp, x):
    """Last hidden layer of the actor (input of its output layer)."""
    for l in bp['actor'][:-1]: x = jnp.tanh(x @ l['W'] + l['b'])
    return x
def add_cls_head(bp):
    """Attempt 4 arm I: zero-initialized linear head (one logit) on the actor's last hidden layer; not used by policy()."""
    n = bp['actor'][-1]['W'].shape[0]; return dict(bp, cls=dict(W=jnp.zeros((n, 1)), b=jnp.zeros(1)))

# rollout: mode 0 sample, 1 deterministic, 2 deterministic motion with the gripper sampled (evaluation); blind 1: the brain sees u_load = 0 (the hand is unchanged)
HOLD = 10                                  # exploration noise of the goal and stiffness held for 10 decisions (0.5 s)
def rollout(par, task, key, cfg, bp, hp, mode, blind=0.0, nsub=4):
    memfn = memfn_of(hp)
    edges = jnp.cumsum(task['nph']); nwin = jnp.round(0.4 * task['nph']).astype(edges.dtype)
    st0 = env.init_state(par, task); cs0 = C.init_ctrl(task['q0'], task['E0'], cfg, NA)
    big, z0 = jnp.asarray(1e30), jnp.asarray(0.0)
    m0 = dict(cost=z0, cost_ph=jnp.zeros(4), pos_err=jnp.zeros(4), pos_n=jnp.zeros(4), f_rel=z0, f_n=z0, fn_sum=z0, grasp_end1=z0, peak_first=z0, t_first=z0 - 1.0,
              fn_prev=z0, gate=z0, cons=z0, sat=z0, guard=z0, nval=z0, viol=z0, nan=z0, Ub_run=z0, tank_bound=big, sat_below=z0, mis_hold=z0, mis_n=z0, mis_all=z0,
              xi_end4=z0, fstar=z0 + task['fstar'], fother=z0 + task['f_light'] + task['f_heavy'] - task['fstar'])
    def inner(carry, k):
        st, cs, prop, m, key = carry
        key, ko = jax.random.split(key)
        obs = env.observe(st, par, task, ko); taup = cs['tau_prev']
        tau, cs, dg = C.ctrl_step(cs, obs, prop, cfg, LNOM, MNOM, memfn=memfn)
        val = (k < edges[3]).astype(tau.dtype); j = env.phase(k, task); t = k * P.TS
        fn, p, _ = env.top_force(st, par); om = env.cost_rate(st, par, task)
        ustar = Mo.gravity(st['q'], env.pe(par, st), st['xi']) - par['d']; mis = jnp.linalg.norm(dg['uad'] - ustar)
        st_n = env.control_step(st, tau, par, nsub)
        Ub_run = jnp.maximum(m['Ub_run'], C.barrier_potential(st['q'], cfg))
        inwin = val * ((k >= edges[j] - nwin[j]) & (k < edges[j])).astype(tau.dtype)
        onset = (fn > 0) & (m['fn_prev'] <= 0) & (val > 0)
        t_first = jnp.where(onset & (m['t_first'] < 0) & (j == 2), t, m['t_first']); near_first = (t_first >= 0) & (t - t_first <= 0.3) & (val > 0)
        ph = jax.nn.one_hot(j, 4, dtype=tau.dtype); pw = inwin * ph * jnp.asarray([0.0, 1.0, 0.0, 1.0]); fw = inwin * (j == 2)
        m = dict(m, cost=m['cost'] + val * om * P.TS, cost_ph=m['cost_ph'] + val * om * P.TS * ph, pos_err=m['pos_err'] + pw * jnp.linalg.norm(p - task['ps']), pos_n=m['pos_n'] + pw,
                 f_rel=m['f_rel'] + fw * jnp.abs(fn - task['fstar']) / task['fstar'], f_n=m['f_n'] + fw, fn_sum=m['fn_sum'] + fw * fn,
                 grasp_end1=jnp.where(k == edges[0] - 1, st['xi'], m['grasp_end1']), peak_first=jnp.where(near_first, jnp.maximum(m['peak_first'], fn), m['peak_first']),
                 t_first=t_first, fn_prev=fn, gate=m['gate'] + val * (dg['alpha'] < 1.0), cons=m['cons'] + val * (jnp.linalg.norm(dg['tau_b']) > 0),
                 sat=m['sat'] + val * (dg['lam'] < 1.0), guard=m['guard'] + val * dg['guard'], nval=m['nval'] + val,
                 viol=m['viol'] + val * jnp.any((st_n['q'] < QLO) | (st_n['q'] > QHI)), nan=m['nan'] + (~jnp.all(jnp.isfinite(st_n['q']))), Ub_run=Ub_run,
                 tank_bound=jnp.where(val > 0, jnp.minimum(m['tank_bound'], cs['T'] - cfg['Tmin'] + Ub_run), m['tank_bound']),
                 sat_below=jnp.maximum(m['sat_below'], val * ((cs['T'] < cfg['Tmin']) & (dg['lam'] < 1.0))),
                 mis_hold=m['mis_hold'] + inwin * (j == 1) * mis, mis_n=m['mis_n'] + inwin * (j == 1), mis_all=m['mis_all'] + val * mis,
                 xi_end4=jnp.where(k == edges[3] - 1, st['xi'], m['xi_end4']))
        rec = dict(q=obs['q'], taup=taup, te=obs['tau_ext'], zeta=dg['zeta'], ustar=ustar, val=val, xi=st['xi'], alpha=dg['alpha'])   # alpha: gate (videos)
        return (st_n, cs, prop, m, key), rec
    def outer(carry, kd):
        st, cs, m, key, eps_h = carry
        key, ko, ki, ka, kb = jax.random.split(key, 5)
        eps_h = jnp.where(kd % HOLD == 0, jax.random.normal(ka, eps_h.shape), eps_h)
        k0 = kd * P.NB; j = env.phase(k0, task); vd = (k0 < edges[3]).astype(st['q'].dtype)
        obs = env.observe(st, par, task, ko)
        led, cs = C.ledger(cs, obs['q'], cfg, LNOM, MNOM, memfn=memfn)
        ob = dict(q=obs['q'], dq=obs['dq'], te=obs['tau_ext'], p=obs['p'], xi=obs['xi'], o=obs['o'], j=j, T=led['T'], lam=led['lam'], uload=led['uload'] * (1.0 - blind))
        x = obs_vec(ob, task, cfg); xc = jnp.concatenate([x, priv_vec(task, par, k0, edges, j)])
        mu, log_std, logit = policy(bp, x)
        u = jnp.where(mode > 0.5, mu, mu + jnp.exp(log_std) * eps_h)
        if GRIP_EPS > 0:                                                          # arm I: exploration mixture in training (mode 0), phase 4 only (amendment 1)
            mix = (mode < 0.5) & (j == 3)                                         # elsewhere the unmodified sigmoid sampling and log-probability
            pg = jnp.where(mix, grip_prob(logit, GRIP_EPS), jax.nn.sigmoid(logit))
            psi = jnp.where((mode > 0.5) & (mode < 1.5), (logit > 0).astype(mu.dtype), (jax.random.uniform(kb) < pg).astype(mu.dtype))
            lp = jnp.where(mix, logp_mix(mu, log_std, logit, u, psi, GRIP_EPS), logp_of(mu, log_std, logit, u, psi)); v = value(bp, xc)
        else:
            psi = jnp.where((mode > 0.5) & (mode < 1.5), (logit > 0).astype(mu.dtype), (jax.random.uniform(kb) < jax.nn.sigmoid(logit)).astype(mu.dtype))   # mode 2: gripper sampled
            lp = logp_of(mu, log_std, logit, u, psi); v = value(bp, xc)
        qdp, Kp = act_map(u, obs['q'])
        st0_k, cs0_k = st, cs                                                     # state at the decision instant (visualization)
        st = env.gripper_event(st, psi, par, task)
        c0 = m['cost']
        (st, cs, _, m, _), rec = jax.lax.scan(inner, (st, cs, dict(qdp=qdp, Kp=Kp), m, ki), k0 + jnp.arange(P.NB))
        t3 = jnp.sum(st0_k['q']); att = st0_k['xi'] > 0.5
        viz = dict(q=st0_k['q'], xi=st0_k['xi'], com=jnp.where(att, Mo.obj_pos(st0_k['q'], env.pe(par, st0_k)), st0_k['obj_c']),
                   opsi=jnp.where(att, t3 + st0_k['odelta'], st0_k['obj_psi']), fn=env.top_force(st0_k, par)[0], T=cs0_k['T'], uload=led['uload'],
                   fe=Mo.contact(Mo.tip(st0_k['q'], par['L']), Mo.jac(st0_k['q'], par['L']) @ st0_k['dq'], par)[0])
        dec = dict(x=x, xc=xc, u=u, psi=psi, logp=lp, v=v, r=-(m['cost'] - c0), valid=vd, Kp=Kp, j=j, viz=viz)
        return (st, cs, m, key, eps_h), (dec, rec)
    (st, cs, m, _, _), (dec, rec) = jax.lax.scan(outer, (st0, cs0, m0, key, jnp.zeros(ACT_DIM)), jnp.arange(ND))
    return st, m, dec, rec

def make_batch_fn(cfg):
    cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
    f = jax.vmap(lambda p, t, k, bp, hp, mode, blind: rollout(p, t, k, cfgj, bp, hp, mode, blind), in_axes=(0, 0, 0, None, None, None, None))
    return jax.jit(f)

def summarize(st, m):
    f = lambda a: np.asarray(a); pos = f(m['pos_err']) / np.maximum(f(m['pos_n']), 1); fb = f(m['fn_sum']) / np.maximum(f(m['f_n']), 1)
    return dict(cost=f(m['cost']), cost_ph=f(m['cost_ph']), pos2=pos[:, 1], pos4=pos[:, 3], frel=f(m['f_rel']) / np.maximum(f(m['f_n']), 1), fbar=fb,
                class_ok=(np.abs(fb - f(m['fstar'])) < np.abs(fb - f(m['fother']))).astype(float), grasp=f(m['grasp_end1']), released=1.0 - f(m['xi_end4']),
                peak_first=f(m['peak_first']), gate=f(m['gate']) / f(m['nval']), cons=f(m['cons']) / f(m['nval']), sat=f(m['sat']) / f(m['nval']), guard=f(m['guard']) / f(m['nval']),
                viol=f(m['viol']), nan=f(m['nan']), tank_bound=f(m['tank_bound']), sat_below=f(m['sat_below']), mis_hold=f(m['mis_hold']) / np.maximum(f(m['mis_n']), 1),
                mis_all=f(m['mis_all']) / f(m['nval']), n_grasp=f(st['n_grasp']))

# hand role loss replayed on the recorded rollouts (the plant trajectory is held fixed)
def replay_loss(hp, rec, cfg, w_mode=0):
    """w_mode 0: every instant counts. w_mode 1 (proposal, pilot): the error at t is weighted by 1 - exp(-lam_s r(t)), r(t) the rest time of the memory
    since the last change of the attached load (grasp or release; privileged event), r = infinity before the first change: the share of the last load change
    that the rest-gated memory could have learned by t."""
    def one(r):
        def step(c, x):
            z, rr, xp = c
            rr = jnp.where(jnp.abs(x['xi'] - xp) > 0.5, 0.0, rr)
            G = hand_G(hp, x['q']); uad = G.T @ (abar_of(hp) + z)
            w = 1.0 if w_mode == 0 else 1.0 - jnp.exp(-cfg['lam_s'] * rr)
            e = x['val'] * w * jnp.sum((uad - x['ustar']) ** 2)
            mhat = uad - x['taup'] - x['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
            dz = G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mhat)
            return (jnp.clip(z - P.TS * cfg['lam_s'] * x['zeta'] * dz, -cfg['zbar'], cfg['zbar']), rr + P.TS * x['zeta'], x['xi']), e
        _, e = jax.lax.scan(jax.checkpoint(step), (jnp.zeros(NA), jnp.asarray(1e9), jnp.zeros(())), r)   # detached before t = 0: an attached start is a load change
        return jnp.sum(e) / jnp.maximum(jnp.sum(r['val']), 1.0)
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec)
    return jnp.mean(jax.vmap(one)(flat))

def transfer_loss(hp, rec, key, npair=64):
    """Transfer term L_tr of the role loss (records/l2_hand_ltr_plan.md): in each episode, npair uniformly drawn pairs (a, b) of control steps, kept if both are
    valid and have the same attachment state; error of the calibration made at a, z_a* = G_a (G_a^T G_a + eps I)^{-1} (u*_a - G_a^T abar), carried to b:
    |G_b^T (abar + z_a*) - u*_b|^2. Mean over kept pairs per episode, then over episodes. Returns (loss, number of kept pairs)."""
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec); n = flat['val'].shape[1]; ab = abar_of(hp)
    def zstar(G, us):
        GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
        return G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), us - G.T @ ab)
    def one(r, k):
        ka, kb = jax.random.split(k); ia = jax.random.randint(ka, (npair,), 0, n); ib = jax.random.randint(kb, (npair,), 0, n)
        keep = r['val'][ia] * r['val'][ib] * (jnp.abs(r['xi'][ia] - r['xi'][ib]) < 0.5)
        e = jax.vmap(lambda a, b: jnp.sum((hand_G(hp, r['q'][b]).T @ (ab + zstar(hand_G(hp, r['q'][a]), r['ustar'][a])) - r['ustar'][b]) ** 2))(ia, ib)
        return jnp.sum(keep * e) / jnp.maximum(jnp.sum(keep), 1.0), jnp.sum(keep)
    v, kept = jax.vmap(one)(flat, jax.random.split(key, flat['val'].shape[0]))
    return jnp.mean(v), jnp.sum(kept)

def make_hand_grad_tr(cfg, w_mode, lam):
    """Role loss + lam * L_tr; value and grad of the sum, aux = (role loss, L_tr). Signature (hp, rec, key)."""
    cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
    def f(hp, rec, key):
        lr_, lt = replay_loss(hp, rec, cfgj, w_mode), transfer_loss(hp, rec, key)[0]
        return lr_ + lam * lt, (lr_, lt)
    return jax.jit(jax.value_and_grad(f, has_aux=True))

def make_hand_grad(cfg, w_mode=0):
    cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
    return jax.jit(jax.value_and_grad(lambda hp, rec: replay_loss(hp, rec, cfgj, w_mode)))

# PPO
def gae(r, v, valid, gamma, lam):
    def step(carry, x):
        adv_next, v_next, valid_next = carry
        r_t, v_t, val_t = x
        delta = r_t + gamma * v_next * valid_next - v_t
        adv = (delta + gamma * lam * valid_next * adv_next) * val_t
        return (adv, v_t, val_t), adv
    _, adv = jax.lax.scan(step, (0.0, 0.0, 0.0), (r, v, valid), reverse=True)
    return adv

def ppo_loss(bp, b, clip_eps, c_v, c_ent):
    mu, log_std, logit = jax.vmap(lambda x: policy(bp, x))(b['x'])
    lp = jax.vmap(logp_of)(mu, log_std, logit, b['u'], b['psi'])
    ratio = jnp.exp(lp - b['logp']); a = b['adv']
    pg = -jnp.mean(jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a))
    vl = jnp.mean(((jax.vmap(lambda xc: value(bp, xc))(b['xc']) - b['ret']) / jax.lax.stop_gradient(bp.get('vscale', 1.0))) ** 2)
    pb = jax.nn.sigmoid(logit); ent = jnp.mean(jnp.sum(log_std, -1)) + jnp.mean(-(pb * jnp.log(pb + 1e-8) + (1 - pb) * jnp.log(1 - pb + 1e-8)))
    return pg + c_v * vl - c_ent * ent, dict(pg=pg, vl=vl, ent=ent, kl=jnp.mean(b['logp'] - lp), clipfrac=jnp.mean(jnp.abs(ratio - 1) > clip_eps),
                                             sd_goal=jnp.mean(jnp.exp(log_std[:, :3])), sd_K=jnp.mean(jnp.exp(log_std[:, 3:])))

ppo_grad = jax.jit(jax.value_and_grad(ppo_loss, has_aux=True), static_argnums=(2, 3, 4))

def ppo_loss_i(bp, b, clip_eps, c_v, c_ent_cont, c_ent_grip, grip_eps, w_cls):
    """Attempt 4 arm I: ppo_loss with separate entropy coefficients of the continuous actions and of the gripper, the gripper log-probability and entropy
    under grip_prob(logit, grip_eps) on phase-4 decisions (observation entry PH4; amendment 1) and the sigmoid elsewhere, and w_cls times the class-reading loss: binary cross-entropy of bp['cls'] on the actor's last hidden layer against the
    privileged heavy label (critic input entry PRIV 4) on decisions with the object held (observation entry 21). The coefficients are traced (no recompilation)."""
    mu, log_std, logit = jax.vmap(lambda x: policy(bp, x))(b['x'])
    ph4 = b['x'][:, PH4] > 0.5
    lp = jnp.where(ph4, jax.vmap(logp_mix, in_axes=(0, 0, 0, 0, 0, None))(mu, log_std, logit, b['u'], b['psi'], grip_eps), jax.vmap(logp_of)(mu, log_std, logit, b['u'], b['psi']))
    ratio = jnp.exp(lp - b['logp']); a = b['adv']
    pg = -jnp.mean(jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a))
    vl = jnp.mean(((jax.vmap(lambda xc: value(bp, xc))(b['xc']) - b['ret']) / jax.lax.stop_gradient(bp.get('vscale', 1.0))) ** 2)
    pp = jnp.where(ph4, grip_prob(logit, grip_eps), jax.nn.sigmoid(logit)); ent_c = jnp.mean(jnp.sum(log_std, -1)); ent_g = jnp.mean(-(pp * jnp.log(pp + 1e-8) + (1 - pp) * jnp.log(1 - pp + 1e-8)))
    zc = (jax.vmap(lambda x: actor_hidden(bp, x))(b['x']) @ bp['cls']['W'] + bp['cls']['b'])[:, 0]
    y = b['xc'][:, -PRIV_DIM + 4]; hold = (b['x'][:, 21] > 0.5).astype(zc.dtype); nh = jnp.maximum(jnp.sum(hold), 1.0)
    bce = jnp.sum(hold * (jnp.logaddexp(0.0, zc) - y * zc)) / nh
    acc = jnp.sum(hold * ((zc > 0) == (y > 0.5))) / nh
    loss = pg + c_v * vl - c_ent_cont * ent_c - c_ent_grip * ent_g + w_cls * bce
    return loss, dict(pg=pg, vl=vl, ent=ent_c + ent_g, kl=jnp.mean(b['logp'] - lp), clipfrac=jnp.mean(jnp.abs(ratio - 1) > clip_eps),
                      sd_goal=jnp.mean(jnp.exp(log_std[:, :3])), sd_K=jnp.mean(jnp.exp(log_std[:, 3:])), cls_bce=bce, cls_acc=acc, n_hold=jnp.sum(hold))

ppo_grad_i = jax.jit(jax.value_and_grad(ppo_loss_i, has_aux=True), static_argnums=(2, 3))

# diagnostics of the hand features: calibration at one posture, residual at another (minimum-norm rest calibration, eps -> 0)
def transfer_residual(G_A, G_B, base_A, base_B, us_A, us_B):
    zA = G_A @ jnp.linalg.solve(G_A.T @ G_A + 1e-9 * jnp.eye(3), us_A - base_A)
    return jnp.linalg.norm(base_B + G_B.T @ zA - us_B), jnp.linalg.norm(base_B - us_B)

def hand_diagnostics(hp, eps):
    """For each episode: residual at the target posture after rest calibration at the grasp posture with the object attached, for the learned
    features (base = G^T a_bar) and for the fixed class basis with the nominal body prior (L0, L1); and the error of the learned prior at the start posture."""
    out = dict(res_learned=[], base_learned=[], res_class=[], base_class=[], prior_learned=[], prior_nominal=[], prior_zero=[])
    for par, task in eps:
        pj = {k: jnp.asarray(v) for k, v in par.items()}
        qA, qB, q0 = jnp.asarray(task['qo_b']), jnp.asarray(task['qs_b']), jnp.asarray(task['q0'])
        usA = Mo.gravity(qA, pj, 1.0) - pj['d']; usB = Mo.gravity(qB, pj, 1.0) - pj['d']; us0 = Mo.gravity(q0, pj, 0.0) - pj['d']
        GA, GB, G0_ = hand_G(hp, qA), hand_G(hp, qB), hand_G(hp, q0); ab = abar_of(hp)
        r, b = transfer_residual(GA, GB, GA.T @ ab, GB.T @ ab, usA, usB); out['res_learned'].append(float(r)); out['base_learned'].append(float(b))
        cA, cB = C.feats_class(qA), C.feats_class(qB)
        r, b = transfer_residual(cA, cB, C.prior_body(qA, LNOM, MNOM), C.prior_body(qB, LNOM, MNOM), usA, usB); out['res_class'].append(float(r)); out['base_class'].append(float(b))
        out['prior_learned'].append(float(jnp.linalg.norm(G0_.T @ ab - us0))); out['prior_nominal'].append(float(jnp.linalg.norm(C.prior_body(q0, LNOM, MNOM) - us0)))
        out['prior_zero'].append(float(jnp.linalg.norm(us0)))
    return {k: np.asarray(v) for k, v in out.items()}
