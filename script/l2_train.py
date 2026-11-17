"""L2 training run (one seed): coupled learning of the brain (PPO) and the hand memory features (role loss), with per-iteration
guarantee checks and periodic component diagnostics. Usage: python3 l2_train.py OUT_DIR N_IT [N_EP] [POOL] [WARM] [cost=1|2|3|4] [grip_reset=1] [role=1] [restart=BETA] [eval_restart=1] [act=joint|task]
WARM: a checkpoint of the first phase of attempt 1 (upgraded function-preservingly) or of the current format (loaded as is).
restart: share BETA of each batch starts from the restart states of env.restart (half at the start of phase 2, half at the start of phase 3); the standard
training statistics are taken on the other episodes, the guarantee checks on all. eval_restart: the evaluation set is also run from both restart states.
vwarm: the first K iterations update only the critic (actor and hand frozen; policy evaluation before improvement); the optimizer state of the brain is then reset.
act: action mode of the brain (l2.goal_of); a warm start keeps the mode of its checkpoint unless act= is given.
group=new|<timestamp> plan=records/<name>_plan.md purpose="...": OUT_DIR is then a run name, created as data/runs/<timestamp>/<name>/ with the group's
manifest.json and data/runs/INDEX.md (pairl.tools.runs); the plan must carry a current hash line. Without group= OUT_DIR is a path as before.
Optimization options (pilot O; absent = the values above, unchanged path): c_ent= entropy coefficient; kl_target= KL-adaptive brain learning rate (after each
iteration lr / 1.5 if the mean KL > 2 target, lr * 1.5 if < target / 2, bounds [1e-5, 1e-3]); lr_h_half= hand learning rate halved every that many
iterations after the start iteration; mb_scale=1 PPO minibatch 4096 * N_EP / 128; cost=5 cost revision 17 (phase-1 speed term, env.cost_rate);
near=SHARE share of the batch starting phase 1 near the object (env.near_start), taken from the first starts; obs_tip=1 the ledger load as a tip load added to
the observation (zero-initialized inputs on a warm start); ltr= weight of the transfer term L_tr added to the hand's role loss
(l2.transfer_loss; 64 pairs per episode, keys PRNGKey(9000 + hand step); hist gains hand_ltr, the mean L_tr of the iteration).
Attempt 4 arm I options (records/l2_attempt4_plan.md; absent = the path above, unchanged): cls_aux=W class-reading loss of a zero-initialized head on the
actor's last hidden layer (l2.ppo_loss_i; added on a warm start if absent, zero Adam moments); c_ent_cont_end=C entropy coefficient of the continuous actions
decayed geometrically from c_ent (first iteration) to C (last iteration), the gripper term keeping c_ent; grip_eps=E training gripper sampled from
(1 - E) p + E / 2 with the PPO log-probability and entropy under it (l2.GRIP_EPS; evaluation unchanged). Any of them selects l2.ppo_grad_i."""
import jax; jax.config.update('jax_enable_x64', True)
import os, sys, json, time, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
from pairl import paths

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = POS[0]; N_IT = int(POS[1]); N_EP = int(POS[2]) if len(POS) > 2 else 128; POOL = int(POS[3]) if len(POS) > 3 else 6144
WARM = POS[4] if len(POS) > 4 else None
l2.ACT_MODE = OPT.get('act', 'joint')
env.COST_V = int(OPT.get('cost', 4))                                    # set before any function is traced (4: spec rev. 15; attempt 2 used 3, attempt 1 used 1)
if 'group' in OPT:                                                      # experiment group under data/runs (manifest, INDEX.md)
    from pairl.tools import runs as RG
    assert os.sep not in OUT, 'with group= the first argument is a run name'
    RUN_DIR = RG.open_run(OPT['group'], OUT, OPT.get('plan'), OPT.get('purpose', ''), sys.argv); OUT = str(RUN_DIR)
    class _Tee:                                                         # train.log in the run directory, as the redirected stdout of earlier runs
        def __init__(s, *f): s.f = f
        def write(s, x): [f.write(x) for f in s.f]
        def flush(s): [f.flush() for f in s.f]
    sys.stdout = _Tee(sys.stdout, open(RUN_DIR / 'train.log', 'w'))
    import atexit; _done = {}; _eh = sys.excepthook
    sys.excepthook = lambda *e: (_done.__setitem__('err', True), _eh(*e))          # an exception marks the run failed; SIGKILL leaves it 'running'
    atexit.register(lambda: RG.close_run(RUN_DIR, 'complete' if _done.get('ok') else ('failed' if _done.get('err') else 'stopped early')))
os.makedirs(OUT, exist_ok=True)
SEED = 0
HP = dict(gamma=0.99, lam=0.95, clip=0.2, c_v=0.5, c_ent=1e-3, lr_b=3e-4, lr_h=1e-3, epochs=5, mb=4096, h_steps=16, h_batch=min(32, N_EP), eval_every=10, n_eval=64)
if 'c_ent' in OPT: HP['c_ent'] = float(OPT['c_ent'])
if OPT.get('mb_scale', '0') == '1': assert N_EP % 128 == 0; HP['mb'] = 4096 * N_EP // 128
KL_T = float(OPT['kl_target']) if 'kl_target' in OPT else None; LR_H_HALF = float(OPT['lr_h_half']) if 'lr_h_half' in OPT else None
lr_b = HP['lr_b']                                                     # current brain learning rate (changes only with kl_target)
cfg = C.default_cfg()
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)

# reference energy E_task of budget episodes: the L1 reference brain with the L1 hand (one compiled function, padded chunks)
REF = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, noqd=1.0, lead_t=0.15, k_c=1.0, f_mode=1.0)
cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
optj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in REF.items()}
ref_fn = jax.jit(jax.vmap(lambda p, t, k: l1.rollout(p, t, k, l2.ND, cfgj, 'ref', optj)))
def e_task(eps, chunk=256):
    out = []
    for i in range(0, len(eps), chunk):
        part = eps[i:i + chunk]; pad = part + [part[-1]] * (chunk - len(part))
        pb, tb = env.stack(l1.with_E0(pad, P.E_NOM)); st, cs, m, _ = ref_fn(pb, tb, jax.random.split(jax.random.PRNGKey(77 + i), chunk))
        out += list(np.asarray(m['E_task'])[:len(part)])
    return np.asarray(out)
def with_budget(eps, all_budget=False):
    idx = [i for i, (p, t) in enumerate(eps) if all_budget or t['budget'] > 0.5]
    Et = e_task([eps[i] for i in idx]) if idx else []
    E0 = [P.E_NOM] * len(eps)
    for i, e in zip(idx, Et): E0[i] = cfg['dT'] + float(eps[i][1]['rho']) * float(e)
    return l1.with_E0(eps, E0)

log('cost version', env.COST_V, 'role weighting', OPT.get('role', '1'), 'warm start', WARM, 'action mode', l2.ACT_MODE)
log('building the training pool and the evaluation sets')
cache = str(paths.POOLS / f'l2_pool_E0_{POOL}.json')
if os.path.exists(cache): pool = l1.with_E0(l1.episodes(10000, POOL, lim=0.7), json.load(open(cache)))
else:                                                                  # a new pool size only; the existing cloud-built pool is never rebuilt
    pool = with_budget(l1.episodes(10000, POOL, lim=0.7))
    with open(cache, 'x') as f: json.dump([float(t['E0']) for p, t in pool], f)        # 'x': refuses to overwrite an existing pool file
ev = l1.episodes(6000, HP['n_eval'], lim=0.7)
ev_nom = l1.with_E0(ev, P.E_NOM); ev_bud = with_budget(ev, all_budget=True)
log('pool ready', len(pool), 'budget episodes', int(sum(t['budget'] for p, t in pool)))

bp, hp = l2.init_params(jax.random.PRNGKey(SEED)); IT0 = 0; TIP_NEW = False
if WARM:
    ck = pickle.load(open(WARM, 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp']); IT0 = int(ck['it'])
    if 'act' not in OPT: l2.set_mode(ck)
    TIP_NEW = OPT.get('obs_tip', '0') == '1' and not l2.OBS_TIP     # arm B: add the tip-load inputs, zero-initialized
    if 'log_std' in bp:
        bp = l2.upgrade_v1(bp); b_ = bp['actor'][-1]['b']
        if OPT.get('sd_reset', '1') == '1':
            bp['actor'][-1]['b'] = b_.at[l2.ACT_DIM:l2.ACT_DIM + 3].set(l2.std_raw(jnp.asarray([0.04] * 3 + [0.3] * 3))[:3])   # goal sd reset to 0.04 (attempt 1b)
    if OPT.get('grip_reset', '0') == '1':                                # gripper head back to its initial value (closed with probability 0.9975 everywhere)
        W_, b_ = bp['actor'][-1]['W'], bp['actor'][-1]['b']; k_ = 2 * l2.ACT_DIM
        bp['actor'][-1] = dict(W=W_.at[:, k_].set(0.0), b=b_.at[k_].set(6.0))
    log('warm start from', WARM, 'iteration', IT0, 'cost version', env.COST_V, 'gripper reset', OPT.get('grip_reset', '0'), 'role weighting', OPT.get('role', '1'), 'action mode', l2.ACT_MODE)
if 'sd_goal' in OPT and not WARM:                                      # initial goal standard deviation (pre-squash) of a run from initialization
    b_ = bp['actor'][-1]['b']; bp['actor'][-1]['b'] = b_.at[l2.ACT_DIM:l2.ACT_DIM + 3].set(l2.std_raw(jnp.asarray([float(OPT['sd_goal'])] * 3 + [0.3] * 3))[:3])
    log('initial goal standard deviation', OPT['sd_goal'])
if OPT.get('rel_zero', '0') == '1' and not WARM:                      # relative-position inputs (functions of other inputs) start with zero first-layer weights
    for net in ('actor', 'critic'):
        W0 = bp[net][0]['W']; bp[net][0] = dict(bp[net][0], W=W0.at[l2.OBS_DIM_V1:l2.OBS_DIM].set(0.0))
    log('relative-position inputs: zero initial weights')
VNORM = OPT.get('vnorm', '0') == '1'
if VNORM and 'vscale' not in bp: bp['vscale'] = jnp.asarray(1.0)          # return scale of the critic, set from data each iteration
ab, ah = l2.adam_init(bp), l2.adam_init(hp)
if WARM and 'opt' in ck and OPT.get('opt_reset', '0') != '1':                 # exact resumption: optimizer states of the checkpoint
    ab, ah = (jax.tree_util.tree_map(jnp.asarray, ck['opt'][k]) for k in ('brain', 'hand')); log('optimizer states restored from', WARM)
if TIP_NEW:                                                             # arm B: zero rows in the first layers (parameters and Adam moments); the outputs must not change
    xs = jax.random.normal(jax.random.PRNGKey(31), (2048, l2.OBS_DIM)); xt = jnp.concatenate([xs, 3 * jax.random.normal(jax.random.PRNGKey(32), (2048, 2))], 1)
    before = jax.vmap(lambda x: l2.policy(bp, x))(xs); vb = jax.vmap(lambda x: l2.value(bp, x))(jnp.concatenate([xs, jnp.zeros((2048, l2.PRIV_DIM))], 1))
    bp = l2.add_tip_inputs(bp); ab = dict(ab, m=l2.add_tip_inputs(ab['m']), v=l2.add_tip_inputs(ab['v'])); l2.OBS_TIP = True
    after = jax.vmap(lambda x: l2.policy(bp, x))(xt); va = jax.vmap(lambda x: l2.value(bp, x))(jnp.concatenate([xt, jnp.zeros((2048, l2.PRIV_DIM))], 1))
    dmax = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(after) + [va], jax.tree_util.tree_leaves(before) + [vb]))
    log('tip-load inputs added (zero-initialized); max change of the actor and critic outputs on 2048 random inputs with random tip loads:', f'{dmax:.3e}')
    assert dmax < 1e-9, 'adding the zero-initialized inputs changed the outputs'
I_MODE = any(k in OPT for k in ('cls_aux', 'c_ent_cont_end', 'grip_eps'))         # attempt 4 arm I: l2.ppo_loss_i
W_CLS = float(OPT.get('cls_aux', 0.0)); C_END = float(OPT['c_ent_cont_end']) if 'c_ent_cont_end' in OPT else HP['c_ent']; l2.GRIP_EPS = float(OPT.get('grip_eps', 0.0))
if I_MODE:
    if 'cls' not in bp:
        bp = l2.add_cls_head(bp); z_ = lambda t: dict(t, cls=jax.tree_util.tree_map(jnp.zeros_like, bp['cls'])); ab = dict(ab, m=z_(ab['m']), v=z_(ab['v']))
        log('class-reading head added (zero-initialized, zero Adam moments)')
    log('arm I: cls_aux', W_CLS, '| continuous entropy coefficient', HP['c_ent'], '->', C_END, '| gripper entropy coefficient', HP['c_ent'], '| grip_eps', l2.GRIP_EPS)
ROLE = int(OPT.get('role', 1)); batch_fn = l2.make_batch_fn(cfg); hand_grad = l2.make_hand_grad(cfg, ROLE)
LTR = float(OPT['ltr']) if 'ltr' in OPT else None; LTR_STEP = [0]; lt_it = []                 # ltr=lambda: role loss + lambda L_tr (pilot L)
if LTR is not None:
    _hg_tr = l2.make_hand_grad_tr(cfg, ROLE, LTR)
    def hand_grad(hp, rec_):                                           # same signature; the value returned is the role loss alone
        LTR_STEP[0] += 1; (_, (lr_, lt)), g = _hg_tr(hp, rec_, jax.random.PRNGKey(9000 + LTR_STEP[0])); lt_it.append(float(lt)); return lr_, g
BETA = float(OPT.get('restart', 0.0)); EV_R = OPT.get('eval_restart', '0') == '1'; VWARM = int(OPT.get('vwarm', 0))
N0 = N_EP - 2 * int(round(BETA * N_EP / 2)); N2 = (N_EP - N0) // 2                     # episodes per batch from the original start, from phase 2, from phase 3
NN = int(round(float(OPT.get('near', 0)) * N_EP)); assert NN < N0; rng_near = np.random.default_rng(SEED + 77); n_near_fail = 0   # near= share of near-object starts
log('start-state distribution: original', N0, 'phase-2 restarts', N2, 'phase-3 restarts', N_EP - N0 - N2, '| restart evaluations', EV_R, '| critic warm-up iterations', VWARM)
ev_r = {kind: [env.restart(p, t, kind) for p, t in ev_nom] for kind in (2, 3)} if EV_R else {}
def restart_stats(eps, dec):
    """Per episode: attached at the start of phase 3 and at the start of phase 4, and attached with top-face force in the force window (last 40% of phase 3)."""
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']); fn = np.asarray(dec['viz']['fn'])
    e = np.cumsum(np.array([t['nph'] for p, t in eps]), 1) // P.NB; kd = np.arange(j.shape[1])[None]; ar = np.arange(len(eps))
    held3 = xi[ar, np.argmax(v & (j == 2), 1)] > 0.5; held4 = xi[ar, np.argmax(v & (j == 3), 1)] > 0.5
    late = v & (j == 2) & (xi > 0.5) & (kd >= e[:, 2:3] - np.round(0.4 * (e[:, 2:3] - e[:, 1:2])))
    return dict(held_start3=held3.astype(float).tolist(), held_start4=held4.astype(float).tolist(), top_window=np.any(late & (fn > 0), 1).astype(float).tolist())
cfgj_h = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
hand_loss_all = jax.jit(lambda hp, rec: l2.replay_loss(hp, rec, cfgj_h, 0))         # unweighted role loss (diagnostic when ROLE = 1)
rng = np.random.default_rng(SEED); key = jax.random.PRNGKey(1000 + SEED)
gae_b = jax.jit(jax.vmap(lambda r, v, d: l2.gae(r, v, d, HP['gamma'], HP['lam'])))
def cls_contact(s):
    c = (s['fbar'] > 0) & (s['grasp'] > 0.5); return (float(s['class_ok'][c].mean()) if c.any() else None), int(c.sum())
logit_fn = jax.jit(lambda bp, x: jax.vmap(jax.vmap(lambda a: l2.policy(bp, a)[2]))(x))
def gripper_stats(bp, dec):
    """Open probability while attached (phases 2-3), releases before the release phase, attachment and contact in phase 3."""
    x = np.asarray(dec['x']); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = x[..., 21]
    lg = np.asarray(logit_fn(bp, jnp.asarray(x)))
    att = v & (xi > 0.5) & ((j == 1) | (j == 2)); rel = (np.diff(xi, axis=1) < -0.5) & v[:, :-1] & (j[:, :-1] <= 2)
    a3 = v & (j == 2) & (xi > 0.5); te = np.linalg.norm(x[..., 6:9] * 10, axis=-1) > cfg['tau_th']
    return dict(p_open_att=float(np.mean(1 / (1 + np.exp(lg[att])))) if att.any() else None, rel_before4=float(np.mean(rel.any(1))),
                att3=float(np.mean(a3.any(1))), touch3=float(np.mean(np.any(a3 & te, 1)[a3.any(1)])) if a3.any() else None)
hist, evals = [], []

def evaluate(it):
    res = {}
    for tag, eps, blind in (('nom', ev_nom, 0.0), ('blind', ev_nom, 1.0), ('bud', ev_bud, 0.0)):
        pb, tb = env.stack(eps); st, m, dec, rec = batch_fn(pb, tb, jax.random.split(jax.random.PRNGKey(555), len(eps)), bp, hp, 1.0, blind)
        s = l2.summarize(st, m); s['Kp_mean_ph'] = [float(np.sum(np.asarray(dec['Kp']).mean(-1) * (np.asarray(dec['j']) == j) * np.asarray(dec['valid'])) / max(1.0, float(np.sum((np.asarray(dec['j']) == j) * np.asarray(dec['valid']))))) for j in range(4)]
        res[tag] = {k: (np.asarray(v).tolist() if hasattr(v, 'tolist') else v) for k, v in s.items()}
    for kind, eps in ev_r.items():
        pb, tb = env.stack(eps); st, m, dec, rec = batch_fn(pb, tb, jax.random.split(jax.random.PRNGKey(555), len(eps)), bp, hp, 1.0, 0.0)
        s = l2.summarize(st, m); res[f'r{kind}'] = dict({k: (np.asarray(v).tolist() if hasattr(v, 'tolist') else v) for k, v in s.items()}, **restart_stats(eps, dec))
    hd = l2.hand_diagnostics(hp, ev_nom); res['hand'] = {k: v.tolist() for k, v in hd.items()}
    res['it'] = it; evals.append(res)
    n, b = res['nom'], res['blind']; g = (np.asarray(n['grasp']) > 0.5) & (np.asarray(n['fbar']) > 0); gb = (np.asarray(b['grasp']) > 0.5) & (np.asarray(b['fbar']) > 0)
    log(f"EVAL it {it}: cost {np.mean(n['cost']):.3f} grasp {np.mean(n['grasp']):.2f} pos2 {np.median(n['pos2']):.4f} frel {np.median(n['frel']):.3f} "
        f"class|contact {np.mean(np.asarray(n['class_ok'])[g]) if g.any() else np.nan:.3f} (n {int(g.sum())}) | blind: class|contact {np.mean(np.asarray(b['class_ok'])[gb]) if gb.any() else np.nan:.3f} (n {int(gb.sum())}) "
        f"frel {np.median(b['frel']):.3f} pos2 {np.median(b['pos2']):.4f} | bud cost {np.mean(res['bud']['cost']):.3f} gate {np.mean(res['bud']['gate']):.3f} | "
        f"hand res learned {np.median(hd['res_learned']):.3f} (base {np.median(hd['base_learned']):.3f}) class {np.median(hd['res_class']):.3f} prior {np.median(hd['prior_learned']):.3f}"
        + (f" | r2: held at phase 3 {np.mean(res['r2']['held_start3']):.2f} pos2 {np.median(np.asarray(res['r2']['pos2'])[np.asarray(res['r2']['held_start3']) > 0.5]) if np.any(np.asarray(res['r2']['held_start3']) > 0.5) else np.nan:.4f}"
           f" | r3: top in window {np.mean(res['r3']['top_window']):.2f} frel {np.median(np.asarray(res['r3']['frel'])[np.asarray(res['r3']['top_window']) > 0.5]) if np.any(np.asarray(res['r3']['top_window']) > 0.5) else np.nan:.3f}"
           f" accurate press {np.mean((np.asarray(res['r3']['frel']) < 0.5) & (np.asarray(res['r3']['held_start4']) > 0.5)):.2f}" if EV_R else ''))
    json.dump(evals, open(os.path.join(OUT, 'evals.json'), 'w'))
    pickle.dump(dict(bp=jax.tree_util.tree_map(np.asarray, bp), hp=jax.tree_util.tree_map(np.asarray, hp), it=it, act=l2.ACT_MODE, **({'obs': 'tipload'} if l2.OBS_TIP else {}),
                     opt=dict(brain=jax.tree_util.tree_map(np.asarray, ab), hand=jax.tree_util.tree_map(np.asarray, ah))), open(os.path.join(OUT, f'params_it{it:04d}.pkl'), 'wb'))

evaluate(IT0)
for it in range(IT0 + 1, IT0 + N_IT + 1):
    t_it = time.time(); warm = it <= IT0 + VWARM
    if VWARM and it == IT0 + VWARM + 1: ab = l2.adam_init(bp); log('critic warm-up done; optimizer state of the brain reset')
    idx = rng.choice(len(pool), N_EP, replace=False)
    eps = [pool[i] for i in idx] if BETA == 0 else [env.restart(*pool[i], 0 if n < N0 else (2 if n < N0 + N2 else 3)) for n, i in enumerate(idx)]
    if NN:                                                              # arm S2: the last NN episodes of the first-start share begin near the object
        for n in range(N0 - NN, N0): p_, t_, ok_ = env.near_start(*eps[n], rng_near); eps[n] = (p_, t_); n_near_fail += (not ok_)
    pb, tb = env.stack(eps)
    key, kr = jax.random.split(key)
    st, m, dec, rec = batch_fn(pb, tb, jax.random.split(kr, N_EP), bp, hp, 0.0, 0.0)
    s_all = l2.summarize(st, m); sub0 = lambda tr: jax.tree_util.tree_map(lambda a: a[:N0 - NN], tr)        # training statistics on the original starts only
    s = s_all if BETA == 0 else l2.summarize(sub0(st), sub0(m)); gs = gripper_stats(bp, dec if BETA == 0 else sub0(dec)); t_roll = time.time() - t_it
    rs = {}
    if BETA > 0:
        sl = lambda tr, a, b: jax.tree_util.tree_map(lambda x: x[a:b], tr)
        r2, r3 = restart_stats(eps[N0:N0 + N2], sl(dec, N0, N0 + N2)), restart_stats(eps[N0 + N2:], sl(dec, N0 + N2, N_EP))
        rs = dict(r2_cost=float(s_all['cost'][N0:N0 + N2].mean()), r2_held_start3=float(np.mean(r2['held_start3'])), r3_cost=float(s_all['cost'][N0 + N2:].mean()),
                  r3_top_window=float(np.mean(r3['top_window'])))
    # hand step (role loss, faster time scale)
    lh = []
    for _ in range(0 if warm else HP['h_steps']):
        sub = rng.choice(N_EP, HP['h_batch'], replace=False)
        l, g = hand_grad(hp, jax.tree_util.tree_map(lambda a: a[sub], rec)); hp, ah = l2.adam_step(hp, g, ah, HP['lr_h'] if LR_H_HALF is None else HP['lr_h'] * 0.5 ** ((it - IT0) / LR_H_HALF), clip=1.0); lh.append(float(l))
    if warm: sub = rng.choice(N_EP, HP['h_batch'], replace=False); lh = [float(hand_grad(hp, jax.tree_util.tree_map(lambda a: a[sub], rec))[0])]
    lh_all = float(hand_loss_all(hp, jax.tree_util.tree_map(lambda a: a[sub], rec))) if ROLE else float(np.mean(lh))
    # brain step (PPO on the task return)
    adv = gae_b(dec['r'], dec['v'], dec['valid']); ret = adv + dec['v']
    if VNORM:                                                            # running scale of the returns (exponential average of their standard deviation)
        sd_ret = float(np.asarray(ret)[np.asarray(dec['valid']) > 0.5].std()); vs0 = float(bp['vscale'])
        bp['vscale'] = jnp.asarray(sd_ret if it == IT0 + 1 and vs0 == 1.0 else 0.9 * vs0 + 0.1 * sd_ret)
    msk = np.asarray(dec['valid']).reshape(-1) > 0.5
    flat = lambda a: np.asarray(a).reshape((-1,) + np.asarray(a).shape[2:])[msk]
    D = dict(x=flat(dec['x']), xc=flat(dec['xc']), u=flat(dec['u']), psi=flat(dec['psi']), logp=flat(dec['logp']), ret=flat(ret), adv=flat(adv))
    D['adv'] = (D['adv'] - D['adv'].mean()) / (D['adv'].std() + 1e-8); nS = len(D['adv']); aux_acc = []
    if I_MODE: c_cont = HP['c_ent'] * (C_END / HP['c_ent']) ** ((it - IT0 - 1) / max(N_IT - 1, 1))
    for _ in range(HP['epochs']):
        perm = rng.permutation(nS)
        mbs = min(HP['mb'], nS)
        for i in range(0, nS - mbs + 1, mbs):
            mb = {k: jnp.asarray(v[perm[i:i + mbs]]) for k, v in D.items()}
            if I_MODE: (lo, aux), g = l2.ppo_grad_i(bp, mb, HP['clip'], HP['c_v'], c_cont, HP['c_ent'], l2.GRIP_EPS, W_CLS)
            else: (lo, aux), g = l2.ppo_grad(bp, mb, HP['clip'], HP['c_v'], HP['c_ent'])
            if warm: g = dict(g, actor=jax.tree_util.tree_map(jnp.zeros_like, g['actor']))      # critic only: zero actor moments, so no actor step
            bp, ab = l2.adam_step(bp, g, ab, lr_b, clip=0.5)
            aux_acc.append({k: float(v) for k, v in aux.items()})
    vv, rr_, va = np.asarray(dec['v']), np.asarray(ret), np.asarray(dec['valid']) > 0.5
    ev_k = lambda a, b: (lambda m: float(1 - np.var((rr_ - vv)[m]) / max(np.var(rr_[m]), 1e-12)) if m.any() else None)(va & (np.arange(N_EP)[:, None] >= a) & (np.arange(N_EP)[:, None] < b))
    rs.update(ev_orig=ev_k(0, N0), **(dict(ev_r2=ev_k(N0, N0 + N2), ev_r3=ev_k(N0 + N2, N_EP)) if BETA > 0 else {}), warm=bool(warm))
    g = s['grasp'] > 0.5; ok = s_all['sat_below'] < 0.5
    h = dict(it=it, cost=float(s['cost'].mean()), grasp=float(s['grasp'].mean()), pos2=float(np.median(s['pos2'])), frel=float(np.median(s['frel'])),
             cls=cls_contact(s)[0], n_contact=cls_contact(s)[1], **gs, released=float(s['released'].mean()), viol=float(s_all['viol'].sum()), nan=float(s_all['nan'].sum()),
             tank_bound_min=float(s_all['tank_bound'][ok].min()) if ok.any() else None, n_sat_below=float((~ok).sum()), gate=float(s_all['gate'].mean()), cons=float(s_all['cons'].mean()),
             sat=float(s_all['sat'].mean()), guard=float(s_all['guard'].mean()), mis_hold=float(np.median(s['mis_hold'])), mis_all=float(s_all['mis_all'].mean()), hand_loss=float(np.mean(lh)), hand_loss_all=lh_all, **rs,
             n_samples=int(nS), t_roll=t_roll, t_it=time.time() - t_it,
             **{k: float(np.mean([a[k] for a in aux_acc])) for k in aux_acc[0]})
    if KL_T is not None:
        h['lr_b'] = lr_b; k_ = h['kl']
        lr_b = min(lr_b * 1.5, 1e-3) if k_ < KL_T / 2 else (max(lr_b / 1.5, 1e-5) if k_ > 2 * KL_T else lr_b)
    if NN: h['near'] = NN; h['near_fail'] = n_near_fail
    if I_MODE: h['c_ent_cont'] = c_cont
    if LTR is not None: h['hand_ltr'] = float(np.mean(lt_it)) if lt_it else None; lt_it.clear()
    if LR_H_HALF is not None: h['lr_h'] = HP['lr_h'] * 0.5 ** ((it - IT0) / LR_H_HALF)
    hist.append(h)
    log(f"it {it}: cost {h['cost']:.3f} grasp {h['grasp']:.2f} pos2 {h['pos2']:.4f} frel {h['frel']:.3f} cls|contact {h['cls'] if h['cls'] is None else round(h['cls'], 3)} (n {h['n_contact']}) "
        f"| p_open_att {h['p_open_att'] if h['p_open_att'] is None else round(h['p_open_att'], 3)} rel<4 {h['rel_before4']:.2f} att3 {h['att3']:.2f} touch3 {h['touch3'] if h['touch3'] is None else round(h['touch3'], 2)} "
        f"| hand {h['hand_loss']:.3f} (all {h['hand_loss_all']:.3f}) mis_hold {h['mis_hold']:.3f} | viol {h['viol']:.0f} tb {h['tank_bound_min'] if h['tank_bound_min'] is None else round(h['tank_bound_min'], 4)} "
        f"gate {h['gate']:.3f} | kl {h['kl']:.4f} clip {h['clipfrac']:.3f} vl {h['vl']:.3f} sd {h['sd_goal']:.4f}/{h['sd_K']:.3f} | {h['t_it']:.1f}s"
        + (f" | r2 cost {rs['r2_cost']:.2f} held3 {rs['r2_held_start3']:.2f} r3 cost {rs['r3_cost']:.2f} top {rs['r3_top_window']:.2f}" if BETA > 0 else '')
        + (f" | cls {h['cls_bce']:.4f} acc {h['cls_acc']:.3f} c_ent {c_cont:.2e}" if I_MODE else '')
        + f" | EV orig {rs['ev_orig']:.3f}" + (f" r2 {rs['ev_r2']:.3f} r3 {rs['ev_r3']:.3f}" if BETA > 0 else '') + (' | critic warm-up' if warm else ''))
    json.dump(hist, open(os.path.join(OUT, 'hist.json'), 'w'))
    if it % HP['eval_every'] == 0 or it == IT0 + N_IT: evaluate(it)
log('done')
if 'group' in OPT: _done['ok'] = True
