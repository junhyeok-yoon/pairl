"""Component dissection of an L2 checkpoint on the 64 evaluation episodes. Usage: python3 l2_dissect.py RUN_DIR IT
Sections: deterministic policy (reach, aim, gripper, stiffness, task, ledger use, press, hand error, hand features); exploring policy (grasp and release
events, gripper and goal standard deviation by phase, critic explained variance); channel test (L1 reference brain driven with the learned hand).
Optional cost=2: rewards of cost version 2 (only the critic's explained variance depends on it)."""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, model as Mo, controller as C, params as P

RUN, IT = sys.argv[1], int(sys.argv[2]); OPT = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a); env.COST_V = int(OPT.get('cost', 1))
ck = pickle.load(open(f'{RUN}/params_it{IT:04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp']); l2.set_mode(ck)
goals = lambda uu, qq: np.asarray(jax.vmap(jax.vmap(l2.goal_of))(jnp.asarray(uu), jnp.asarray(qq)))   # goal postures of actions at the measured postures
cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM)
bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
cen = np.array([0.55, 0.1])
def unpack(dec):
    """Layout of l2.obs_vec."""
    x = np.asarray(dec['x'])
    return dict(x=x, v=np.asarray(dec['valid']) > 0.5, u=np.asarray(dec['u']), psi=np.asarray(dec['psi']), Kp=np.asarray(dec['Kp']), j=np.asarray(dec['j']),
                ph=x[..., 19:21] * 0.3 + cen, o=x[..., 13:15] * 0.3 + cen, dq=x[..., 3:6] * 2, xi=x[..., 21], te=x[..., 6:9] * 10)
def heads(x):
    mu, ls, lg = jax.vmap(jax.vmap(lambda xx: l2.policy(bp, xx)))(jnp.asarray(x)); return np.asarray(mu), np.asarray(ls), np.asarray(lg)
def class_given_contact(s):
    c = (s['fbar'] > 0) & (s['grasp'] > 0.5)
    return (float(s['class_ok'][c].mean()) if c.any() else None), int(c.sum())
def med(a): return float(np.median(a)) if np.size(a) else None
R = {}

# deterministic policy
st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
s = l2.summarize(st, m); D = unpack(dec); x, v, u, psi, Kp, j, ph, o, dq, xi = (D[k] for k in ('x', 'v', 'u', 'psi', 'Kp', 'j', 'ph', 'o', 'dq', 'xi'))
q = x[..., 0:3] * np.asarray(P.QH) + np.asarray(P.QC); qdp = goals(u, q)
p1 = v & (j == 0) & (xi < 0.5); dist = np.linalg.norm(ph - o, axis=-1); spd = np.max(np.abs(dq), axis=-1)
dmin = np.where(p1, dist, np.inf).min(1); imin = np.where(p1, dist, np.inf).argmin(1); smin = spd[np.arange(64), imin]
R['reach'] = dict(min_dist_median=float(np.median(dmin)), frac_within_2cm=float((dmin <= P.GRASP_DIST).mean()), frac_slow_when_nearest=float((smin <= P.GRASP_SPEED).mean()),
                  grasp=float(s['grasp'].mean()), n_grasp_mean=float(s['n_grasp'].mean()))
L = np.asarray(P.L_NOM); th = np.cumsum(qdp, -1); tipg = np.stack([np.sum(L * np.cos(th), -1), np.sum(L * np.sin(th), -1)], -1)
R['aim'] = dict(goal_tip_to_grasp_point_median=float(np.median(np.linalg.norm(tipg - o, axis=-1)[p1])), goal_tip_to_tip_median=float(np.median(np.linalg.norm(tipg - ph, axis=-1)[p1])))
R['gripper_closed_by_phase'] = [float(psi[v & (j == k)].mean()) for k in range(4)]
R['stiffness_mean_by_phase'] = [np.round(Kp[v & (j == k)].mean(0), 1).tolist() for k in range(4)]
cg, ncg = class_given_contact(s)
R['task'] = dict(cost=float(s['cost'].mean()), cost_by_phase=np.round(s['cost_ph'].mean(0), 3).tolist(), hold=float(np.median(s['pos2'])), frel=float(np.median(s['frel'])),
                 class_given_contact=cg, n_contact=ncg, released=float(s['released'].mean()), mis_hold=float(np.median(s['mis_hold'])))
xb = x.copy(); xb[..., 24:27] = 0.0
mu, _, _ = heads(x); mub, _, _ = heads(xb)
dg = np.linalg.norm(goals(mu, q) - goals(mub, q), axis=-1)
R['goal_shift_from_learned_load_by_phase'] = [med(dg[v & (j == k) & (xi > 0.5)]) for k in range(4)]
p3 = v & (j == 2) & (xi > 0.5); ye = np.array([p['ye'] for p, t in ev]); psy = np.array([t['ps'][1] for p, t in ev])
hmin = np.where(p3, ph[..., 1], np.inf).min(1) - ye; touched = np.any(p3 & (np.linalg.norm(D['te'], axis=-1) > cfg['tau_th']), axis=1); gg = s['grasp'] > 0.5
R['press'] = dict(n_grasped=int(gg.sum()), frac_touched_given_grasp=float(touched[gg].mean()) if gg.any() else None,
                  min_height_above_block_median=med(hmin[gg & np.isfinite(hmin)]), drop_median=med((psy - ye)[gg]))
# press detail among episodes held during phase 3: contact timing within the phase, goal depth below the top face, lateral offset
e3 = np.cumsum(np.array([t['nph'] for p, t in ev]), 1) // P.NB; held3 = p3.any(1); kd = np.arange(j.shape[1])[None, :]
late = p3 & (kd >= (e3[:, 2:3] - np.round(0.4 * (e3[:, 2:3] - e3[:, 1:2]))))
con = np.linalg.norm(D['te'], axis=-1) > cfg['tau_th']; gy = tipg[..., 1] - ye[:, None]
R['press_detail'] = dict(n_held_phase3=int(held3.sum()), contact_frac_early=float(con[p3 & ~late].mean()) if np.any(p3 & ~late) else None,
                         contact_frac_late=float(con[late].mean()) if late.any() else None, goal_depth_median_early=med(gy[p3 & ~late]), goal_depth_median_late=med(gy[late]),
                         tip_height_median_late=med((ph[..., 1] - ye[:, None])[late]), lateral_offset_median_late=med(np.abs(ph[..., 0] - psy[:, None] * 0 - np.array([t['ps'][0] for p, t in ev])[:, None])[late]),
                         stiffness_mean_late=np.round(Kp[late].mean(0), 1).tolist() if late.any() else None)
# face of the block contact at held phase-3 decisions, and the stop-rule share (held with top-face force in the force window)
VZ = {k: np.asarray(a) for k, a in dec['viz'].items()}; fnd = VZ['fn']; side_f = (np.abs(VZ['fe'][..., 0]) > 0.05) & (fnd <= 0)
R['contact_face'] = dict(top_share_held_phase3=float(np.mean(fnd[p3] > 0)) if p3.any() else None, side_share_held_phase3=float(np.mean(side_f[p3])) if p3.any() else None,
                         share_episodes_held_top_force_window=float(np.mean(np.any(late & (fnd > 0), axis=1))))
# carry: episodes held through phase 2; goal tip and tip against p* in the last 40% of phase 2, tip against p* at the start of phase 2
e2 = np.cumsum(np.array([t['nph'] for pp_, t in ev]), 1) // P.NB; psx = np.array([t['ps'] for pp_, t in ev])
held2 = np.all(~(v & (j == 1)) | (xi > 0.5), axis=1) & np.any(v & (j == 1), axis=1)
late2 = v & (j == 1) & (kd >= e2[:, 1:2] - np.round(0.4 * (e2[:, 1:2] - e2[:, 0:1]))) & held2[:, None]
R['carry'] = dict(n_held_phase2=int(held2.sum()), goal_tip_to_target_late=med(np.linalg.norm(tipg - psx[:, None, :], axis=-1)[late2]),
                  tip_to_target_late=med(np.linalg.norm(ph - psx[:, None, :], axis=-1)[late2]),
                  tip_to_target_start=med(np.linalg.norm(ph[np.arange(64), np.minimum(e2[:, 0], j.shape[1] - 1)] - psx, axis=-1)[held2]))
def replay_err(hp, r):
    def step(z, xx):
        G = l2.hand_G(hp, xx['q']); uad = G.T @ (l2.abar_of(hp) + z); e = jnp.linalg.norm(uad - xx['ustar'])
        mh = uad - xx['taup'] - xx['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
        return jnp.clip(z - P.TS * cfg['lam_s'] * xx['zeta'] * (G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mh)), -cfg['zbar'], cfg['zbar']), e
    return jax.lax.scan(step, jnp.zeros(l2.NA), r)[1]
flat = jax.tree_util.tree_map(lambda a: jnp.asarray(a).reshape((a.shape[0], -1) + a.shape[3:]), rec)
err = np.asarray(jax.jit(jax.vmap(lambda r: replay_err(hp, r)))(flat)); zeta = np.asarray(flat['zeta']); val = np.asarray(flat['val']) > 0.5
jstep = np.repeat(j, P.NB, axis=1)
R['hand_error_by_phase'] = {f'phase{k + 1}': dict(rest=med(err[val & (jstep == k) & (zeta > 0.5)]), motion=med(err[val & (jstep == k) & (zeta < 0.5)])) for k in range(4)}
hd = l2.hand_diagnostics(hp, ev); R['hand'] = {k: float(np.median(a)) for k, a in hd.items()}

# exploring policy (as in training), two noise draws per episode
EX = dict(grasp=[], n_grasp=[], rel=[], acc_rel=[], touched=[], att3=[]); popen = [[], [], [], []]; sdg = [[], [], [], []]; sdd = [[], [], [], []]; sdk = [[], [], [], []]; near_sd = [[], [], []]; evr = [[], [], [], []]
gae_b = jax.jit(jax.vmap(lambda r, vv, d: l2.gae(r, vv, d, 0.99, 0.95)))
for rep in range(2):
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(900 + rep), 64), bp, hp, 0.0, 0.0)
    s = l2.summarize(st, m); D = unpack(dec); x, v, j, xi, ph, o = D['x'], D['v'], D['j'], D['xi'], D['ph'], D['o']
    _, ls, lg = heads(x); sd = np.exp(ls)
    dxi = np.diff(xi, axis=1); relev = (dxi < -0.5) & v[:, :-1]; jj = j[:, :-1]
    EX['grasp'] += list(s['grasp']); EX['n_grasp'] += list(s['n_grasp']); EX['rel'].append([relev[jj == k].sum() / 64 for k in range(4)])
    EX['acc_rel'] += list(np.any(relev & (jj <= 2), axis=1))                    # release before the release phase
    p3a = v & (j == 2) & (xi > 0.5); EX['att3'] += list(p3a.any(1)); EX['touched'] += list(np.any(p3a & (np.linalg.norm(D['te'], axis=-1) > cfg['tau_th']), axis=1))
    po_ = 1 / (1 + np.exp(lg))
    for k in range(4):
        sel = v & (j == k); popen[k] += list(po_[sel & (xi > 0.5)]); sdg[k] += list(sd[sel][:, :3].mean(-1)); sdd[k] += list(sd[sel][:, :3]); sdk[k] += list(sd[sel][:, 3:].mean(-1))
    dd = np.linalg.norm(ph - o, axis=-1); s1 = v & (j == 0) & (xi < 0.5)
    for b, (lo, hi) in enumerate(((0, 0.02), (0.02, 0.05), (0.05, 9))): near_sd[b] += list(sd[s1 & (dd >= lo) & (dd < hi)][:, :3].mean(-1))
    adv = np.asarray(gae_b(dec['r'], dec['v'], dec['valid'])); ret = adv + np.asarray(dec['v'])
    for k in range(4):
        sel = v & (j == k); evr[k].append(1 - np.var(ret[sel] - np.asarray(dec['v'])[sel]) / max(np.var(ret[sel]), 1e-12))
att3 = np.asarray(EX['att3'])
R['explore'] = dict(grasp=float(np.mean(EX['grasp'])), n_grasp_mean=float(np.mean(EX['n_grasp'])), releases_per_episode_by_phase=np.round(np.mean(EX['rel'], 0), 3).tolist(),
                    frac_release_before_phase4=float(np.mean(EX['acc_rel'])), frac_attached_in_phase3=float(att3.mean()),
                    frac_touched_given_attached_phase3=float(np.asarray(EX['touched'])[att3].mean()) if att3.any() else None,
                    p_open_while_attached_by_phase=[float(np.mean(a)) if a else None for a in popen],
                    sd_goal_by_phase=[float(np.mean(a)) for a in sdg], sd_goal_by_dim_by_phase=[np.mean(a, 0).tolist() if a else None for a in sdd], sd_K_by_phase=[float(np.mean(a)) for a in sdk],
                    sd_goal_phase1_by_distance={'<2cm': float(np.mean(near_sd[0])) if near_sd[0] else None, '2-5cm': float(np.mean(near_sd[1])) if near_sd[1] else None,
                                                '>5cm': float(np.mean(near_sd[2])) if near_sd[2] else None},
                    critic_explained_variance_by_phase=[float(np.mean(a)) for a in evr])

# channel test: the L1 reference brain driven with the learned hand, against the L1 hand (same episodes and noise)
REF = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, noqd=1.0, lead_t=0.15, k_c=1.0, f_mode=1.0)
cfgj = {k: (jnp.asarray(a) if not isinstance(a, (int, float)) else a) for k, a in cfg.items()}; optj = {k: (jnp.asarray(a) if not isinstance(a, (int, float)) else a) for k, a in REF.items()}
CH = {}
for tag, mf in (('learned_hand', l2.memfn_of(hp)), ('l1_hand', None)):
    fn = jax.jit(jax.vmap(lambda p, t, k: l1.rollout(p, t, k, l2.ND, cfgj, 'ref', optj, memfn=mf)))
    st, cs, m, _ = fn(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64)); s1 = l1.summarize(st, m)
    c = (np.asarray(s1['fbar']) > 0) & (np.asarray(s1['grasp']) > 0.5)
    CH[tag] = dict(cost=float(np.mean(s1['cost'])), grasp=float(np.mean(s1['grasp'])), hold=float(np.median(s1['pos2'])), frel=float(np.median(np.asarray(s1['frel'])[c])) if c.any() else None,
                   class_given_contact=float(np.mean(np.asarray(s1['class_ok'])[c])) if c.any() else None, n_contact=int(c.sum()), mis_hold=float(np.median(s1['mis_hold'])),
                   viol=float(np.sum(s1['viol'])))
R['channel_reference_brain'] = CH
print(json.dumps(R, indent=1)); json.dump(R, open(f'{RUN}/dissect_it{IT:04d}.json', 'w'), indent=1)
