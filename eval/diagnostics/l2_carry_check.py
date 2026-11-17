"""Carry in phase 2 (deterministic, episodes held through phase 2): where the brain aims (goal tip, nominal kinematics) and where the tip is, relative to p*."""
import json
REC = {}
import jax; jax.config.update('jax_enable_x64', True)
from pairl import paths
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
cen = np.array([0.55, 0.1]); ps = np.array([t['ps'] for p, t in ev]); Ln = np.asarray(P.L_NOM)
for spec in sys.argv[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    x = np.asarray(dec['x']); u = np.asarray(dec['u']); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = x[..., 21]; p = x[..., 19:21] * 0.3 + cen
    qd = np.asarray(P.QC) + np.asarray(P.QH) * np.tanh(u[..., :3]); th = np.cumsum(qd, -1); g = np.stack([np.sum(Ln * np.cos(th), -1), np.sum(Ln * np.sin(th), -1)], -1)
    held2 = np.all(~(v & (j == 1)) | (xi > 0.5), axis=1) & np.any(v & (j == 1), axis=1)
    e2 = np.cumsum(np.array([t['nph'] for pp, t in ev]), 1) // P.NB; kd = np.arange(j.shape[1])[None, :]
    late = v & (j == 1) & (kd >= e2[:, 1:2] - np.round(0.4 * (e2[:, 1:2] - e2[:, 0:1])))
    sel = late & held2[:, None]
    d_goal = np.linalg.norm(g - ps[:, None, :], axis=-1); d_tip = np.linalg.norm(p - ps[:, None, :], axis=-1); d_gt = np.linalg.norm(g - p, axis=-1)
    start = np.linalg.norm(p[np.arange(64), e2[:, 0]] - ps, axis=-1)
    REC[f'{run.split("/")[-1]}:{it}'] = dict(held_phase2=int(held2.sum()), goal_tip_to_target_late=float(np.median(d_goal[sel])) if sel.any() else None,
        tip_to_target_late=float(np.median(d_tip[sel])) if sel.any() else None, goal_tip_to_tip_late=float(np.median(d_gt[sel])) if sel.any() else None, tip_to_target_start=float(np.median(start[held2])) if held2.any() else None)
    print(f'{run.split("/")[-1]} it {it}: episodes held through phase 2: {held2.sum()}; last 40% of phase 2 (medians): goal tip to p* {np.median(d_goal[sel]):.3f} m, '
          f'tip to p* {np.median(d_tip[sel]):.3f} m, goal tip to tip {np.median(d_gt[sel]):.3f} m; tip to p* at the start of phase 2 {np.median(start[held2]):.3f} m')
json.dump(REC, open(str(paths.RECORDS / 'l2_carry_check.json'), 'w'), indent=1)
