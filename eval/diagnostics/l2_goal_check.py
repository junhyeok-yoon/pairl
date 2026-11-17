"""(writes records/l2_goal_check.json) How far the commanded goal tip is from the tip of the zero-action posture (box center), and the spread of goals across episodes, phase 1, deterministic."""
import jax; jax.config.update('jax_enable_x64', True)
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
Ln = np.asarray(P.L_NOM); th0 = np.cumsum(np.asarray(P.QC)); tip0 = np.array([np.sum(Ln * np.cos(th0)), np.sum(Ln * np.sin(th0))])
cen = np.array([0.55, 0.1]); import json, os
OUTF = str(paths.RECORDS / 'l2_goal_check.json'); REC = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for spec in sys.argv[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    u = np.asarray(dec['u']); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); x = np.asarray(dec['x']); o = x[..., 13:15] * 0.3 + cen
    qd = np.asarray(P.QC) + np.asarray(P.QH) * np.tanh(u[..., :3]); th = np.cumsum(qd, -1); g = np.stack([np.sum(Ln * np.cos(th), -1), np.sum(Ln * np.sin(th), -1)], -1)
    s1 = v & (j == 0); last = np.array([np.where(s1[e])[0].max() for e in range(64)])
    gl = g[np.arange(64), last]; ol = o[np.arange(64), last]; psv = np.array([t['ps'] for p_, t in ev])
    REC[f"{run.split('/')[-1]}:{it}"] = dict(goal_to_default=float(np.median(np.linalg.norm(gl - tip0, axis=-1))), goal_to_grasp_point=float(np.median(np.linalg.norm(gl - ol, axis=-1))),
        goal_spread=float(np.linalg.norm(gl.std(0))), grasp_point_spread=float(np.linalg.norm(ol.std(0))),
        goal_to_carry_target=float(np.median(np.linalg.norm(gl - psv, axis=-1))), share_goal_closer_to_carry_target=float(np.mean(np.linalg.norm(gl - psv, axis=-1) < np.linalg.norm(gl - ol, axis=-1))))
    print(f"{run.split('/')[-1]} it {it}: end of phase 1, goal tip to the zero-action tip median {np.median(np.linalg.norm(gl - tip0, axis=-1)):.3f} m, "
          f"goal tip to the grasp point median {np.median(np.linalg.norm(gl - ol, axis=-1)):.3f} m, spread of goal tips across episodes (std) {np.linalg.norm(gl.std(0)):.3f} m, spread of grasp points {np.linalg.norm(ol.std(0)):.3f} m")
json.dump(REC, open(OUTF, 'w'), indent=1)
