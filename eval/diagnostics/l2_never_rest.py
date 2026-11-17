"""Who never grasps, and how much the arm rests (descriptive; deterministic rollouts of the 64 evaluation episodes, no training): for attempt 3 at 300, the
episodes by grasp outcome (nearest distance, time inside the 2 cm ball, place of the grasp point), joint speed and commanded displacement by distance; for
the arms of pilot T at 100, the share of rest steps by phase. Usage: python3 l2_never_rest.py OUT_JSON"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl import paths
from pairl.stageb import env, controller as C, params as P, model as Mo
env.COST_V = 4; cfg = C.default_cfg(); OUT = sys.argv[1]
ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); pb, tb = env.stack(ev); keys = jax.random.split(jax.random.PRNGKey(555), 64)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0))); med = lambda a: round(float(np.median(a)), 4) if np.size(a) else None
out = {}
for run, it in (('l2_attempt3', 300), ('l2_pilotT_T', 100), ('l2_pilotT_J', 100)):
    ck = pickle.load(open(paths.run_dir(run) / f'params_it{it:04d}.pkl', 'rb')); l2.set_mode(ck); bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, 1.0, 0.0)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']) > 0.5
    zeta = np.asarray(rec['zeta']).reshape(64, j.shape[1], -1); val = np.asarray(rec['val']).reshape(64, j.shape[1], -1)
    dqc = np.abs(np.asarray(rec['dq'])).max(-1).reshape(64, j.shape[1], -1) if 'dq' in rec else None
    o = dict(mode=l2.ACT_MODE, rest_share_by_phase=[round(float((zeta * val)[(j == k) & v].sum() / max(val[(j == k) & v].sum(), 1)), 3) for k in range(4)],
             rest_share_free_phase1=round(float((zeta * val)[(j == 0) & v & ~xi].sum() / max(val[(j == 0) & v & ~xi].sum(), 1)), 3),
             rest_share_held=round(float((zeta * val)[v & xi].sum() / max(val[v & xi].sum(), 1)), 3))
    if run == 'l2_attempt3':
        x = np.asarray(dec['x']); q = np.asarray(dec['viz']['q']); p = np.asarray(tipf(jnp.asarray(q), pb['L'])); dq = np.abs(x[..., 3:6] * 2).max(-1)
        com, ops = np.asarray(dec['viz']['com']), np.asarray(dec['viz']['opsi']); lo = np.asarray(pb['Lobj'])[:, None, None]
        gp = com + lo * np.stack([np.cos(ops), np.sin(ops)], -1); d = np.linalg.norm(p - gp, axis=-1); free = v & ~xi
        never = ~(v & xi).any(1); ph1 = (xi & v & (j == 0)).any(1); late = ~never & ~ph1
        dmin = np.where(free, d, 9).min(1); inb = free & (d <= P.GRASP_DIST); nin = inb.sum(1); smin = np.where(inb, dq, 9).min(1)
        po = np.asarray(tb['po']); r_o = np.linalg.norm(gp[:, 0], axis=-1); mo = np.asarray(pb['mo']); ps = np.asarray(tb['ps'])
        for name, s in (('never', never), ('late', late), ('phase1', ph1)):
            o[name] = dict(n=int(s.sum()), min_dist_median=med(dmin[s]), min_dist_p75=round(float(np.percentile(dmin[s], 75)), 4), share_ever_inside_2cm=round(float((nin[s] > 0).mean()), 3),
                           decisions_inside_median=med(nin[s]), min_speed_inside_median=med(smin[s][nin[s] > 0]), grasp_point_radius_median=med(r_o[s]),
                           grasp_point_height_median=med(gp[s, 0, 1]), grasp_point_x_median=med(gp[s, 0, 0]), object_mass_median=med(mo[s]),
                           dist_end_episode_median=med(d[np.arange(64), np.maximum(v.sum(1) - 1, 0)][s]))
        # speed profile while free, by distance band
        for a, b in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)):
            sel = free & (d >= a) & (d < b); o[f'speed_free_{a}-{b}m'] = dict(n=int(sel.sum()), speed_median=med(dq[sel]), slow_share=round(float((dq[sel] <= P.GRASP_SPEED).mean()), 4))
        # goal displacement commanded while inside the ball (deterministic action)
        u = np.asarray(dec['u']); dp = 0.3 * np.tanh(u[..., :2]); o['dp_norm_inside_ball_median_m'] = med(np.linalg.norm(dp, axis=-1)[inb]); o['dp_norm_far_median_m'] = med(np.linalg.norm(dp, axis=-1)[free & (d > 0.1)])
        rel = gp - p; cosang = (dp * rel).sum(-1) / (np.linalg.norm(dp, axis=-1) * np.linalg.norm(rel, axis=-1) + 1e-9)
        o['dp_toward_object_cos_median_2-10cm'] = med(cosang[free & (d > 0.02) & (d < 0.1)]); o['dp_norm_2-10cm_median_m'] = med(np.linalg.norm(dp, axis=-1)[free & (d > 0.02) & (d < 0.1)])
        o['dp_over_distance_ratio_2-10cm_median'] = med((np.linalg.norm(dp, axis=-1) / d)[free & (d > 0.02) & (d < 0.1)])
    out[f'{run}:{it}'] = o; print(run, it, json.dumps(o), flush=True)
json.dump(out, open(OUT, 'w'), indent=1); print('NRDONE')
