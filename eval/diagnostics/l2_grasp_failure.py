"""Grasp failure analysis (descriptive, not registered): deterministic rollouts of the 256 original-start evaluation episodes of pilot O (episodes and keys
of l2_pilotO_eval.py) for checkpoints given as RUN_DIR:IT. Phase-2 and phase-3 restarts begin with the object held, so grasping is only measured here.
Per episode: grasp at the end of phase 1; grasp point (radius from the base, height); distance of the tip to the grasp point at the end of phase 1 and its
minimum in phase 1; whether the tip came within 2 cm, slowly enough (joint speed <= 0.1 rad/s), with the gripper closed. Structural checks: margin of the
grasp point inside the joint box (kin_np.best_margin with the episode's link lengths; > 0 means reachable inside the box), the distance the brain would
have to cover per decision against the action range (|dp| <= 0.3 m), and the share of saturated displacement commands.
Usage: python eval/diagnostics/l2_grasp_failure.py OUT_JSON RUN_DIR:IT [...]   (best through script/parallel_run.py)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo
from pairl.stageb.kin_np import best_margin

OUT = sys.argv[1]; env.COST_V = 4; cfg = C.default_cfg(); N = 256
ev = l1.with_E0(l1.episodes(6000, N, lim=0.7), P.E_NOM); pb, tb = env.stack(ev); keys = jax.random.split(jax.random.PRNGKey(555), N)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
Ls = np.asarray(pb['L']); R = json.load(open(OUT)) if os.path.exists(OUT) else {}
med = lambda a: float(np.median(a)) if np.size(a) else None
for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, 1.0, 0.0); s = l2.summarize(st, m)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']) > 0.5; q = np.asarray(dec['viz']['q'])
    p = np.asarray(tipf(jnp.asarray(q), pb['L'])); com, ops = np.asarray(dec['viz']['com']), np.asarray(dec['viz']['opsi']); lo = np.asarray(pb['Lobj'])[:, None, None]
    gp = com + lo * np.stack([np.cos(ops), np.sin(ops)], -1); d = np.linalg.norm(p - gp, axis=-1)
    x = np.asarray(dec['x']); spd = np.abs(x[..., 3:6] * 2).max(-1); psi = np.asarray(dec['psi']) > 0.5; u = np.asarray(dec['u'])
    dp = l2.DP_MAX * np.tanh(u[..., :2]); sat = (np.abs(np.tanh(u[..., :2])) > 0.95).any(-1)
    p1 = v & (j == 0); free1 = p1 & ~xi; ar = np.arange(N); last1 = np.maximum(p1.sum(1) - 1, 0)
    grasp = np.asarray(s['grasp']) > 0.5
    d_end1 = d[ar, last1]; dmin1 = np.where(free1, d, 9.0).min(1)
    inb = free1 & (d <= P.GRASP_DIST); slow = inb & (spd <= P.GRASP_SPEED); closed = slow & psi
    g0 = gp[:, 0]; rad = np.linalg.norm(g0, axis=-1); hgt = g0[:, 1]
    margin = np.array([best_margin(g0[i], Ls[i], P.QC, P.QH)[0] for i in range(N)])
    cat = np.where(grasp, 'grasped', np.where(~inb.any(1), 'never_within_2cm', np.where(~slow.any(1), 'within_2cm_never_slow',
                   np.where(~closed.any(1), 'slow_within_2cm_gripper_open', 'other'))))
    bins = lambda a: np.quantile(a, [1 / 3, 2 / 3])
    rb, hb = bins(rad), bins(hgt)
    by = lambda a, b: [dict(range=[float(lo_), float(hi_)], n=int(((a >= lo_) & (a < hi_)).sum()), grasp=float(grasp[(a >= lo_) & (a < hi_)].mean()))
                       for lo_, hi_ in zip([-9, *b], [*b, 9])]
    near = free1 & (d > 0.02) & (d < 0.1)
    o = dict(n=N, grasp_end_phase1=float(grasp.mean()), categories={c: int((cat == c).sum()) for c in np.unique(cat)},
             by_radius=by(rad, rb), by_height=by(hgt, hb),
             tip_to_grasp_point_end_phase1=dict(median=med(d_end1), grasped=med(d_end1[grasp]), not_grasped=med(d_end1[~grasp]),
                                                not_grasped_share_above_0p05=float((d_end1[~grasp] > 0.05).mean()) if (~grasp).any() else None),
             min_dist_phase1=dict(median=med(dmin1), not_grasped=med(dmin1[~grasp])),
             structural=dict(grasp_point_margin_in_joint_box=dict(min=float(margin.min()), median=med(margin), share_unreachable=float((margin <= 0).mean()),
                                                                  not_grasped_min=float(margin[~grasp].min()) if (~grasp).any() else None),
                             start_distance_median=med(d[:, 0]), start_distance_max=float(d[:, 0].max()),
                             share_phase1_free_decisions_with_distance_above_0p3=float((d[free1] > 0.3).mean()),
                             phase1_decisions_median=med(p1.sum(1)),
                             share_phase1_free_decisions_saturated_dp=float(sat[free1].mean())),
             near_2_10cm=dict(decisions=int(near.sum()), dp_norm_median=med(np.linalg.norm(dp, axis=-1)[near]),
                              dp_over_distance_median=med((np.linalg.norm(dp, axis=-1) / d)[near]),
                              dp_toward_cos_median=med(((dp * (gp - p)).sum(-1) / (np.linalg.norm(dp, axis=-1) * d + 1e-9))[near]),
                              speed_median=med(spd[near])),
             inside_2cm=dict(decisions=int(inb.sum()), speed_median=med(spd[inb]), slow_share=float((spd[inb] <= P.GRASP_SPEED).mean()) if inb.any() else None,
                             gripper_closed_share=float(psi[inb].mean()) if inb.any() else None, dp_norm_median=med(np.linalg.norm(dp, axis=-1)[inb])),
             device=str(jax.devices()[0]))
    key = f"{run.rstrip('/').split('/')[-1]}:{it}"; R[key] = o; print(key, json.dumps(o), flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)
