"""Trajectory statistics of a checkpoint (descriptive; rollouts only, no training): deterministic rollouts of the 64 evaluation episodes from the original
start and from both restart starts. Reaching (distance and joint speed near the grasp point, crossings of the 2 cm ball, time of the grasp), outcome classes
and their cost, pressing (window force against the target, by class, and its regression on the two candidate forces), release in phase 4 (deterministic and
sampled gripper). Usage: python3 l2_traj_check.py OUT_JSON RUN:IT"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo

OUT, SPEC = sys.argv[1], sys.argv[2]; run, it = SPEC.rsplit(':', 1); env.COST_V = 4; cfg = C.default_cfg()
ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); l2.set_mode(ck); bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); batch = l2.make_batch_fn(cfg); TH = P.TH
med = lambda a: float(np.median(a)) if np.size(a) else None; mean = lambda a: float(np.mean(a)) if np.size(a) else None
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
R = {}
for start in (0, 2, 3):
    eps = ev if start == 0 else [env.restart(p, t, start) for p, t in ev]; pb, tb = env.stack(eps); keys = jax.random.split(jax.random.PRNGKey(555), 64)
    out = {}
    for mode, name in ((1.0, 'deterministic'), (0.0, 'sampled')):
        st, m, dec, rec = batch(pb, tb, keys, bp, hp, mode, 0.0); s = l2.summarize(st, m)
        v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']) > 0.5; x = np.asarray(dec['x'])
        q = np.asarray(dec['viz']['q']); p = np.asarray(tipf(jnp.asarray(q), pb['L'])); dq = np.abs(x[..., 3:6] * 2).max(-1)
        nph = np.asarray(tb['nph']) // P.NB; e0 = np.cumsum(nph, 1); ar = np.arange(j.shape[1])[None]
        o = dict(cost=mean(s['cost']), cost_median=med(s['cost']), cost_by_phase=np.asarray(s['cost_ph']).mean(0).tolist(), released=mean(s['released']))
        if start == 0:
            com, ops = np.asarray(dec['viz']['com']), np.asarray(dec['viz']['opsi']); lo = np.asarray(pb['Lobj'])[:, None, None]
            gp = com + lo * np.stack([np.cos(ops), np.sin(ops)], -1); d = np.linalg.norm(p - gp, axis=-1); free = v & ~xi
            p1 = free & (j == 0); inb = p1 & (d <= P.GRASP_DIST)
            t_in = np.where(inb, ar, 10 ** 6).min(1); reached = t_in < 10 ** 6
            cross = (np.diff((d <= P.GRASP_DIST).astype(int), axis=1) == -1) & p1[:, 1:]              # leaving the ball while free in phase 1
            tg = np.where(v & xi, ar, 10 ** 6).min(1); gph = np.where(tg < 10 ** 6, (tg[:, None] >= e0).sum(1), -1)   # phase of the first held decision
            end1 = np.clip(e0[:, 0] - 1, 0, None); d_end1 = d[np.arange(64), end1]
            o['reach'] = dict(phase1_length_median_s=med(nph[:, 0] * TH), reached_2cm_in_phase1=mean(reached),
                              time_to_2cm_median_s=med(t_in[reached] * TH), decisions_inside_2cm_median=med(inb.sum(1)[reached]),
                              speed_inside_2cm_median=med(dq[inb]), share_inside_2cm_slow=mean(dq[inb] <= P.GRASP_SPEED),
                              speed_free_phase1_median=med(dq[p1]), leaves_of_2cm_ball_mean=mean(cross.sum(1)[reached]),
                              distance_end_phase1_median_free=med(d_end1[gph != 0]), gripper_closed_inside_2cm=mean(np.asarray(dec['psi'])[inb]))
            cls = np.where(gph == 0, 0, np.where(gph == 1, 1, np.where(gph >= 2, 2, 3))); names = ('grasp_in_phase1', 'grasp_in_phase2', 'grasp_later', 'never')
            o['outcome'] = {n: dict(episodes=int((cls == c).sum()), cost=mean(s['cost'][cls == c]),
                                    cost_by_phase=(np.asarray(s['cost_ph'])[cls == c].mean(0).tolist() if (cls == c).any() else None),
                                    grasp_delay_into_phase2_median_s=(med((tg - e0[:, 0])[cls == c] * TH) if c == 1 else None)) for c, n in enumerate(names)}
        held3 = (v & xi & (j == 2)).any(1); fb = np.asarray(s['fbar']); fs = np.asarray(tb['fstar']); fl, fh = np.asarray(tb['f_light']), np.asarray(tb['f_heavy'])
        hv = np.asarray(tb['heavy']) > 0.5; c = held3 & (fb > 0)
        if c.sum() >= 8:
            A = np.stack([np.ones(c.sum()), fl[c], fh[c], hv[c].astype(float) * (fh[c] - fl[c])], 1); co = np.linalg.lstsq(A, fb[c], rcond=None)[0]
            o['press'] = dict(episodes_with_window_force=int(c.sum()), force_ratio_median_light=med((fb / fs)[c & ~hv]), force_ratio_median_heavy=med((fb / fs)[c & hv]),
                              n_light=int((c & ~hv).sum()), n_heavy=int((c & hv).sum()), corr_force_target=float(np.corrcoef(fb[c], fs[c])[0, 1]),
                              corr_force_mid=float(np.corrcoef(fb[c], 0.5 * (fl + fh)[c])[0, 1]),
                              regression_const_light_heavy_classgap=co.tolist(), relative_error_median=med(np.asarray(s['frel'])[c]),
                              relative_error_if_mid_force=med((np.abs(0.5 * (fl + fh) - fs) / fs)[c]), class_decision=mean(s['class_ok'][c]))
        p4 = v & (j == 3); held_at4 = (xi & p4).any(1); trel = np.where(p4 & ~xi, ar, 10 ** 6).min(1); rel = held_at4 & (trel < 10 ** 6)
        o['release'] = dict(held_at_phase4=int(held_at4.sum()), released_share=mean(rel[held_at4]), delay_median_s=med((trel - e0[:, 2])[rel] * TH),
                            phase4_length_median_s=med(nph[:, 3] * TH))
        out[name] = o
    R[f'start{start}'] = out; print(start, json.dumps(out), flush=True)
R['spec'] = SPEC.split('/')[-1]; json.dump(R, open(OUT, 'w'), indent=1)
