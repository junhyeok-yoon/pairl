"""Grasp design study (descriptive, not registered; no training): why grasps of high objects fail, and audits of two candidate changes of handoff section 8,
item 4 (a phase-1 speed term near the object; phase-1 start states near the object). Rollouts of the 256 original-start evaluation episodes of pilot O for
checkpoints RUN_DIR:IT (deterministic, and fully sampled for the training exposure), and of 256 near-object starts built here (analysis only; the
environment is unchanged). Usage: python eval/diagnostics/l2_grasp_design.py OUT_JSON RUN_DIR:IT [...]   (best through script/parallel_run.py)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo
from pairl.stageb.kin_np import best_margin, ik_all, gauge

OUT = sys.argv[1]; env.COST_V = 4; cfg = C.default_cfg(); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}; N = 256
ev = l1.with_E0(l1.episodes(6000, N, lim=0.7), P.E_NOM); keys = jax.random.split(jax.random.PRNGKey(555), N)
UB = np.asarray(P.UB); med = lambda a: float(np.median(a)) if np.size(a) else None; R = json.load(open(OUT)) if os.path.exists(OUT) else {}
HB = (0.121, 0.248)                                                         # height tertiles of the grasp points (records/l2_grasp_failure.json)
jacf = jax.jit(jax.vmap(jax.vmap(Mo.jac, in_axes=(0, None)), in_axes=(0, 0))); tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))


def hand_err(hp, rec):
    """|u_ad - u*| at every control step, with the rest-gated memory replayed on the recorded trajectory (as l2.replay_loss)."""
    def one(r):
        def step(z, x):
            G = l2.hand_G(hp, x['q']); uad = G.T @ (l2.abar_of(hp) + z)
            mhat = uad - x['taup'] - x['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
            dz = G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mhat)
            return jnp.clip(z - P.TS * cfgj['lam_s'] * x['zeta'] * dz, -cfgj['zbar'], cfgj['zbar']), jnp.linalg.norm(uad - x['ustar'])
        return jax.lax.scan(step, jnp.zeros(l2.NA), r)[1]
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec)
    return jax.jit(jax.vmap(one))(flat)


def rollout(bp, hp, eps, mode):
    pb, tb = env.stack(eps); st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, keys[:len(eps)], bp, hp, mode, 0.0); s = l2.summarize(st, m)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']) > 0.5; q = np.asarray(dec['viz']['q'])
    p = np.asarray(tipf(jnp.asarray(q), pb['L'])); com, ops = np.asarray(dec['viz']['com']), np.asarray(dec['viz']['opsi']); lo = np.asarray(pb['Lobj'])[:, None, None]
    gp = com + lo * np.stack([np.cos(ops), np.sin(ops)], -1); d = np.linalg.norm(p - gp, axis=-1); dq = np.asarray(dec['x'])[..., 3:6] * 2
    vt = np.linalg.norm(np.einsum('ekij,ekj->eki', np.asarray(jacf(jnp.asarray(q), pb['L'])), dq), axis=-1)
    he = np.asarray(hand_err(hp, rec)).reshape(len(eps), q.shape[1], -1).mean(-1)                       # mean |u_ad - u*| over each decision interval
    us = np.asarray(rec['ustar']).reshape(len(eps), q.shape[1], -1, 3).mean(2); load = (np.abs(us) / UB).max(-1)   # gravity-load ratio (max over joints)
    gg = gauge(q, P.QC, P.QH); Kp = np.asarray(dec['Kp']); dp = l2.DP_MAX * np.tanh(np.asarray(dec['u'])[..., :2])
    return dict(s=s, v=v, j=j, xi=xi, d=d, dq=np.abs(dq).max(-1), vt=vt, he=he, load=load, gauge=gg, Kp=Kp, dp=np.linalg.norm(dp, axis=-1), gp=gp, psi=np.asarray(dec['psi']) > 0.5)


def near_starts(seed=0):
    """Phase-1 starts with the tip 2-8 cm from the grasp point at rest: posture by inverse kinematics (best gauge inside the box) of a point on that ring."""
    rng = np.random.default_rng(seed); out, margins = [], []
    for par, task in ev:
        po = np.asarray(task['po']); L = np.asarray(par['L'])
        for _ in range(20):
            r, a = rng.uniform(0.02, 0.08), rng.uniform(-np.pi, np.pi); x = po + r * np.array([np.cos(a), np.sin(a)])
            mg, qn = best_margin(x, L, P.QC, P.QH)
            if mg > 0 and x[1] > par['ye'] + P.CLEAR: break
        t = dict(task); t['q0'] = np.asarray(qn, np.float64); out.append((par, t)); margins.append(float(mg))
    return out, np.asarray(margins)


for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp')); o = {}
    for mode, mname in ((1.0, 'deterministic'), (0.0, 'fully_sampled')):
        D = rollout(bp, hp, ev, mode); free1 = D['v'] & (D['j'] == 0) & ~D['xi']; h0 = D['gp'][:, 0, 1]; grasp = np.asarray(D['s']['grasp']) > 0.5
        near, inb = free1 & (D['d'] < 0.1), free1 & (D['d'] < 0.02); ring = free1 & (D['d'] < 0.05)
        bins = []
        for lo_, hi_ in zip((-9, *HB), (*HB, 9)):
            e = (h0 >= lo_) & (h0 < hi_); ne, ie = near & e[:, None], inb & e[:, None]
            bins.append(dict(height=[lo_, hi_], n=int(e.sum()), grasp=float(grasp[e].mean()), decisions_within_10cm=int(ne.sum()), decisions_within_2cm=int(ie.sum()),
                             tip_speed_within_10cm=med(D['vt'][ne]), joint_speed_within_10cm=med(D['dq'][ne]), joint_speed_within_2cm=med(D['dq'][ie]),
                             slow_share_within_2cm=float((D['dq'][ie] <= P.GRASP_SPEED).mean()) if ie.any() else None,
                             hand_error_within_10cm=med(D['he'][ne]), gravity_load_ratio_within_10cm=med(D['load'][ne]), joint_gauge_within_10cm=med(D['gauge'][ne]),
                             stiffness_within_10cm=np.median(D['Kp'][ne], 0).round(1).tolist() if ne.any() else None, dp_within_10cm=med(D['dp'][ne]),
                             min_dist_median=med(np.where(free1 & e[:, None], D['d'], 9).min(1)[e])))
        # audit of a phase-1 speed term c_v = kappa * max(0, 1 - d / 0.05) * min(max(0, |dq|_inf - 0.1) / 0.1, 3) on the visited decision states
        sv = np.maximum(0, 1 - D['d'] / 0.05) * np.minimum(np.maximum(0, D['dq'] - P.GRASP_SPEED) / P.GRASP_SPEED, 3.0)
        base = D['d'] / 0.1; aud = {}
        for kappa in (0.25, 0.5, 1.0):
            add = kappa * sv * free1
            aud[f'kappa_{kappa}'] = dict(share_of_phase1_free_decisions_with_term=float((add[free1] > 0).mean()), added_rate_mean_where_active=med(add[add > 0]),
                                         added_over_distance_rate_phase1=float(add[free1].sum() / base[free1].sum()),
                                         added_per_episode_mean=float((add * P.TH).sum(1).mean()), max_rate_inside_ring=float(kappa * 3.0), distance_rate_at_ring=0.5)
        o[mname] = dict(grasp=float(grasp.mean()), by_height=bins, decisions_within_5cm_per_episode=float(ring.sum() / N), decisions_within_2cm_per_episode=float(inb.sum() / N),
                        grasps_per_episode=float(grasp.mean()), speed_term_audit=aud)
    S, margins = near_starts(); D = rollout(bp, hp, S, 1.0); grasp = np.asarray(D['s']['grasp']) > 0.5; free1 = D['v'] & (D['j'] == 0) & ~D['xi']; inb = free1 & (D['d'] < 0.02)
    o['near_starts'] = dict(n=N, start_margin_min=float(margins.min()), start_margin_median=med(margins), start_inside_box_share=float((margins > 0).mean()),
                            grasp=float(grasp.mean()), decisions_within_2cm_per_episode=float(inb.sum() / N),
                            slow_share_within_2cm=float((D['dq'][inb] <= P.GRASP_SPEED).mean()) if inb.any() else None, joint_speed_within_2cm=med(D['dq'][inb]),
                            phase1_cost_mean=float(np.asarray(D['s']['cost_ph'])[:, 0].mean()))
    o['device'] = str(jax.devices()[0]); key = f"{run.rstrip('/').split('/')[-1]}:{it}"; R[key] = o
    print(key, json.dumps({k: (v if k != 'deterministic' and k != 'fully_sampled' else {kk: vv for kk, vv in v.items() if kk != 'by_height'}) for k, v in o.items()}), flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)
