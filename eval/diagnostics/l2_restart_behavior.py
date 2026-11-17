"""Behavior of a checkpoint from the restart states (pilot R, descriptive): the 64 evaluation episodes started at phase 2 and at phase 3 with the object held,
deterministic policy, key 555. Phase-3 start: contact in the force window (last 40% of phase 3), force ratio in contact, tip height above the top face and lateral
offset from p_c in the window, releases in phase 3, and the response of the goal to the contact force (least-squares slope of the goal-tip height above the face on
f/f* over held contact decisions). Phase-2 start: held at the start of phase 3, hold error and goal tip at p* at the end of phase 2, releases in phase 2.
Usage: python3 l2_restart_behavior.py OUT_JSON RUN:IT [...] [cost=4]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, model as Mo, params as P

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = POS[0]; REC = json.load(open(OUT)) if os.path.exists(OUT) else {}
env.COST_V = int(OPT.get('cost', 4)); cfg = C.default_cfg(); BF = {}
ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
med = lambda a: float(np.median(a)) if np.size(a) else None
def run(bp, hp, kind):
    eps = [env.restart(p, t, kind) for p, t in ev]; pb, tb = env.stack(eps)
    bf = BF.setdefault(l2.ACT_MODE, l2.make_batch_fn(cfg))
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']); fn = np.asarray(dec['viz']['fn'])
    p = np.asarray(tipf(dec['viz']['q'], pb['L'])); g = np.asarray(tipf(jax.vmap(jax.vmap(l2.goal_of))(dec['u'], dec['x'][..., 0:3] * l2.QH + l2.QC), pb['L']))
    e = np.cumsum(np.array([t['nph'] for p_, t in eps]), 1) // P.NB; kd = np.arange(j.shape[1])[None]
    ps = np.asarray(tb['ps']); ye = np.asarray(pb['ye'])[:, None]; fs = np.asarray(tb['fstar'])[:, None]
    rel = lambda ph: np.any((np.diff(xi, axis=1) < -0.5) & v[:, :-1] & (j[:, :-1] == ph), 1)
    return dict(v=v, j=j, xi=xi, fn=fn, p=p, g=g, e=e, kd=kd, ps=ps, ye=ye, fs=fs, rel=rel)
for spec in POS[1:]:
    run_dir, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run_dir}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp']); l2.set_mode(ck)
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    D = run(bp, hp, 3); v, j, xi, fn, p, g, e, kd, ps, ye, fs = (D[k] for k in ('v', 'j', 'xi', 'fn', 'p', 'g', 'e', 'kd', 'ps', 'ye', 'fs'))
    win = v & (j == 2) & (kd >= e[:, 2:3] - np.round(0.4 * (e[:, 2:3] - e[:, 1:2]))); hw = win & (xi > 0.5); con = hw & (fn > 0)
    h = p[..., 1] - ye; lat = np.abs(p[..., 0] - ps[:, :1]); held3c = v & (j == 2) & (xi > 0.5) & (fn > 0)
    x, y = (fn / fs)[held3c], (g[..., 1] - ye)[held3c]
    slope = float(np.polyfit(x, y, 1)[0]) if x.size >= 10 else None
    hov = np.array([hw[i].any() and not con[i].any() for i in range(64)])
    R3 = dict(held_window_episodes=int(hw.any(1).sum()), contact_episodes=int(con.any(1).sum()), contact_share_of_held_window=float(con.sum() / max(hw.sum(), 1)),
              force_ratio_median_contact=med((fn / fs)[con]), force_ratio_p90_contact=float(np.percentile((fn / fs)[con], 90)) if con.any() else None,
              hover_episodes=int(hov.sum()), hover_tip_height_median=med(h[hw & hov[:, None]]), tip_height_median_held_window=med(h[hw]),
              lateral_offset_median_held_window=med(lat[hw]), releases_phase3=int(D['rel'](2).sum()),
              goal_height_on_force_slope=slope, goal_height_on_force_corr=float(np.corrcoef(x, y)[0, 1]) if x.size >= 10 else None, n_contact_decisions=int(x.size))
    D = run(bp, hp, 2); v, j, xi, p, g, e, kd, ps = (D[k] for k in ('v', 'j', 'xi', 'p', 'g', 'e', 'kd', 'ps'))
    ar = np.arange(64); f3 = np.argmax(v & (j == 2), 1); held3 = xi[ar, f3] > 0.5
    w2 = v & (j == 1) & (kd >= e[:, 1:2] - np.round(0.4 * (e[:, 1:2] - e[:, 0:1])))
    perr = np.array([np.mean(np.linalg.norm(p[i][w2[i]] - ps[i], axis=-1)) for i in range(64)]); gerr = np.array([np.linalg.norm(g[i][w2[i]][-1] - ps[i]) for i in range(64)])
    start = np.array([np.linalg.norm(p[i][v[i] & (j[i] == 1)][0] - ps[i]) for i in range(64)])
    R2 = dict(held_start3=int(held3.sum()), hold_error_median_held=med(perr[held3]), goal_tip_to_p_star_end2_median=med(gerr[held3]), tip_to_p_star_start_median=med(start),
              releases_phase2=int(D['rel'](1).sum()))
    REC[f"{run_dir.rstrip('/').split('/')[-1]}:{it}"] = dict(phase3_start=R3, phase2_start=R2); print(spec.split('/')[-1], json.dumps(dict(phase3_start=R3, phase2_start=R2)), flush=True)
    json.dump(REC, open(OUT, 'w'), indent=1)
