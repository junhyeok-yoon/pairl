"""Pressing and releasing from the phase-3 restart (pilot R, descriptive): exploring rollouts of a checkpoint from the phase-3 restart states of the first 256 pool
episodes (key 11): held phase-3 decisions, the share of held contacts above 2 f*, the force ratio in contact, the open probability in held phase-3 states, the share
of episodes that release in phase 3, and the share of held phase-3 decisions where releasing would lower the rate of cost revision 14.
Usage: python3 l2_overpress_check.py OUT_JSON RUN:IT [...] [cost=4]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, model as Mo

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = POS[0]; REC = json.load(open(OUT)) if os.path.exists(OUT) else {}
env.COST_V = int(OPT.get('cost', 4)); cfg = C.default_cfg()
pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json'))))
R3 = [env.restart(p, t, 3) for p, t in pool[:256]]; pb, tb = env.stack(R3); bf = l2.make_batch_fn(cfg)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
for spec in POS[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(11), 256), bp, hp, 0.0, 0.0)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']); fn = np.asarray(dec['viz']['fn'])
    p = np.asarray(tipf(dec['viz']['q'], pb['L'])); fs = np.asarray(tb['fstar'])[:, None]; ps = np.asarray(tb['ps'])
    pc = np.stack([ps[:, 0], np.asarray(pb['ye'])], -1)[:, None, :]; dpc = np.linalg.norm(p - pc, axis=-1) / 0.1
    held3 = v & (j == 2) & (xi > 0.5); con = held3 & (fn > 0)
    lg = np.asarray(jax.vmap(jax.vmap(lambda a: l2.policy(bp, a)[2]))(dec['x'])); po = 1 / (1 + np.exp(lg))
    rel3 = np.any((np.diff(xi, axis=1) < -0.5) & v[:, :-1] & (j[:, :-1] == 2), 1)
    R = dict(held_phase3_decisions=int(held3.sum()), contact_share_of_held=float(con.sum() / max(held3.sum(), 1)),
             contact_above_2fstar_share=float(np.mean((fn / fs)[con] > 2)) if con.any() else None,
             force_ratio_median=float(np.median((fn / fs)[con])) if con.any() else None, force_ratio_p90=float(np.percentile((fn / fs)[con], 90)) if con.any() else None,
             open_probability_held_phase3=float(po[held3].mean()) if held3.any() else None, episodes_releasing_phase3=float(rel3.mean()),
             rev14_release_cheaper_share_of_held=float(np.sum(con & (((fn - fs) / fs) ** 2 > 2 + dpc)) / max(held3.sum(), 1)))
    ch = con & (((fn - fs) / fs) ** 2 > 2 + dpc)
    R.update(rev14_release_cheaper_decisions=int(ch.sum()), contact_decisions=int(con.sum()), open_probability_rev14_cheaper=float(po[ch].mean()) if ch.any() else None,
             open_probability_other_held3=float(po[held3 & ~ch].mean()) if np.any(held3 & ~ch) else None)
    ev_e, ev_k = np.nonzero((np.diff(xi, axis=1) < -0.5) & v[:, :-1] & (j[:, :-1] == 2))          # release in phase 3: held at k, released at k + 1
    if len(ev_e):
        k2 = np.minimum(ev_k + 3, xi.shape[1] - 1); r0, r2 = (fn / fs)[ev_e, ev_k], (fn / fs)[ev_e, k2]
        R.update(release_events_phase3=int(len(ev_e)), force_ratio_at_release_median=float(np.median(r0)), force_ratio_0p1s_after_median=float(np.median(r2)),
                 share_release_with_force_above_2fstar=float(np.mean(r0 > 2)), share_force_zero_0p1s_after=float(np.mean(r2 <= 0)))
    REC[f"{run.rstrip('/').split('/')[-1]}:{it}"] = R; print(spec.split('/')[-1], json.dumps(R), flush=True)
    json.dump(REC, open(OUT, 'w'), indent=1)
