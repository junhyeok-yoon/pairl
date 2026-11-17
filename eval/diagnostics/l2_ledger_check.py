"""Load in the ledger along training (attempt 3, descriptive): for each checkpoint, deterministic rollouts of the 64 evaluation episodes (brain and hand of the
checkpoint, as in the evaluation); at held decisions of phases 2 and 3, the ledger load as a tip mass (L1 rule, l1.tip_force(q, u_load)[1] / g) against
the true object mass, and the share of held episodes whose L1 class rule (tip load above M_TH g) matches the true class at the end of phase 3.
Usage: python3 l2_ledger_check.py OUT_JSON RUN:IT [RUN:IT ...]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo

OUT = sys.argv[1]; env.COST_V = 4; cfg = C.default_cfg(); R = {}
ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); pb, tb = env.stack(ev)
mo = np.asarray(pb['mo']); heavy = np.asarray(tb['heavy']) > 0.5
mass = jax.jit(jax.vmap(jax.vmap(lambda q, u: l1.tip_force(q, u)[1] / Mo.G0)))
for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    x = np.asarray(dec['x']); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = x[..., 21] > 0.5
    q = x[..., 0:3] * np.asarray(P.QH) + np.asarray(P.QC); mh = np.asarray(mass(jnp.asarray(q), jnp.asarray(x[..., 24:27] * 10)))
    out = {}
    for k, name in ((1, 'phase2'), (2, 'phase3')):
        s = v & (j == k) & xi; err = (mh - mo[:, None])[s]
        last = np.where(s, np.arange(j.shape[1])[None], -1).max(1); he = last >= 0
        dec_ok = (mh[np.arange(64), np.maximum(last, 0)] > P.M_TH) == heavy
        out[name] = dict(held_decisions=int(s.sum()), held_episodes=int(he.sum()), mass_error_median=float(np.median(err)) if s.any() else None,
                         mass_abs_error_median=float(np.median(np.abs(err))) if s.any() else None,
                         class_rule_correct_at_last_held=float(dec_ok[he].mean()) if he.any() else None)
    R[f"{run.rstrip('/').split('/')[-1]}:{it}"] = out; print(spec.split('/')[-1], json.dumps(out), flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
