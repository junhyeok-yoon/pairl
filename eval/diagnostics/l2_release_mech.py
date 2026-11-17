"""Release mechanism along an L2 run: at the attached decisions of phases 2 and 3 in exploring rollouts (64 evaluation episodes, one noise draw),
the held cost rate against the dropped rate of spec rev. 13 (1 with the tip at the object), the critic's value with xi flipped to 0, and the open probability.
Usage: python3 l2_release_mech.py OUT_JSON RUN_DIR:IT [RUN_DIR:IT ...]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
cen = np.array([0.55, 0.1]); ps = np.array([t['ps'] for p, t in ev]); ye = np.array([p['ye'] for p, t in ev])
pol = jax.jit(lambda bp, x: jax.vmap(jax.vmap(lambda a: l2.policy(bp, a)[2]))(x)); val = jax.jit(lambda bp, x: jax.vmap(jax.vmap(lambda a: l2.value(bp, a)))(x))
out = []
for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); it = int(it)
    ck = pickle.load(open(f'{run}/params_it{it:04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)                       # first phase of attempt 1: function-preserving upgrade
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(900), 64), bp, hp, 0.0, 0.0)
    x = np.asarray(dec['x']); xc = np.asarray(dec['xc']); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = x[..., 21]; p = x[..., 19:21] * 0.3 + cen
    po = 1 / (1 + np.exp(np.asarray(pol(bp, jnp.asarray(x))))); xf = xc.copy(); xf[..., 21] = 0.0
    dV = np.asarray(val(bp, jnp.asarray(xf))) - np.asarray(val(bp, jnp.asarray(xc)))
    held = np.where(j == 1, np.linalg.norm(p - ps[:, None, :], axis=-1) / 0.1,
                    1 + np.maximum(p[..., 1] - ye[:, None], 0) / 0.1 + np.abs(p[..., 0] - ps[:, None, 0]) / 0.1)   # phase 3: before contact
    r = dict(run=run.rstrip('/').split('/')[-1], it=it)
    for k, name in ((1, 'phase2'), (2, 'phase3')):
        s = v & (xi > 0.5) & (j == k)
        r[name] = dict(n=int(s.sum()), p_open=float(po[s].mean()), held_gt_dropped=float(np.mean(held[s] > 1)), held_rate_median=float(np.median(held[s])),
                       dV_median=float(np.median(dV[s])), dV_pos=float(np.mean(dV[s] > 0))) if s.any() else None
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
