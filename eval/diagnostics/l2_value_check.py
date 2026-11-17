"""Value-fit diagnostics of a checkpoint (exploring rollouts on 256 pool episodes): critic residual at the first decision and the shares of its variance explained by
the later phase durations, the grasp outcome, and the geometry with the grasp outcome; the same rollouts scored under cost versions 1 and 3 (episode-cost spread, share of
the phase 2+3 cost variance explained by geometry and durations); gradient norms of the actor and the critic and the global-norm clipping factor of the actor's step.
Usage: python3 l2_value_check.py OUT_JSON RUN:IT:COSTV [...]   (COSTV: the cost the checkpoint was trained under)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

OUT = sys.argv[1]; REC = json.load(open(OUT)) if os.path.exists(OUT) else {}
pool = l1.with_E0(l1.episodes(10000, 256, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json')))[:256]); pb, tb = env.stack(pool)
po = np.array([t['po'] for p, t in pool]); ps = np.array([t['ps'] for p, t in pool]); ye = np.array([p['ye'] for p, t in pool]); pc = np.column_stack([ps[:, 0], ye])
dur = np.array([t['nph'] for p, t in pool]) * P.TS
def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None); return float(1 - np.var(y - X @ b) / np.var(y))
geom = np.column_stack([np.linalg.norm(po - ps, axis=1), np.abs(po[:, 0] - ps[:, 0]), np.linalg.norm(po - pc, axis=1), po[:, 1] - ye])
for spec in sys.argv[2:]:
    run, it, cv = spec.rsplit(':', 2)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    R, cost_ph = {}, {}
    for c in sorted({1, 3, int(cv)}, key=lambda c: c != int(cv)):
        env.COST_V = c; cfg = C.default_cfg()
        st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(4242), 256), bp, hp, 0.0, 0.0)
        cost_ph[c] = np.asarray(m['cost_ph'])
        if c == int(cv):
            v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j'])
            res_all = np.asarray(jax.jit(jax.vmap(lambda r, vv, d: l2.gae(r, vv, d, 0.99, 1.0)))(dec['r'], dec['v'], dec['valid']))
            k0 = np.array([np.where(v[e] & (j[e] == 0))[0][0] for e in range(256)]); res = res_all[np.arange(256), k0]
            gr = (np.asarray(st['n_grasp']) > 0).astype(float); xi = np.asarray(dec['x'])[..., 21]; held = (xi * (v & (j >= 1))).sum(1) / np.maximum((v & (j >= 1)).sum(1), 1)
            R['first_decision_residual_sd'] = float(res.std()); R['share_durations'] = r2(dur[:, 1:], res); R['share_grasp'] = r2(np.column_stack([gr, held]), res)
            R['share_geometry_grasp'] = r2(np.column_stack([geom, gr, held]), res)
            adv = np.asarray(jax.jit(jax.vmap(lambda r, vv, d: l2.gae(r, vv, d, 0.99, 0.95)))(dec['r'], dec['v'], dec['valid'])); ret = adv + np.asarray(dec['v'])
            msk = v.reshape(-1); fl = lambda a: np.asarray(a).reshape((-1,) + np.asarray(a).shape[2:])[msk]
            D = dict(x=fl(dec['x']), xc=fl(dec['xc']), u=fl(dec['u']), psi=fl(dec['psi']), logp=fl(dec['logp']), ret=fl(ret), adv=fl(adv))
            D['adv'] = (D['adv'] - D['adv'].mean()) / (D['adv'].std() + 1e-8); perm = np.random.default_rng(0).permutation(len(D['adv'])); na, nc, fac = [], [], []
            for i in range(0, len(perm) - 4096 + 1, 4096):
                mb = {k: jnp.asarray(a[perm[i:i + 4096]]) for k, a in D.items()}; (lo, aux), g = l2.ppo_grad(bp, mb, 0.2, 0.5, 1e-3)
                ga = float(np.sqrt(sum(np.sum(np.asarray(x) ** 2) for x in jax.tree_util.tree_leaves(g['actor'])))); gc = float(np.sqrt(sum(np.sum(np.asarray(x) ** 2) for x in jax.tree_util.tree_leaves(g['critic']))))
                na.append(ga); nc.append(gc); fac.append(min(1.0, 0.5 / np.sqrt(ga ** 2 + gc ** 2)))
            R.update(actor_grad_norm=float(np.mean(na)), critic_grad_norm=float(np.mean(nc)), clip_factor_mean=float(np.mean(fac)))
    for c in (1, 3):
        tot = cost_ph[c].sum(1); R[f'cost{c}'] = dict(episode_cost_mean=float(tot.mean()), episode_cost_sd=float(tot.std()),
                                                       share_phase23_geometry_durations=r2(np.column_stack([geom[:, :3], po[:, 1] - ye, dur[:, 1] + dur[:, 2]]), cost_ph[c][:, 1] + cost_ph[c][:, 2]))
    REC[f"{run.rstrip('/').split('/')[-1]}:{it}:cost{cv}"] = R; print(spec.split('/')[-1], json.dumps(R), flush=True)
    json.dump(REC, open(OUT, 'w'), indent=1)
