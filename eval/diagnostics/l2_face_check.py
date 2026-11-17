"""Which face of the block the tip touches during phase 3 (deterministic policy, 64 evaluation episodes)."""
import json
REC = {}
import jax; jax.config.update('jax_enable_x64', True)
from pairl import paths
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo
cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
for spec in sys.argv[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    q = np.asarray(rec['q']).reshape(64, -1, 3); j = np.repeat(np.asarray(dec['j']), P.NB, axis=1); xi = np.repeat(np.asarray(dec['x'])[..., 21], P.NB, axis=1); val = np.asarray(rec['val']).reshape(64, -1) > 0.5
    L = np.array([p['L'] for p, t in ev]); xe = np.array([p['xe'] for p, t in ev]); ye = np.array([p['ye'] for p, t in ev]); we = np.array([p['we'] for p, t in ev])
    th = np.cumsum(q, -1); px = np.sum(L[:, None, :] * np.cos(th), -1); py = np.sum(L[:, None, :] * np.sin(th), -1)
    dtop = ye[:, None] - py; dside = we[:, None] - np.abs(px - xe[:, None]); inside = (dtop > 0) & (dside > 0)
    sel = val & (j == 2) & (xi > 0.5)
    top = inside & (dtop <= dside); side = inside & (dtop > dside)
    REC[f'{run.split("/")[-1]}:{it}'] = dict(held_phase3_steps=int(sel.sum()), inside=float(np.mean(inside[sel])), top=float(np.mean(top[sel])), side=float(np.mean(side[sel])),
        side_depth_median=float(np.median(dtop[sel & side])) if np.any(sel & side) else None, side_penetration_median=float(np.median(dside[sel & side])) if np.any(sel & side) else None,
        episodes_top=float(np.mean(np.any(sel & top, 1)[np.any(sel, 1)])), episodes_side=float(np.mean(np.any(sel & side, 1)[np.any(sel, 1)])))
    print(f'{run.split("/")[-1]} it {it}: held phase-3 steps {sel.sum()}; inside the block {np.mean(inside[sel]):.3f} (top face {np.mean(top[sel]):.3f}, side face {np.mean(side[sel]):.3f}); '
          f'median depth below the top face when on the side {np.median(dtop[sel & side]) if np.any(sel & side) else float("nan"):.3f} m, side penetration {np.median(dside[sel & side]) if np.any(sel & side) else float("nan"):.4f} m; '
          f'episodes with any top-face contact {np.mean(np.any(sel & top, 1)[np.any(sel, 1)]):.2f}, with side contact {np.mean(np.any(sel & side, 1)[np.any(sel, 1)]):.2f}')
json.dump(REC, open(str(paths.RECORDS / 'l2_face_check.json'), 'w'), indent=1)
