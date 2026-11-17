"""Role-loss share of the segments after a load change (grasp or release) before the memory has rested 0.1 s, exploring rollouts on 64 evaluation episodes."""
import json
REC = {}
import jax; jax.config.update('jax_enable_x64', True)
from pairl import paths
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); bf = l2.make_batch_fn(cfg); pb, tb = env.stack(ev)
def replay_err(hp, r):
    def step(z, xx):
        G = l2.hand_G(hp, xx['q']); uad = G.T @ (l2.abar_of(hp) + z); e = jnp.sum((uad - xx['ustar']) ** 2)
        mh = uad - xx['taup'] - xx['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
        return jnp.clip(z - P.TS * cfg['lam_s'] * xx['zeta'] * (G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mh)), -cfg['zbar'], cfg['zbar']), e
    return jax.lax.scan(step, jnp.zeros(l2.NA), r)[1]
rep = jax.jit(lambda hp, r: jax.vmap(lambda rr: replay_err(hp, rr))(r))
for spec in sys.argv[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(900), 64), bp, hp, 0.0, 0.0)
    flat = jax.tree_util.tree_map(lambda a: jnp.asarray(a).reshape((a.shape[0], -1) + a.shape[3:]), rec)
    e = np.asarray(rep(hp, flat)); val = np.asarray(flat['val']) > 0.5; zeta = np.asarray(flat['zeta']); us = np.asarray(flat['ustar'])
    jump = np.r_['1', np.zeros((64, 1), bool), np.linalg.norm(np.diff(us, axis=1), axis=-1) > 0.5]
    unc = np.zeros_like(val); need = int(0.1 / P.TS)
    for b in range(64):
        cnt = need
        for k in range(val.shape[1]):
            if jump[b, k]: cnt = 0
            unc[b, k] = cnt < need
            if zeta[b, k] > 0.5: cnt += 1
    tot = e[val].sum(); u = e[val & unc].sum()
    REC[f'{run.split("/")[-1]}:{it}'] = dict(load_changes_per_episode=float(jump.sum() / 64), uncalibrated_time_share=float(np.mean(unc[val])), uncalibrated_loss_share=float(u / tot),
        mse_calibrated=float(e[val & ~unc].mean()), mse_uncalibrated=float(e[val & unc].mean()) if np.any(val & unc) else None)
    print(f'{run.split("/")[-1]} it {it}: load changes per episode {jump.sum() / 64:.2f}; uncalibrated time share {np.mean(unc[val]):.3f}; role-loss share of uncalibrated segments {u / tot:.3f}; '
          f'mean squared error calibrated {e[val & ~unc].mean():.3f} vs uncalibrated {e[val & unc].mean() if np.any(val & unc) else float("nan"):.3f}')
json.dump(REC, open(str(paths.RECORDS / 'l2_hand_share.json'), 'w'), indent=1)
