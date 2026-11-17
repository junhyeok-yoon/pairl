"""Advantage scale by phase in exploring rollouts (128 pool episodes) of a checkpoint under a given cost version: raw GAE advantage standard deviation per phase,
its share of the batch standard deviation used for normalization, critic explained variance per phase. Usage: python3 l2_adv_check.py OUT_JSON RUN:IT:COSTV [...]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
OUT = sys.argv[1]; REC = json.load(open(OUT)) if os.path.exists(OUT) else {}
for spec in sys.argv[2:]:
    run, it, cv = spec.rsplit(':', 2); env.COST_V = int(cv)
    cfg = C.default_cfg(); pool = l1.with_E0(l1.episodes(10000, 128, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json')))[:128]); pb, tb = env.stack(pool)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    bf = l2.make_batch_fn(cfg)
    st, m, dec, rec = bf(pb, tb, jax.random.split(jax.random.PRNGKey(4242), 128), bp, hp, 0.0, 0.0)
    adv = np.asarray(jax.jit(jax.vmap(lambda r, v, d: l2.gae(r, v, d, 0.99, 0.95)))(dec['r'], dec['v'], dec['valid'])); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j'])
    ret = adv + np.asarray(dec['v']); sd_all = adv[v].std()
    ph = {f'phase{k + 1}': dict(adv_sd=float(adv[v & (j == k)].std()), share_of_batch_sd=float(adv[v & (j == k)].std() / sd_all),
                               ev=float(1 - np.var(ret[v & (j == k)] - np.asarray(dec['v'])[v & (j == k)]) / max(np.var(ret[v & (j == k)]), 1e-12))) for k in range(4)}
    REC[f"{run.rstrip('/').split('/')[-1]}:{it}:cost{cv}"] = dict(batch_adv_sd=float(sd_all), **ph)
    print(spec.split('/')[-1], f'batch sd {sd_all:.3f}', ' | '.join(f"{k}: sd {d['adv_sd']:.3f} share {d['share_of_batch_sd']:.2f} ev {d['ev']:.2f}" for k, d in ph.items()), flush=True)
    json.dump(REC, open(OUT, 'w'), indent=1)
