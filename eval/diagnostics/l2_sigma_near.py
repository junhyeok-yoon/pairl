"""Exploration scale near the object (descriptive; pilot G): deterministic rollouts of the 256 original-start evaluation episodes for checkpoints RUN_DIR:IT;
at free phase-1 decisions, the policy's standard deviation of the tip-displacement action (pre-squash, and as a tip distance 0.3 (1 - tanh(mu)^2) sigma,
norm over the two components) and the commanded displacement |dp| = 0.3 |tanh(mu)|, within 5 cm of the grasp point against beyond 10 cm.
Usage: python eval/diagnostics/l2_sigma_near.py OUT_JSON RUN_DIR:IT [...]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo

OUT = sys.argv[1]; env.COST_V = 4; cfg = C.default_cfg(); N = 256
ev = l1.with_E0(l1.episodes(6000, N, lim=0.7), P.E_NOM); pb, tb = env.stack(ev); keys = jax.random.split(jax.random.PRNGKey(555), N)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0))); R = json.load(open(OUT)) if os.path.exists(OUT) else {}
med = lambda a: float(np.median(a)) if np.size(a) else None
for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, 1.0, 0.0)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']) > 0.5; q = np.asarray(dec['viz']['q'])
    p = np.asarray(tipf(jnp.asarray(q), pb['L'])); com, ops = np.asarray(dec['viz']['com']), np.asarray(dec['viz']['opsi']); lo = np.asarray(pb['Lobj'])[:, None, None]
    d = np.linalg.norm(p - (com + lo * np.stack([np.cos(ops), np.sin(ops)], -1)), axis=-1)
    mu, ls, _ = jax.vmap(jax.vmap(lambda x: l2.policy(bp, x)))(dec['x']); mu, sd = np.asarray(mu)[..., :2], np.exp(np.asarray(ls))[..., :2]
    sd_tip = np.linalg.norm(l2.DP_MAX * (1 - np.tanh(mu) ** 2) * sd, axis=-1); dp = np.linalg.norm(l2.DP_MAX * np.tanh(mu), axis=-1)
    free1 = v & (j == 0) & ~xi; near, far = free1 & (d < 0.05), free1 & (d > 0.1)
    o = {name: dict(decisions=int(s.sum()), sd_pre_squash=med(sd[s].mean(-1)), sd_tip_m=med(sd_tip[s]), dp_m=med(dp[s]), sd_tip_over_dp=med((sd_tip / np.maximum(dp, 1e-9))[s]),
                    distance_m=med(d[s])) for name, s in (('within_5cm', near), ('beyond_10cm', far))}
    key = f"{run.rstrip('/').split('/')[-1]}:{it}"; R[key] = o; print(key, json.dumps(o), flush=True); json.dump(R, open(OUT, 'w'), indent=1)
