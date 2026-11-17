"""Transfer exposure of the hand's role loss (attempt 3, descriptive): on rollouts of a checkpoint (brain and hand) from the training start distribution (128
episodes: 64 original, 32 phase-2 and 32 phase-3 restarts, keys 5000+), the role weight w(t) = 1 - exp(-lam_s r(t)) of valid steps by the joint-space distance
of the posture from the posture at the last rest step (the memory is frozen since then), and by attachment. Only steps with a frozen memory after a rest test
how a calibration transfers. Usage: python3 l2_transfer_exposure.py OUT_JSON RUN:IT [RUN:IT ...]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

OUT = sys.argv[1]; env.COST_V = 4; cfg = C.default_cfg(); R = {}; BINS = (0.0, 0.05, 0.2, 0.5, 9.0)
pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json'))))
rng = np.random.default_rng(31); idx = rng.choice(len(pool), 128, replace=False)
eps = [pool[i] for i in idx[:64]] + [env.restart(*pool[i], 2) for i in idx[64:96]] + [env.restart(*pool[i], 3) for i in idx[96:]]
pb, tb = env.stack(eps); batch = l2.make_batch_fn(cfg)
for spec in sys.argv[2:]:
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    _, _, _, rec = batch(pb, tb, jax.random.split(jax.random.PRNGKey(5000), 128), bp, hp, 0.0, 0.0)
    f = {k: np.asarray(v).reshape((128, -1) + np.asarray(v).shape[3:]) for k, v in rec.items()}
    val, zeta, xi, q = f['val'] > 0.5, f['zeta'] > 0.5, f['xi'], f['q']
    n, T = val.shape; w = np.zeros((n, T)); dist = np.zeros((n, T)); rr = np.full(n, 1e9); xp = np.zeros(n); qr = q[:, 0].copy()
    for t in range(T):                                                     # rest time since the last load change, and the last rest posture
        rr = np.where(np.abs(xi[:, t] - xp) > 0.5, 0.0, rr); w[:, t] = 1.0 - np.exp(-cfg['lam_s'] * rr)
        qr = np.where(zeta[:, t, None], q[:, t], qr); dist[:, t] = np.linalg.norm(q[:, t] - qr, axis=-1)
        rr = rr + P.TS * zeta[:, t]; xp = xi[:, t]
    wv = w * val; tot = wv.sum(); out = dict(weight_total_per_episode=float(tot / n), weight_moving_share=float(wv[~zeta].sum() / tot))
    for name, sel in (('all', np.ones_like(val)), ('held', xi > 0.5), ('free', xi < 0.5)):
        out[name] = {f'{a:g}-{b:g} rad': float(wv[sel & ~zeta & (dist >= a) & (dist < b)].sum() / tot) for a, b in zip(BINS[:-1], BINS[1:])}
    R[f"{run.rstrip('/').split('/')[-1]}:{it}"] = out; print(spec.split('/')[-1], json.dumps(out), flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
