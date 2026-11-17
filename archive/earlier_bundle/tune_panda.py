import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import panda as P

DT, T, n = 0.004, 3.0, 8
k1, k2 = random.split(random.PRNGKey(0))
tasks = P.sample_tasks(k1, n)
noise = P.SIG_N * random.normal(k2, (n, int(T / P.DT_NOISE) + 2, 2 * P.N))
best = {}
for nm, op, grid in [("c0", P.sl_op, [dict(gamma=g, k=k) for g in [10.0, 30.0, 100.0] for k in [20.0, 40.0, 80.0]])]:
    sim = P.make_sim(op, P.NZ, DT, int(T / DT))
    f = jit(vmap(lambda p, a, e, r, nn: sim(p, (a, e, r), nn)[0], in_axes=(None, 0, 0, 0, 0)))
    res = []
    for d in grid:
        c = float(np.nanmean(np.asarray(f(d, *tasks, noise))))
        res.append((c if np.isfinite(c) else 1e9, d))
        print(nm, d, f"{c:.3e}", flush=True)
    best[nm] = min(res, key=lambda r: r[0])
sim = P.make_sim(P.oracle_op, P.NZ, DT, int(T / DT))
f = jit(vmap(lambda p, a, e, r, nn: sim(p, (a, e, r), nn)[0], in_axes=(None, 0, 0, 0, 0)))
res = []
for gb in [0.3, 1.0, 3.0, 10.0, 30.0]:
    d = dict(best["c0"][1], gb=gb)
    c = float(np.nanmean(np.asarray(f(d, *tasks, noise))))
    res.append((c if np.isfinite(c) else 1e9, d))
    print("c1", d, f"{c:.3e}", flush=True)
best["c1"] = min(res, key=lambda r: r[0])
print("best", best, flush=True)
pickle.dump(best, open("panda_classical.pkl", "wb"))
