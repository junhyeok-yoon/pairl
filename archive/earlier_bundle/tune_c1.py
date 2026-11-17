import pickle
import numpy as np
import jax
from jax import random, jit, vmap
import panda as P

DT, T, n = 0.004, 3.0, 8
k1, k2 = random.split(random.PRNGKey(0))
tasks = P.sample_tasks(k1, n)
noise = P.SIG_N * random.normal(k2, (n, int(T / P.DT_NOISE) + 2, 2 * P.N))
best = pickle.load(open("panda_classical.pkl", "rb"))
sim = P.make_sim(P.oracle_op, P.NZ, DT, int(T / DT))
f = jit(vmap(lambda p, a, e, r, nn: sim(p, (a, e, r), nn)[0], in_axes=(None, 0, 0, 0, 0)))
res = [best["c1"]]
for gb in [100.0, 300.0, 1000.0]:
    d = dict(best["c0"][1], gb=gb)
    c = float(np.nanmean(np.asarray(f(d, *tasks, noise))))
    res.append((c if np.isfinite(c) else 1e9, d))
    print("c1", d, f"{c:.3e}", flush=True)
best["c1"] = min(res, key=lambda r: r[0])
print("best c1", best["c1"], flush=True)
pickle.dump(best, open("panda_classical.pkl", "wb"))
