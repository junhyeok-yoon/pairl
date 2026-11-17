import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, value_and_grad
import optax
import panda as P

DT, T, BS, IT = 0.004, 1.5, 8, int(os.environ.get("IT", 30))
INIT, OUT = os.environ.get("INIT"), os.environ.get("OUT", "panda_ph.pkl")
sim = P.make_sim(P.ph_op, P.NZ, DT, int(T / DT), method="rk2")
cl = pickle.load(open("panda_classical.pkl", "rb"))["c0"][1]
p0 = P.init_ph(random.PRNGKey(1), gamma0=cl["gamma"], k0=cl["k"])
if INIT:
    p0 = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(INIT, "rb"))["p"])


def loss(p, key):
    k1, k2 = random.split(key)
    tasks = P.sample_tasks(k1, BS)
    noise = P.SIG_N * random.normal(k2, (BS, int(T / P.DT_NOISE) + 2, 2 * P.N))
    return jnp.mean(vmap(lambda a, e, r, n: sim(p, (a, e, r), n)[0])(*tasks, noise))


opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, IT, 0.1)))
st = opt.init(p0)


@jit
def step(p, st, k):
    l, g = value_and_grad(loss)(p, k)
    g = jax.tree_util.tree_map(jnp.nan_to_num, g)
    up, st = opt.update(g, st)
    return optax.apply_updates(p, up), st, l


p, hist, t0, key = p0, [], time.time(), random.PRNGKey(3 if INIT else 2)
for i in range(IT):
    key, k = random.split(key)
    p, st, l = step(p, st, k)
    hist.append(float(l))
    if i % 5 == 0 or i == IT - 1:
        print(f"it {i:3d} loss {np.mean(hist[-5:]):.3e} ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(jax.tree_util.tree_map(np.asarray, {"p": p, "h": hist}), open(OUT, "wb"))
