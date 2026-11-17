import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax
import arm3 as P
import train_arm3 as TR

# ablation: same class without dissipative feature states (phi = 0); learned gains, geometry, damping and prior remain
_, g, k, ke = pickle.load(open("arm3_classical.pkl", "rb"))["c0"]
k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
p0 = P.init_ph(k1, gamma0=g, k0=k, ke0=ke)
p0["Wphi"] = jnp.zeros_like(p0["Wphi"])
p0["bphi"] = jnp.zeros_like(p0["bphi"])
FROZEN = ("Wphi", "bphi")
cost = TR.batch_cost(P.ph_op, P.NZ)
loss = lambda p, tasks, noise: cost(p, tasks, noise).mean()
IT = 250
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, IT, 0.1)))
st = opt.init(p0)


@jit
def step(p, st, kk):
    tasks, noise = TR.draw(kk, 16)
    l, gr = value_and_grad(loss)(p, tasks, noise)
    gr = {n: (jnp.zeros_like(v) if n in FROZEN else jnp.nan_to_num(v)) for n, v in gr.items()}
    up, st = opt.update(gr, st)
    return optax.apply_updates(p, up), st, l


p, hist, t0, key = p0, [], time.time(), k2
for i in range(IT):
    key, kk = random.split(key)
    p, st, l = step(p, st, kk)
    hist.append(float(l))
    if i % 25 == 0 or i == IT - 1:
        print(f"  it {i:4d} loss {np.mean(hist[-25:]):.3e}  ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(jax.tree_util.tree_map(np.asarray, {"p": p, "h": hist}), open("arm3_ablate.pkl", "wb"))
