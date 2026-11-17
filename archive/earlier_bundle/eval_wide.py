import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import arm3 as P
import eval_seeds as E

p = jax.tree_util.tree_map(jnp.asarray, pickle.load(open("arm3_wide.pkl", "rb"))["p"])
prev = pickle.load(open("eval_seeds.pkl", "rb"))["res"]
orig = pickle.load(open("eval_orig.pkl", "rb"))
res = {}
for (sid, name, n, T, sw, kw, kw2), key in zip(E.SETS, random.split(random.PRNGKey(21), len(E.SETS))):
    k1, k2, k3 = random.split(key, 3)
    tasks = P.sample_tasks(k1, n, **kw)
    noise = P.SIG_N * random.normal(k2, (n, int(T / P.DT_NOISE) + 2, 4))
    args = [tasks[0], tasks[1], tasks[2], noise]
    if sw is not None:
        t2 = P.sample_tasks(k3, n, **kw2)
        args += [t2[0], t2[1]]
    res[sid] = E.metrics(np.asarray(E.fn(P.ph_wide_op, P.NZ, T, sw)(p, *args))[:, ::5, :])
    print(sid, "wide", round(float(res[sid]["rms"]), 3), round(float(res[sid]["dtau"]), 3), float(res[sid]["div"]),
          "| narrow seed0", round(float(orig[sid]["pH0"]["rms"]), 3), round(float(orig[sid]["pH0"]["dtau"]), 3),
          "| oracle", round(float(prev[sid]["C1"]["rms"]), 3), round(float(prev[sid]["C1"]["dtau"]), 3), flush=True)
pickle.dump(res, open("eval_wide.pkl", "wb"))
print("done", flush=True)
