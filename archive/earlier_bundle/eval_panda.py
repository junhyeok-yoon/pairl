import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import panda as P

DT = 0.004
cl = pickle.load(open("panda_classical.pkl", "rb"))
c0, c1 = cl["c0"][1], cl["c1"][1]
if os.environ.get("C1GB"):
    c1 = dict(c1, gb=float(os.environ["C1GB"]))
J = lambda d: jax.tree_util.tree_map(jnp.asarray, d)
f = lambda d: {k: float(v) for k, v in d.items()}
ctrl = {"nominal": (P.sl_op, {"gamma": 0.0, "k": float(c0["k"])}), "C0": (P.sl_op, f(c0)), "C1": (P.oracle_op, f(c1)),
        "pH": (P.ph_op, J(pickle.load(open(os.environ.get("PHF", "panda_ph.pkl"), "rb"))["p"]))}
if os.environ.get("ONLY"):
    ctrl = {k: v for k, v in ctrl.items() if k == os.environ["ONLY"]}
SETS = [("A", "분포 안", 8, 5.0, None, {}, None),
        ("B", "payload 6 kg, 마찰 2배", 8, 5.0, None, dict(mp_hi=6.0, visc_hi=2.0, un=2.0), None),
        ("C", "payload pick-up", 8, 6.0, 3.0, dict(mp_hi=0.5), dict(mp_hi=5.0)),
        ("D", "긴 horizon (15 s)", 4, 15.0, None, {}, None)]
_fns = {}


def fn(op, T, sw):
    key = (op.__name__, T, sw)
    if key not in _fns:
        sim = P.make_sim(op, P.NZ, DT, int(T / DT), switch=sw)
        if sw is None:
            _fns[key] = jit(vmap(lambda p, a, e, r, nn: sim(p, (a, e, r), nn)[1], in_axes=(None, 0, 0, 0, 0)))
        else:
            _fns[key] = jit(vmap(lambda p, a, e, r, nn, a2, e2: sim(p, (a, e, r), nn, (a2, e2, r))[1], in_axes=(None, 0, 0, 0, 0, 0, 0)))
    return _fns[key]


res, traj = {}, {}
for (sid, name, n, T, sw, kw, kw2), key in zip(SETS, random.split(random.PRNGKey(9), len(SETS))):
    k1, k2, k3 = random.split(key, 3)
    tasks = P.sample_tasks(k1, n, **kw)
    noise = P.SIG_N * random.normal(k2, (n, int(T / P.DT_NOISE) + 2, 2 * P.N))
    args = list(tasks) + [noise]
    if sw is not None:
        t2 = P.sample_tasks(k3, n, **kw2)
        args += [t2[0], t2[1]]
    res[sid] = {}
    for cn, (op, p) in ctrl.items():
        out = np.asarray(fn(op, T, sw)(p, *args))
        bad = ~np.isfinite(out).all(axis=(1, 2)) | (np.nanmax(np.abs(out[:, :, :P.N]), axis=(1, 2)) > 3)
        g = ~bad
        en = np.degrees(np.linalg.norm(out[g, :, :P.N], axis=2))
        tn = np.linalg.norm(out[g, :, P.N:2 * P.N], axis=2)
        res[sid][cn] = dict(rms=np.sqrt(np.mean(en ** 2)), p95=np.percentile(en.max(axis=1), 95),
                            tail=np.sqrt(np.mean(en[:, -en.shape[1] // 5:] ** 2)), dtau=np.sqrt(np.mean(tn ** 2)), div=bad.mean())
        if sid in ("C", "D"):
            traj[(sid, cn)] = out[:2]
        r = res[sid][cn]
        print(f"{sid} {cn:8s} rms {r['rms']:.3f} deg  tail {r['tail']:.3f}  p95max {r['p95']:.2f}  dtau {r['dtau']:.2f} Nm  div {100*r['div']:.0f}%", flush=True)
pickle.dump({"res": res, "traj": traj, "sets": [(s[0], s[1]) for s in SETS]}, open(os.environ.get("EOUT", "eval_panda.pkl"), "wb"))
print("done", flush=True)
