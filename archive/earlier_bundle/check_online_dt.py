import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import arm3 as P

# certificate under arbitrary time variation of weights: components outside the storage versus H and the offset
p0 = jax.tree_util.tree_map(jnp.asarray, pickle.load(open("arm3_ph.pkl", "rb"))["p"])
FREE = ["W1", "c1", "Wphi", "bphi", "WR", "bR", "WJ", "bJ", "WK", "bK", "vk", "dk"]
STOR = ["L", "A0", "b0", "U1", "A1", "b1", "u2", "a0"]
DT, T = 0.0004, 3.0
base_rhs = P.make_rhs(P.ph_op)


def make(group, amp, key):
    ks = random.split(key, 3 * len(group))
    D = {k: random.normal(ks[3 * i], p0[k].shape) for i, k in enumerate(group)}
    W = {k: random.uniform(ks[3 * i + 1], (), minval=1.0, maxval=5.0) for i, k in enumerate(group)}
    F = {k: random.uniform(ks[3 * i + 2], (), maxval=6.28) for i, k in enumerate(group)}

    def p_at(t):
        return {k: (v + amp * jnp.sin(W[k] * t + F[k]) * D[k] if k in D else v) for k, v in p0.items()}

    @jit
    def traj(a, eta, rp):
        def step(s, i):
            t = i * DT
            f = lambda tt, ss: base_rhs(p_at(tt), tt, ss, (a, eta, rp), jnp.zeros(4))[0]
            k1 = f(t, s); k2 = f(t + DT / 2, s + DT / 2 * k1); k3 = f(t + DT / 2, s + DT / 2 * k2); k4 = f(t + DT, s + DT * k3)
            return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), s
        qd0, dqd0, _ = P.ref(0.0, rp)
        s0 = jnp.concatenate([qd0 + 0.1, dqd0, P.Y_h(qd0 + 0.1, dqd0).reshape(-1), jnp.zeros(14 + 2 + P.NZ)])
        return jax.lax.scan(step, s0, jnp.arange(int(T / DT)))[1]
    return p_at, traj


a, eta, rp = P.sample_tasks(random.PRNGKey(4), 4, mp_hi=5.0, un=0.0)
for name, group, amp in [("outside the storage, amplitude 0.3", FREE, 0.3),
                         ("outside the storage, amplitude 1.0", FREE, 1.0)]:
    p_at, traj = make(group, amp, random.PRNGKey(9))
    worst, rel, fin = -np.inf, -np.inf, True
    for i in range(2):
        o = np.asarray(traj(a[i], eta[i], rp[i]))[::10]
        fin = fin and np.isfinite(o).all()
        t = np.arange(o.shape[0]) * 10 * DT
        q, dq, z = o[:, 0:2], o[:, 2:4], o[:, P.OFF:]
        qd, dqd, _ = vmap(lambda tt: P.ref(tt, rp[i]))(jnp.asarray(t))
        s = (dq - np.asarray(dqd)) + P.LAM_S * (q - np.asarray(qd))
        Ms = np.asarray(vmap(lambda qq: P.MCg(a[i], qq, jnp.zeros(2))[0])(jnp.asarray(q)))
        Hs = np.asarray(vmap(lambda tt, zz: P.H_fn(p_at(tt), zz) - zz[:P.NP_] @ (a[i] - P.A_NOM - p_at(tt)["a0"]))(jnp.asarray(t), jnp.asarray(z)))
        V = 0.5 * np.einsum('ti,tij,tj->t', s, Ms, s) + Hs
        worst = max(worst, np.diff(V).max()); rel = max(rel, np.diff(V).max() / (V[0] - V.min() + 1e-12))
    print(f"{name}: finite {fin}; max step increase of V {worst:.2e} (relative to the total decrease {rel:.2e})", flush=True)
