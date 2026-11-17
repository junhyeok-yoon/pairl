import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import arm3 as P

p = jax.tree_util.tree_map(jnp.asarray, pickle.load(open("arm3_wide.pkl", "rb"))["p"])
DT, T = 0.002, 8.0
rhs = P.make_rhs(P.ph_wide_op)


@jit
def traj(a, eta, rp):
    def step(s, i):
        t = i * DT
        n = jnp.zeros(4)
        k1 = rhs(p, t, s, (a, eta, rp), n)[0]; k2 = rhs(p, t + DT / 2, s + DT / 2 * k1, (a, eta, rp), n)[0]
        k3 = rhs(p, t + DT / 2, s + DT / 2 * k2, (a, eta, rp), n)[0]; k4 = rhs(p, t + DT, s + DT * k3, (a, eta, rp), n)[0]
        return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), s
    qd0, dqd0, _ = P.ref(0.0, rp)
    s0 = jnp.concatenate([qd0 + 0.1, dqd0, P.Y_h(qd0 + 0.1, dqd0).reshape(-1), jnp.zeros(14 + 2 + P.NZ)])
    return jax.lax.scan(step, s0, jnp.arange(int(T / DT)))[1]


a, eta, rp = P.sample_tasks(random.PRNGKey(4), 6, mp_hi=5.0, un=0.0)
worst, hmin = -np.inf, np.inf
hess = jax.hessian(P.H_wide, argnums=1)
for i in range(6):
    o = np.asarray(traj(a[i], eta[i], rp[i]))[::10]
    t = np.arange(o.shape[0]) * 10 * DT
    q, dq, z = o[:, 0:2], o[:, 2:4], o[:, P.OFF:]
    qd, dqd, _ = vmap(lambda tt: P.ref(tt, rp[i]))(jnp.asarray(t))
    s = (dq - np.asarray(dqd)) + P.LAM_S * (q - np.asarray(qd))
    Ms = np.asarray(vmap(lambda qq: P.MCg(a[i], qq, jnp.zeros(2))[0])(jnp.asarray(q)))
    ca = np.asarray(a[i] - P.A_NOM - p["a0"])
    V = 0.5 * np.einsum('ti,tij,tj->t', s, Ms, s) + np.asarray(vmap(lambda zz: P.H_wide(p, zz))(jnp.asarray(z))) - z[:, :P.NP_] @ ca
    worst = max(worst, np.diff(V).max())
    hmin = min(hmin, min(float(jnp.linalg.eigvalsh(hess(p, jnp.asarray(z[j]))).min()) for j in range(0, len(z), 40)))
print("trained widened law, exact matching, 6 tasks: max step increase of V %.2e ; min Hessian eigenvalue of H along trajectories %.3f" % (worst, hmin), flush=True)
