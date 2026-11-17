import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import panda as P

DT, T, NW = 0.002, 8.0, 6
sim = jit(P.make_sim(P.ph_op, P.NZ, DT, int(T / DT), full=True))
hess = jax.hessian(P.H_fn, argnums=1)


def z_star(p, c):
    z = jnp.zeros(P.NZ)
    for _ in range(40):
        z = z - jnp.linalg.solve(hess(p, z), P.grad_H(p, z) - c)
    return z


key = random.PRNGKey(5)
out = []
for i in range(NW):
    key, k1, k2, k3, k4 = random.split(key, 5)
    u = random.uniform(k3, (2,))
    p = P.init_ph(k1, gamma0=float(1.0 * 100 ** u[0]), k0=float(2.0 * 15 ** u[1]), scale=0.3)
    p["U1"] = p["U1"] + 2.0
    p["u2"] = p["u2"] + 2.0
    a, eta, rp = P.sample_tasks(k2, 1, mp_hi=5.0, un=0.0)
    task = (a[0], eta[0], rp[0])
    q0err = 0.1 * random.normal(k4, (P.N,))
    _, o = sim(p, task, jnp.zeros((int(T / P.DT_NOISE) + 2, 2 * P.N)), None, q0err)
    o = np.asarray(o)[::10]
    sv, q, z = o[:, 2 * P.N:3 * P.N], o[:, 3 * P.N:4 * P.N], o[:, 5 * P.N:]
    Ms = np.asarray(vmap(lambda qq: P.mass_matrix(qq, task[0]) + jnp.diag(P.ARM))(jnp.asarray(q)))
    c = jnp.concatenate([task[0] - P.A_NOM - p["a0"], jnp.zeros(P.NQ)])
    zs = z_star(p, c)
    Hs = np.asarray(vmap(lambda zz: P.H_fn(p, zz))(jnp.asarray(z)))
    D = Hs - float(P.H_fn(p, zs)) - (z - np.asarray(zs)) @ np.asarray(c)
    V = 0.5 * np.einsum('ti,tij,tj->t', sv, Ms, sv) + D
    en = np.degrees(np.linalg.norm(o[:, :P.N], axis=1))
    out.append(dict(V=V, en=en, maxinc=np.diff(V).max(), V0=V[0], Dmin=D.min(), tail=en[-int(1.0 / (10 * DT)):].max(), finite=np.isfinite(o).all()))
    print(f"w{i}: finite {out[-1]['finite']}, V0 {V[0]:.3g}, max step increase of V {out[-1]['maxinc']:.2e}, min D {D.min():.2e}, "
          f"|q-qd| max in last 1 s {out[-1]['tail']:.3f} deg, start {en[0]:.2f} deg", flush=True)
pickle.dump(out, open("struct_panda.pkl", "wb"))
