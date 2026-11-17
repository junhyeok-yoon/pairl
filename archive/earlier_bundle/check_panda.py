import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import panda as P

key = random.PRNGKey(0)
ks = random.split(key, 6)
a = P.params(2.0, jnp.array([0.01, -0.02, 0.17]), 0.5 * jnp.ones(P.N))
q = P.Q0 + 0.5 * random.normal(ks[0], (P.N,))
dq, x, y = [random.normal(k, (P.N,)) for k in ks[1:4]]
M = P.mass_matrix(q, a) + jnp.diag(P.ARM)
print("M symmetric: %.1e ; min eig %.3f ; max eig %.3f" % (float(jnp.abs(M - M.T).max()), float(jnp.linalg.eigvalsh(M).min()), float(jnp.linalg.eigvalsh(M).max())))
Md = jax.jvp(lambda qq: P.mass_matrix(qq, a), (q,), (dq,))[1]
g = jax.grad(P.potential)(q, a)
Cx = P.tau_r(a, q, dq, x, jnp.zeros(P.N)) - g - a[70:] * x
print("skew check x'(dM - 2C)x: %.2e (scale %.2e)" % (float(x @ Md @ x - 2 * x @ Cx), float(abs(x @ Md @ x))))
a2 = P.params(0.5, jnp.array([0.0, 0.0, 0.12]), 0.2 * jnp.ones(P.N))
lin = P.tau_r(a + a2, q, dq, x, y) - P.tau_r(a, q, dq, x, y) - P.tau_r(a2, q, dq, x, y)
print("linearity residual: %.1e" % float(jnp.abs(lin).max()))
Y = jax.jacfwd(lambda aa: P.tau_r(aa, q, dq, x, y))(jnp.zeros(P.NA))
s = random.normal(ks[4], (P.N,))
vj = jax.vjp(lambda aa: P.tau_r(aa, q, dq, x, y), a)[1](s)[0]
print("Y^T s via vjp vs jacfwd: %.1e ; Y shape %s" % (float(jnp.abs(vj - Y.T @ s).max()), Y.shape))
print("gravity torque at Q0, no payload [Nm]:", np.round(np.asarray(jax.grad(P.potential)(P.Q0, P.A_NOM)), 2))
a0 = P.params(1.0, jnp.array([0.0, 0.0, 0.15]), jnp.zeros(P.N))
f = jit(lambda s: jnp.concatenate([s[P.N:], P.plant_acc(s[:P.N], s[P.N:], jnp.zeros(P.N), a0, jnp.zeros(2 * P.N))]))
E = jit(lambda s: 0.5 * s[P.N:] @ (P.mass_matrix(s[:P.N], a0) + jnp.diag(P.ARM)) @ s[P.N:] + P.potential(s[:P.N], a0))
st = jnp.concatenate([P.Q0, jnp.zeros(P.N)]); dt = 0.001; E0 = float(E(st))
for i in range(2000):
    k1 = f(st); k2 = f(st + dt / 2 * k1); k3 = f(st + dt / 2 * k2); k4 = f(st + dt * k3)
    st = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
print("free fall 2 s, no friction: energy drift %.2e J (E0 %.2f J), joint travel %.2f rad" % (float(E(st)) - E0, E0, float(jnp.abs(st[:P.N] - P.Q0).max())))
tasks = P.sample_tasks(ks[5], 16)
noise = P.SIG_N * random.normal(ks[5], (16, 401, 2 * P.N))
for nm, op, p in [("C0", P.sl_op, {"gamma": 10.0, "k": 10.0}), ("pH", P.ph_op, P.init_ph(ks[0]))]:
    rhs = P.make_rhs(op)
    g1 = jit(vmap(lambda a, e, r, n: rhs(p, 0.3, jnp.concatenate([P.Q0, jnp.zeros(P.N), jnp.zeros(P.NZ)]), (a, e, r), n[0])[0]))
    jax.block_until_ready(g1(*tasks, noise))
    t0 = time.time(); [jax.block_until_ready(g1(*tasks, noise)) for _ in range(5)]
    print(f"{nm}: one RHS evaluation for a batch of 16: {(time.time()-t0)/5*1e3:.1f} ms")
