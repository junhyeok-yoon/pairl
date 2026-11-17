import jax.numpy as jnp

# Learned feature map of the adaptive core: phi1(q, dq) and phi2(q) (x) a, a = reference acceleration (class-level structure)
N1, N2 = 8, 6
NF_META = N1 + 2 * N2

def mlp(ps, x):
    h = jnp.tanh(ps[0] @ x + ps[1]); h = jnp.tanh(ps[2] @ h + ps[3])
    return ps[4] @ h + ps[5]

def meta_phi(P, q, dq, a, cen, half):
    qn = (q - cen) / half
    f1 = jnp.tanh(mlp(P[0:6], jnp.concatenate([qn, dq / 2.0])))
    f2 = jnp.tanh(mlp(P[6:12], qn))
    return jnp.concatenate([f1, 0.2 * jnp.outer(f2, a).reshape(-1)])
