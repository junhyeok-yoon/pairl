import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad
import optax

jax.config.update("jax_enable_x64", True)
NA, NF, AM, BOX, BX, BY = 2, 2, 2.0, 1.5, 2.0, 0.6
NZ = NA + NF
S2 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
Phi = lambda x: jnp.array([x, 1.0])

# target operator: non-convex coercive Hamiltonian, state-dependent interconnection, damping, features and feedthrough
H_s = lambda z: 0.5 * z @ z + 0.8 * jnp.cos(1.5 * z[0]) * jnp.sin(z[2]) + 0.5 * jnp.tanh(z[1] * z[3])
J_s = lambda x, z: 0.7 * jnp.cos(z[0]) * S2
R_s = lambda x, z: jnp.diag(jnp.array([0.5 + 0.4 * jnp.sin(x) ** 2, 0.3 + 0.2 * jnp.tanh(z[2]) ** 2]))
G_s = lambda x, z: jnp.array([jnp.sin(2 * x) + 0.3 * z[3], 0.5 * jnp.cos(x)])
k_s = lambda x, z: 0.5 + 0.5 * x ** 2 / (1 + x ** 2)


def op(comp, z, x, y):
    H, J, R, G, kap = comp
    g = grad(H)(z)
    zdot = jnp.concatenate([-Phi(x) * y, (J(x, z) - R(x, z)) @ g[NA:] - G(x, z) * y])
    return zdot, Phi(x) @ g[:NA] + G(x, z) @ g[NA:] - kap(x, z) * y


TARGET = (H_s, J_s, R_s, G_s, k_s)


def init(key, m):
    k = random.split(key, 8)
    return {"W1": 0.7 * random.normal(k[0], (m, NZ)), "b1": random.normal(k[1], (m,)), "w2": random.normal(k[2], (m,)) / jnp.sqrt(m),
            "V1": 0.7 * random.normal(k[3], (m, 1 + NZ)), "c1": random.normal(k[4], (m,)),
            "Vh": random.normal(k[6], (m, m)) / jnp.sqrt(m), "ch": jnp.zeros(m), "V2": random.normal(k[5], (8, m)) / jnp.sqrt(m), "c2": jnp.zeros(8)}


def comp_nn(p):
    H = lambda z: 0.5 * z @ z + p["w2"] @ jnp.tanh(p["W1"] @ z + p["b1"])          # coercive for every weight
    o = lambda x, z: p["V2"] @ jnp.tanh(p["Vh"] @ jnp.tanh(p["V1"] @ jnp.concatenate([jnp.array([x]), z]) + p["c1"]) + p["ch"]) + p["c2"]
    J = lambda x, z: o(x, z)[0] * S2
    R = lambda x, z: o(x, z)[1:5].reshape(2, 2) @ o(x, z)[1:5].reshape(2, 2).T
    G = lambda x, z: o(x, z)[5:7]
    kap = lambda x, z: jnp.logaddexp(0.0, o(x, z)[7])
    return (H, J, R, G, kap)


def sample(key, n):
    k1, k2, k3 = random.split(key, 3)
    return random.uniform(k1, (n, NZ), minval=-BOX, maxval=BOX), random.uniform(k2, (n,), minval=-BX, maxval=BX), random.uniform(k3, (n,), minval=-BY, maxval=BY)


def field_err(p, z, x, y):
    zd, u = vmap(lambda a, b, c: op(comp_nn(p), a, b, c))(z, x, y)
    zs, us = vmap(lambda a, b, c: op(TARGET, a, b, c))(z, x, y)
    return jnp.sum((zd - zs) ** 2, 1) + (u - us) ** 2, jnp.maximum(jnp.abs(zd - zs).max(1), jnp.abs(u - us))


def closed_loop(comp, task, T=10.0, dt=0.005):
    c, lam, rp, x0 = task

    def rhs(t, s):
        x, xr, z = s[0], s[1], s[2:]
        r = rp[0] * jnp.sin(rp[1] * t + rp[2])
        zd, u = op(comp, z, x, x - xr)
        return jnp.concatenate([jnp.array([-AM * x + AM * r + lam * (u - Phi(x) @ c), -AM * xr + AM * r]), zd]), u

    def step(s, i):
        t = i * dt
        k1, u = rhs(t, s); k2, _ = rhs(t + dt / 2, s + dt / 2 * k1); k3, _ = rhs(t + dt / 2, s + dt / 2 * k2); k4, _ = rhs(t + dt, s + dt * k3)
        V = 0.5 * (s[0] - s[1]) ** 2 + lam * (comp[0](s[2:]) - c @ s[2:2 + NA])
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), jnp.concatenate([jnp.array([s[0] - s[1], u, V]), s[2:]])

    return jax.lax.scan(step, jnp.concatenate([jnp.array([x0, x0]), jnp.zeros(NZ)]), jnp.arange(int(T / dt)))[1]


kt = random.split(random.PRNGKey(7), 4)
NT = 20
tasks = (random.uniform(kt[0], (NT, NA), minval=-1, maxval=1), random.uniform(kt[1], (NT,), minval=0.5, maxval=1.0),
         jnp.stack([random.uniform(kt[2], (NT,), minval=0.3, maxval=1.0), random.uniform(kt[2], (NT,), minval=0.5, maxval=2.5),
                    random.uniform(kt[3], (NT,), maxval=6.28)], 1), random.uniform(kt[3], (NT,), minval=-0.5, maxval=0.5))
ref_tr = np.asarray(jit(vmap(lambda c, l, r, x: closed_loop(TARGET, (c, l, r, x))))(*tasks))
print("target operator: max |z| on closed-loop trajectories %.2f, max |x - x_r| %.2f (training box: |z| <= %.1f, |x| <= %.1f, |y| <= %.1f)" % (np.abs(ref_tr[:, :, 3:]).max(), np.abs(ref_tr[:, :, 0]).max(), BOX, BX, BY))
dVt = np.diff(ref_tr[:, :, 2], axis=1).max()
print("target operator itself: max step increase of V %.1e" % dVt)
zt, xt, yt = sample(random.PRNGKey(99), 20000)
res = {}
NIT = 12000
for m in [4, 16, 64, 256]:
    p = init(random.PRNGKey(m), m)
    opt = optax.adam(optax.cosine_decay_schedule(3e-3, NIT, 0.01))
    st = opt.init(p)

    @jit
    def step(p, st, k):
        z, x, y = sample(k, 1024)
        l, g = value_and_grad(lambda q: jnp.mean(field_err(q, z, x, y)[0]))(p)
        up, st = opt.update(g, st)
        return optax.apply_updates(p, up), st, l

    key = random.PRNGKey(1000 + m)
    for i in range(NIT):
        key, k = random.split(key)
        p, st, l = step(p, st, k)
    mse, sup = jit(field_err)(p, zt, xt, yt)
    tr = np.asarray(jit(vmap(lambda c, l_, r, x: closed_loop(comp_nn(p), (c, l_, r, x))))(*tasks))
    res[m] = dict(rmse=float(jnp.sqrt(jnp.mean(mse))), sup=float(sup.max()), du=float(np.abs(tr[:, :, 1] - ref_tr[:, :, 1]).max()),
                  de=float(np.abs(tr[:, :, 0] - ref_tr[:, :, 0]).max()), dz=float(np.abs(tr[:, :, 3:] - ref_tr[:, :, 3:]).max()),
                  dV=float(np.diff(tr[:, :, 2], axis=1).max()))
    r = res[m]
    print(f"width {m:4d}: field RMSE {r['rmse']:.4f}, sup error on box {r['sup']:.3f}; closed loop (10 s, 20 tasks): max |u - u_target| {r['du']:.4f}, "
          f"max |e - e_target| {r['de']:.5f}, max |z - z_target| {r['dz']:.4f}; certificate of the approximant: max step increase of V {r['dV']:.1e}", flush=True)
pickle.dump(res, open("ua_approx.pkl", "wb"))
