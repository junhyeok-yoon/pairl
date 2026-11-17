import numpy as np, jax, jax.numpy as jnp, json
from jax import jit, vmap, grad
jax.config.update('jax_enable_x64', True)
import rl_compare as R

# Lower bound on the mean joint error reachable by any motion that the tolerance hierarchy permits on the test tasks:
# eta = r_g + d with ||dr_g/dt|| <= vbar and ||d|| <= psi_max (torque limits, clearance and plant dynamics dropped),
# position from dq = eta - k1 q. The problem is convex; projected gradient (Adam) on all tasks at once.
th, q0, tgt = R.sample_tasks(np.random.default_rng(777), R.N_TEST)
DTB, T = 0.005, int(round(R.T_EP / 0.005))
tq = np.where((np.arange(T) * DTB < R.T_SW)[None, :, None], tgt[:, 0][:, None, :], tgt[:, 1][:, None, :])

def proj(z, rad):
    n = jnp.sqrt(jnp.sum(z ** 2, -1, keepdims=True))
    return z * jnp.minimum(1.0, rad / jnp.maximum(n, 1e-12))

def traj(w, d, q0):
    def step(c, wd):
        q, r = c
        wt, dt_ = wd
        qn = q + DTB * (r + dt_ - R.K1 * q)
        return (qn, r + DTB * wt), qn
    _, qs = jax.lax.scan(step, (q0, R.K1 * q0), (w, d))
    return qs

def cost(w, d, q0, tq):
    e = traj(w, d, q0) - tq
    return jnp.mean(jnp.sqrt(jnp.sum(e ** 2, -1) + 1e-10))

vcost = jit(vmap(cost)); vgrad = jit(vmap(grad(cost, argnums=(0, 1))))
out = {}
for vbar, psimax in [(4.0, 0.4), (4.0, 1.0), (6.0, 0.4)]:
    w = jnp.zeros((R.N_TEST, T, 2)); d = jnp.zeros((R.N_TEST, T, 2))
    mw = vw = jnp.zeros_like(w); md = vd = jnp.zeros_like(d)
    for it in range(1, 3001):
        gw, gd = vgrad(w, d, jnp.asarray(q0), jnp.asarray(tq))
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw ** 2
        md = 0.9 * md + 0.1 * gd; vd = 0.999 * vd + 0.001 * gd ** 2
        lr = 2.0 if it < 2000 else 0.5
        w = proj(w - lr * (mw / (1 - 0.9 ** it)) / (jnp.sqrt(vw / (1 - 0.999 ** it)) + 1e-8), vbar)
        d = proj(d - 0.2 * lr * (md / (1 - 0.9 ** it)) / (jnp.sqrt(vd / (1 - 0.999 ** it)) + 1e-8), psimax)
        if it in (1000, 2000, 3000):
            c = np.asarray(vcost(w, d, jnp.asarray(q0), jnp.asarray(tq)))
            print(f'vbar {vbar} psi_max {psimax} iter {it}: mean bound {c.mean():.4f}', flush=True)
    out[f'vbar{vbar:g}_psi{psimax:g}'] = dict(mean=float(c.mean()), per_task=[float(x) for x in c])
json.dump(out, open('../records/bound_reference.json', 'w'), indent=1)
