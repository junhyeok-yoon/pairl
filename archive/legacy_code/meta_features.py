import numpy as np, jax, jax.numpy as jnp, json
from jax import jit, vmap, grad
jax.config.update('jax_enable_x64', True)
import rl_compare as R
from lib_meta import meta_phi, N1, N2, NF_META

# Offline meta-learning of the core's feature map on samples of the plant class (no plant model inside the controller):
# each task has its own least-squares weights; the shared feature map minimizes the per-task residual.
def ustar(q, dq, a, th):
    a1, a2, a3, g1, g2, f1, f2 = th
    c, s = jnp.cos(q[1]), jnp.sin(q[1]); h = a2 * s
    m11 = a1 + 2 * a2 * c + R.ROT[0]; m12 = a3 + a2 * c; m22 = a3 + R.ROT[1]
    n1 = -h * dq[1] * dq[0] - h * (dq[0] + dq[1]) * dq[1] + g1 * jnp.cos(q[0]) + g2 * jnp.cos(q[0] + q[1]) + f1 * dq[0]
    n2 = h * dq[0] ** 2 + g2 * jnp.cos(q[0] + q[1]) + f2 * dq[1]
    return jnp.stack([m11 * a[0] + m12 * a[1] + n1, m12 * a[0] + m22 * a[1] + n2])

def sample(rng, K, N):
    th = R.theta_np(rng.uniform(0, R.PMAX, K), rng.uniform(0.2, 1, K), rng.uniform(0.2, 1, K))
    q = rng.uniform(R.LO - 0.1, R.HI + 0.1, (K, N, 2)); dq = rng.uniform(-2.5, 2.5, (K, N, 2)); a = rng.uniform(-8, 8, (K, N, 2))
    U = vmap(vmap(ustar, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))(q, dq, a, th)
    return q, dq, a, U

def task_fit(Xk, Uk, lam=1e-6):
    A = Xk.T @ Xk; W = jnp.linalg.solve(A + lam * jnp.trace(A) / A.shape[0] * jnp.eye(A.shape[0]), Xk.T @ Uk)
    return W, jnp.sum((Xk @ W - Uk) ** 2) / jnp.sum(Uk ** 2)

feat_meta = lambda P: vmap(vmap(lambda q, dq, a: meta_phi(P, q, dq, a, R.CEN, R.HALF)))
feat_rand = vmap(vmap(R.features))

def loss(P, q, dq, a, U):
    X = feat_meta(P)(q, dq, a)
    return jnp.mean(vmap(task_fit)(X, U)[1])

rng = np.random.default_rng(0)
shapes = [(32, 4), (32,), (32, 32), (32,), (N1, 32), (N1,), (32, 2), (32,), (32, 32), (32,), (N2, 32), (N2,)]
P = [jnp.asarray(rng.normal(0, 1 / np.sqrt(s[1]), s)) if len(s) == 2 else jnp.zeros(s) for s in shapes]
m = [jnp.zeros_like(p) for p in P]; v = [jnp.zeros_like(p) for p in P]
g_fn = jit(grad(loss)); l_fn = jit(loss)
for it in range(1, 2001):
    batch = sample(rng, 32, 48)
    g = g_fn(P, *batch)
    m = [0.9 * mi + 0.1 * gi for mi, gi in zip(m, g)]; v = [0.999 * vi + 0.001 * gi ** 2 for vi, gi in zip(v, g)]
    lr = 3e-3 if it < 1500 else 1e-3
    P = [p - lr * (mi / (1 - 0.9 ** it)) / (jnp.sqrt(vi / (1 - 0.999 ** it)) + 1e-8) for p, mi, vi in zip(P, m, v)]
    if it % 500 == 0:
        print('step', it, 'train relative residual', float(l_fn(P, *batch)), flush=True)
val = sample(np.random.default_rng(99), 64, 200)
Xm = feat_meta(P)(*val[:3]); Xr = feat_rand(*val[:3])
Wm, rm = vmap(task_fit)(Xm, val[3]); Wr, rr = vmap(task_fit)(Xr, val[3])
res = dict(nf_meta=NF_META, nf_random=int(R.NF), rel_residual_meta=float(jnp.mean(rm)), rel_residual_meta_max=float(jnp.max(rm)),
           rel_residual_random=float(jnp.mean(rr)), rel_residual_random_max=float(jnp.max(rr)))
print(json.dumps(res))
np.savez('../records/meta_features.npz', nf=NF_META, w_mean=np.asarray(jnp.mean(Wm, 0)), **{f'p{i:02d}': np.asarray(p) for i, p in enumerate(P)})
json.dump(res, open('../records/meta_features.json', 'w'), indent=1)
