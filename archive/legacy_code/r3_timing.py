import jax, jax.numpy as jnp, numpy as np, time
L = jnp.array([0.4, 0.35, 0.25]); MK = jnp.array([3.0, 2.0, 1.0]); IK = MK * L ** 2 / 12; LC = L / 2; G0 = 9.81
def kin(q):
    th = jnp.cumsum(q)
    c, s = jnp.cos(th), jnp.sin(th)
    joints = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(jnp.stack([L * c, L * s], 1), 0)], 0)
    com = joints[:3] + jnp.stack([LC * c, LC * s], 1)
    return com, joints[3]
def energy_terms(q, dq, mp):
    Jc = jax.jacfwd(lambda x: kin(x)[0])(q)
    Jt = jax.jacfwd(lambda x: kin(x)[1])(q)
    Jw = jnp.tril(jnp.ones((3, 3)))
    M = jnp.einsum('i,iak,ial->kl', MK, Jc, Jc) + jnp.einsum('i,ik,il->kl', IK, Jw, Jw) + mp * Jt.T @ Jt + 0.02 * jnp.eye(3)
    return M, Jt
def V(q, mp):
    com, tip = kin(q)
    return G0 * (jnp.sum(MK * com[:, 1]) + mp * tip[1])
def acc(q, dq, tau, mp, ys):
    M, Jt = energy_terms(q, dq, mp)
    dM = jax.jacfwd(lambda x: energy_terms(x, dq, mp)[0])(q)
    h = jnp.einsum('klj,j,l->k', dM, dq, dq) - 0.5 * jnp.einsum('jlk,j,l->k', dM, dq, dq)
    g = jax.grad(V)(q, mp)
    tip = kin(q)[1]; vt = Jt @ dq
    pen = ys - tip[1]
    fn = jnp.where(pen > 0, jnp.maximum(3000.0 * pen - 20.0 * vt[1], 0.0), 0.0)
    ft = -0.5 * fn * jnp.tanh(vt[0] / 0.01)
    return jnp.linalg.solve(M, tau - h - g - 0.3 * dq + Jt.T @ jnp.array([ft, fn]))
def run(q0, mp, ys, nstep, sub=4, dt=1e-3):
    def step(c, k):
        q, dq = c
        tau = jax.grad(V)(q, mp) + 30.0 * (jnp.array([0.3, 0.8, -0.6]) - q) - 3.0 * dq
        h = dt / sub
        for _ in range(sub):
            a1 = acc(q, dq, tau, mp, ys); a2 = acc(q + 0.5 * h * dq, dq + 0.5 * h * a1, tau, mp, ys)
            a3 = acc(q + 0.5 * h * (dq + 0.5 * h * a1), dq + 0.5 * h * a2, tau, mp, ys); a4 = acc(q + h * (dq + 0.5 * h * a2), dq + h * a3, tau, mp, ys)
            q, dq = q + h * (dq + h / 6 * (a1 + a2 + a3)), dq + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        return (q, dq), q
    _, qs = jax.lax.scan(step, (q0, jnp.zeros(3)), jnp.arange(nstep))
    return qs[-1]
f = jax.jit(jax.vmap(run, in_axes=(0, 0, 0, None)), static_argnums=3)
N, S = 256, 2000
q0 = jnp.tile(jnp.array([0.2, 0.9, -0.5]), (N, 1)); mp = jnp.linspace(0, 2, N); ys = jnp.full(N, -0.3)
t0 = time.time(); f(q0, mp, ys, S).block_until_ready(); t1 = time.time(); f(q0, mp, ys, S).block_until_ready(); t2 = time.time()
print('compile+run %.1f s; run %.2f s for %d episodes x %d steps (1 ms, 4 RK4 substeps) -> %.2f M episode-steps/s' % (t1 - t0, t2 - t1, N, S, N * S / (t2 - t1) / 1e6))
