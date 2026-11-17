import jax, jax.numpy as jnp, numpy as np, time
exec(open('r3_timing.py').read().split("f = jax.jit")[0])
ROT = 0.02
perp = lambda r: jnp.array([-r[1], r[0]])
cross2 = lambda a, b: a[0] * b[1] - a[1] * b[0]
def rnea(q, dq, ddq, mp, grav, fext):
    th, w, al = jnp.cumsum(q), jnp.cumsum(dq), jnp.cumsum(ddq)
    u = jnp.stack([jnp.cos(th), jnp.sin(th)], 1)
    a_o = jnp.array([0.0, G0 * grav]); acs, rs, rcs = [], [], []
    for i in range(3):
        r, rc = L[i] * u[i], LC[i] * u[i]
        acs.append(a_o + al[i] * perp(rc) - w[i] ** 2 * rc); rs.append(r); rcs.append(rc)
        a_o = a_o + al[i] * perp(r) - w[i] ** 2 * r
    f, n, tau = mp * a_o - fext, 0.0, [None] * 3
    for i in reversed(range(3)):
        n = IK[i] * al[i] + n + cross2(rcs[i], MK[i] * acs[i]) + cross2(rs[i], f)
        f = MK[i] * acs[i] + f; tau[i] = n
    return jnp.array(tau) + ROT * ddq
def acc_rnea(q, dq, tau, mp, fext):
    M = jax.vmap(lambda e: rnea(q, jnp.zeros(3), e, mp, 0.0, jnp.zeros(2)))(jnp.eye(3))
    hg = rnea(q, dq, jnp.zeros(3), mp, 1.0, fext)
    return jnp.linalg.solve(M, tau - 0.3 * dq - hg)
def acc_ref(q, dq, tau, mp, fext):
    M, Jt = energy_terms(q, dq, mp)
    dM = jax.jacfwd(lambda x: energy_terms(x, dq, mp)[0])(q)
    h = jnp.einsum('klj,j,l->k', dM, dq, dq) - 0.5 * jnp.einsum('jlk,j,l->k', dM, dq, dq)
    return jnp.linalg.solve(M, tau - h - jax.grad(V)(q, mp) - 0.3 * dq + Jt.T @ fext)
rng = np.random.default_rng(0)
err = max(float(jnp.max(jnp.abs(acc_rnea(*a) - acc_ref(*a)))) for a in
          [(jnp.array(rng.uniform(-2, 2, 3)), jnp.array(rng.uniform(-3, 3, 3)), jnp.array(rng.uniform(-20, 20, 3)), float(rng.uniform(0, 2)), jnp.array(rng.uniform(-30, 30, 2))) for _ in range(200)])
print('max |acc_rnea - acc_autodiff| over 200 random states: %.2e' % err)
def tipkin(q, dq):
    th, w = jnp.cumsum(q), jnp.cumsum(dq); u = jnp.stack([jnp.cos(th), jnp.sin(th)], 1)
    r = L[:, None] * u
    return jnp.sum(r, 0), jnp.sum(w[:, None] * jax.vmap(perp)(r), 0)
def run2(q0, mp, ys, nstep, sub, dt=1e-3):
    def fx(q, dq, tau):
        tip, vt = tipkin(q, dq); pen = ys - tip[1]
        fn = jnp.where(pen > 0, jnp.maximum(3000.0 * pen - 20.0 * vt[1], 0.0), 0.0)
        return acc_rnea(q, dq, tau, mp, jnp.array([-0.5 * fn * jnp.tanh(vt[0] / 0.01), fn]))
    def step(c, k):
        q, dq = c
        tau = rnea(q, jnp.zeros(3), jnp.zeros(3), mp, 1.0, jnp.zeros(2)) + 30.0 * (jnp.array([0.3, 0.8, -0.6]) - q) - 3.0 * dq
        h = dt / sub
        for _ in range(sub):
            a1 = fx(q, dq, tau); a2 = fx(q + 0.5 * h * dq, dq + 0.5 * h * a1, tau)
            a3 = fx(q + 0.5 * h * (dq + 0.5 * h * a1), dq + 0.5 * h * a2, tau); a4 = fx(q + h * (dq + 0.5 * h * a2), dq + h * a3, tau)
            q, dq = q + h * (dq + h / 6 * (a1 + a2 + a3)), dq + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        return (q, dq), q
    return jax.lax.scan(step, (q0, jnp.zeros(3)), jnp.arange(nstep))[1][-1]
N, S = 256, 2000
q0 = jnp.tile(jnp.array([0.2, 0.9, -0.5]), (N, 1)); mp = jnp.linspace(0, 2, N); ys = jnp.full(N, -0.3)
for sub in (4, 2):
    g = jax.jit(jax.vmap(run2, in_axes=(0, 0, 0, None, None)), static_argnums=(3, 4))
    g(q0, mp, ys, S, sub).block_until_ready(); t1 = time.time(); g(q0, mp, ys, S, sub).block_until_ready(); t2 = time.time()
    print('RNEA, %d RK4 substeps: %.2f s for %d x %d steps -> %.2f M episode-steps/s' % (sub, t2 - t1, N, S, N * S / (t2 - t1) / 1e6))
