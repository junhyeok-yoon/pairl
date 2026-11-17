import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

jax.config.update("jax_enable_x64", True)

N = 7
NA = 10 * N + N            # 10 inertial parameters per link + viscous friction per joint
NQ = 2 * N                 # dissipative (feature) states; equals the oracle basis size
NZ = NA + NQ
G0 = 9.81
LAM = 5.0
DT_NOISE = 0.005
SIG_N = jnp.concatenate([0.001 * jnp.ones(N), 0.01 * jnp.ones(N)])
# Franka Panda kinematics, Craig (modified) DH: (a_{i-1}, d_i, alpha_{i-1})
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
D_FLANGE = 0.107
ARM = jnp.array([0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1])     # reflected rotor inertia, assumed known
MASS = np.array([4.97, 0.65, 3.23, 3.59, 1.23, 1.67, 0.74])
COM = np.array([[0.004, 0.002, -0.048], [-0.003, -0.029, 0.003], [0.028, 0.039, -0.067], [-0.053, 0.104, 0.027],
                [-0.012, 0.041, -0.038], [0.060, -0.014, -0.011], [0.011, -0.004, 0.062]])
LEN = np.array([0.30, 0.15, 0.30, 0.20, 0.35, 0.15, 0.12])
Q0 = jnp.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8])
JSC = jnp.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])     # per-joint scale of unmodeled friction


def skew(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def inertial(m, c, Ic):
    S = skew(c)
    Io = Ic + m * S.T @ S
    return jnp.array([m, m * c[0], m * c[1], m * c[2], Io[0, 0], Io[0, 1], Io[0, 2], Io[1, 1], Io[1, 2], Io[2, 2]])


def params(mp, cp, visc):
    links = []
    for k in range(N):
        r, l = 0.06, LEN[k]
        Ic = MASS[k] * jnp.diag(jnp.array([(3 * r ** 2 + l ** 2) / 12, (3 * r ** 2 + l ** 2) / 12, r ** 2 / 2]))
        links.append(inertial(MASS[k], jnp.asarray(COM[k]), Ic))
    a = jnp.concatenate(links)
    a = a.at[60:70].add(inertial(mp, cp, jnp.zeros((3, 3))))
    return jnp.concatenate([a, visc])


A_NOM = params(0.0, jnp.array([0.0, 0.0, 0.15]), 0.3 * jnp.ones(N))
PRIOR = params(1.5, jnp.array([0.0, 0.0, 0.147]), 0.55 * jnp.ones(N)) - A_NOM
SCA = jnp.concatenate([0.01 * jnp.ones(60), jnp.array([1.5, 0.05, 0.05, 0.3, 0.05, 0.01, 0.01, 0.05, 0.01, 0.02]), 0.5 * jnp.ones(N)])


EPS = np.zeros((3, 3, 3))
for _i, _j, _k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[_i, _j, _k] = 1.0
    EPS[_i, _k, _j] = -1.0
EPS = jnp.asarray(EPS)
MASK = jnp.asarray(np.tril(np.ones((N, N))))            # MASK[k, j] = 1 if joint j moves link k
IDX = np.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])


def fk(q):
    T = jnp.eye(4)
    Rs, ps = [], []
    for i in range(N):
        a, d, al = DH[i]
        ca, sa = np.cos(al), np.sin(al)
        ct, st = jnp.cos(q[i]), jnp.sin(q[i])
        A = jnp.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0.0, 0.0, 0.0, 1.0]])
        T = T @ A
        Rs.append(T[:3, :3])
        ps.append(T[:3, 3])
    return jnp.stack(Rs), jnp.stack(ps)


def mass_matrix(q, a):
    Rs, ps = fk(q)
    z = Rs[:, :, 2]
    cr = jnp.cross(z[None, :, :], ps[:, None, :] - ps[None, :, :]) * MASK[:, :, None]     # (link k, joint j, 3)
    Jv = jnp.transpose(cr, (0, 2, 1))
    Jw = jnp.transpose(z[None, :, :] * MASK[:, :, None], (0, 2, 1))
    A = a[:10 * N].reshape(N, 10)
    hw = jnp.einsum('kij,kj->ki', Rs, A[:, 1:4])
    Sh = -jnp.einsum('ijl,kl->kij', EPS, hw)
    Iw = jnp.einsum('kia,kab,kjb->kij', Rs, A[:, IDX], Rs)
    X = jnp.einsum('kai,kab,kbj->ij', Jv, Sh, Jw)
    return jnp.einsum('k,kai,kaj->ij', A[:, 0], Jv, Jv) - X - X.T + jnp.einsum('kai,kab,kbj->ij', Jw, Iw, Jw)


def potential(q, a):
    Rs, ps = fk(q)
    A = a[:10 * N].reshape(N, 10)
    return G0 * (A[:, 0] @ ps[:, 2] + jnp.sum(jnp.einsum('kij,kj->ki', Rs, A[:, 1:4])[:, 2]))


def tau_r(a, q, dq, dqr, ddqr):
    # Y(q, dq, dqr, ddqr) a = M ddqr + C(q, dq) dqr + g + F dqr, Christoffel C (dM - 2C skew); linear in a
    Mq = lambda qq: mass_matrix(qq, a)
    M, Md = jax.jvp(Mq, (q,), (dq,))
    t2 = jax.jvp(lambda qq: Mq(qq) @ dq, (q,), (dqr,))[1]
    t3 = jax.grad(lambda qq: dqr @ Mq(qq) @ dq)(q)
    g = jax.grad(potential)(q, a)
    return M @ ddqr + 0.5 * (Md @ dqr + t2 - t3) + g + a[10 * N:] * dqr


def tau_u(dq, eta):
    t = jnp.tanh(dq / 0.05)
    return eta[:N] * t + eta[N:] * t * jnp.exp(-(dq / 0.3) ** 2)


def ref(t, rp):
    A, w, ph = rp[:, 0], rp[:, 1], rp[:, 2]
    return Q0 + A * jnp.sin(w * t + ph) - A * jnp.sin(ph), A * w * jnp.cos(w * t + ph), -A * w ** 2 * jnp.sin(w * t + ph)


def softplus(x):
    return jnp.logaddexp(x, 0.0)


# controllers: sig = (q, dq, qt, sv, dqr, ddqr)

def sl_op(p, z, sig):
    q, dq, qt, sv, dqr, ddqr = sig
    ah = A_NOM + PRIOR + z[:NA]
    tau, vjp = jax.vjp(lambda aa: tau_r(aa, q, dq, dqr, ddqr), ah)
    zdot = -p["gamma"] * SCA ** 2 * vjp(sv)[0]
    return jnp.concatenate([zdot, jnp.zeros(NQ)]), tau + ARM * ddqr - p["k"] * sv


def phi_oracle(dq):
    t = jnp.tanh(dq / 0.05)
    e = jnp.exp(-(dq / 0.3) ** 2)
    i = jnp.arange(N)
    return jnp.zeros((NQ, N)).at[i, i].set(t).at[N + i, i].set(t * e)


def oracle_op(p, z, sig):
    q, dq, qt, sv, dqr, ddqr = sig
    zd, tau = sl_op(p, z, sig)
    Po = phi_oracle(dq)
    return zd.at[NA:].set(-p["gb"] * Po @ sv), tau + Po.T @ z[NA:]


def init_ph(key, gamma0=10.0, k0=10.0, scale=0.05, nh=12, nm=32, rank=4):
    ks = random.split(key, 12)
    return {
        "a0": PRIOR, "la": jnp.log(gamma0 * SCA ** 2), "U": scale * 0.1 * random.normal(ks[0], (NA, rank)),
        "Lf": 0.5 * jnp.eye(NQ),
        "A0": scale * random.normal(ks[1], (nh, NZ)), "b0": jnp.zeros(nh),
        "U1": -3.0 + scale * random.normal(ks[2], (nh, nh)),
        "A1": scale * random.normal(ks[3], (nh, NZ)), "b1": jnp.zeros(nh),
        "u2": -3.0 + scale * random.normal(ks[4], (nh,)),
        "W1": 0.5 * random.normal(ks[5], (nm, 5 * N)), "c1": jnp.zeros(nm),
        "Wphi": 0.3 * random.normal(ks[6], (NQ * N, nm)), "bphi": jnp.zeros(NQ * N),
        "WR": scale * random.normal(ks[7], (NQ * NQ, nm)), "bR": 0.3 * jnp.eye(NQ).reshape(-1),
        "WJ": scale * random.normal(ks[8], (NQ * NQ, nm)), "bJ": jnp.zeros(NQ * NQ),
        "WK": scale * random.normal(ks[9], (N * N, nm)), "bK": (jnp.sqrt(k0) * jnp.eye(N)).reshape(-1),
    }


def H_fn(p, z):
    za, zf = z[:NA], z[NA:]
    Mf = p["Lf"] @ p["Lf"].T + 1e-3 * jnp.eye(NQ)
    quad = 0.5 * za @ (jnp.exp(p["la"]) * za) + 0.5 * jnp.sum((p["U"].T @ za) ** 2) + 0.5 * zf @ Mf @ zf
    h1 = softplus(p["A0"] @ z + p["b0"])
    h2 = softplus(softplus(p["U1"]) @ h1 + p["A1"] @ z + p["b1"])
    return quad + softplus(p["u2"]) @ h2


grad_H = jax.grad(H_fn, argnums=1)


def ph_terms(p, q, dq, qt, sv):
    m = jnp.tanh(p["W1"] @ jnp.concatenate([jnp.sin(q), jnp.cos(q), dq, qt, sv]) + p["c1"])
    phi = (p["Wphi"] @ m + p["bphi"]).reshape(NQ, N)
    LR = (p["WR"] @ m + p["bR"]).reshape(NQ, NQ)
    A = (p["WJ"] @ m + p["bJ"]).reshape(NQ, NQ)
    LK = (p["WK"] @ m + p["bK"]).reshape(N, N)
    return phi, LR @ LR.T + 1e-3 * jnp.eye(NQ), A - A.T, LK @ LK.T


def ph_op(p, z, sig):
    q, dq, qt, sv, dqr, ddqr = sig
    g = grad_H(p, z)
    ga, gf = g[:NA], g[NA:]
    tau, vjp = jax.vjp(lambda aa: tau_r(aa, q, dq, dqr, ddqr), A_NOM + p["a0"] + ga)
    phi, R, J, K = ph_terms(p, q, dq, qt, sv)
    zdot = jnp.concatenate([-vjp(sv)[0], (J - R) @ gf - phi @ sv])
    return zdot, tau + ARM * ddqr + phi.T @ gf - K @ sv


# closed loop

def signals(t, q, dq, rp):
    qd, dqd, ddqd = ref(t, rp)
    qt, dqt = q - qd, dq - dqd
    return qt, dqt + LAM * qt, dqd - LAM * qt, ddqd - LAM * dqt


def plant_acc(q, dq, tau, a, eta):
    M = mass_matrix(q, a) + jnp.diag(ARM)
    return jnp.linalg.solve(M, tau - tau_r(a, q, dq, dq, jnp.zeros(N)) - tau_u(dq, eta))


def make_rhs(op):
    def rhs(p, t, s, task, n):
        a, eta, rp = task
        q, dq, z = s[:N], s[N:2 * N], s[2 * N:]
        qt, sv, dqr, ddqr = signals(t, q + n[:N], dq + n[N:], rp)
        zdot, tau = op(p, z, (q + n[:N], dq + n[N:], qt, sv, dqr, ddqr))
        return jnp.concatenate([dq, plant_acc(q, dq, tau, a, eta), zdot]), tau
    return rhs


def aux_out(t, s, tau, task):
    # true tracking error, torque deviation from the ideal certainty-equivalence torque, true sliding variable
    a, eta, rp = task
    q, dq = s[:N], s[N:2 * N]
    qt0, sv0, dqr0, ddqr0 = signals(t, q, dq, rp)
    return jnp.concatenate([qt0, tau - tau_r(a, q, dq, dqr0, ddqr0) - ARM * ddqr0 - tau_u(dq, eta), sv0])


def make_sim(op, nz, dt, nsteps, rho=1e-4, switch=None, full=False, method="rk4"):
    rhs = make_rhs(op)

    def sim(p, task, noise, task2=None, q0err=None):
        def cur(t):
            if switch is None:
                return task
            return jax.tree_util.tree_map(lambda u, v: jnp.where(t < switch, u, v), task, task2)

        def step(s, i):
            t = i * dt
            tk = cur(t)
            n = noise[jnp.minimum((t / DT_NOISE).astype(int), noise.shape[0] - 1)]
            k1, tau = rhs(p, t, s, tk, n)
            aux = aux_out(t, s, tau, tk)
            if method == "euler":
                sn = s + dt * k1
            elif method == "rk2":
                k2, _ = rhs(p, t + dt / 2, s + dt / 2 * k1, tk, n)
                sn = s + dt * k2
            else:
                k2, _ = rhs(p, t + dt / 2, s + dt / 2 * k1, tk, n)
                k3, _ = rhs(p, t + dt / 2, s + dt / 2 * k2, tk, n)
                k4, _ = rhs(p, t + dt, s + dt * k3, tk, n)
                sn = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return sn, (jnp.concatenate([aux, s]) if full else aux)

        qd0, dqd0, _ = ref(0.0, task[2])
        q0 = qd0 if q0err is None else qd0 + q0err
        s0 = jnp.concatenate([q0, dqd0, jnp.zeros(nz)])
        _, out = lax.scan(step, s0, jnp.arange(nsteps))
        return jnp.mean(jnp.sum(out[:, :N] ** 2, axis=1) + rho * jnp.sum(out[:, N:2 * N] ** 2, axis=1)), out
    return sim


def sample_tasks(key, n, mp_hi=3.0, visc_hi=1.0, un=1.0, amp=0.3, w_hi=2.0):
    ks = random.split(key, 9)
    mp = random.uniform(ks[0], (n,), minval=0.0, maxval=mp_hi)
    cp = jnp.concatenate([random.uniform(ks[1], (n, 2), minval=-0.03, maxval=0.03),
                          D_FLANGE + random.uniform(ks[2], (n, 1), minval=0.0, maxval=0.08)], axis=1)
    visc = random.uniform(ks[3], (n, N), minval=0.1, maxval=visc_hi)
    a = jax.vmap(params)(mp, cp, visc)
    eta = un * jnp.concatenate([random.uniform(ks[4], (n, N), maxval=2.0) * JSC, random.uniform(ks[5], (n, N), maxval=1.0) * JSC], axis=1)
    rp = jnp.stack([random.uniform(ks[6], (n, N), maxval=amp), random.uniform(ks[7], (n, N), minval=0.5, maxval=w_hi),
                    random.uniform(ks[8], (n, N), maxval=2 * np.pi)], axis=2)
    return (a, eta, rp)
