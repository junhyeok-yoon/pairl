import jax
import jax.numpy as jnp
from jax import lax, random
from arm2 import (params, MCg, Y_reg, Y_h, Y_g, ref, softplus, inv_softplus, A_NOM, SC, LAM_S, TAUF, KE_MAX,
                  DT_NOISE, SIG_N, L1, L2, G0)

NP_, NQ = 7, 8            # rigid-body/viscous parameters, states driven by learned features
NZ = NP_ + NQ
NU = 16
OFF = 4 + 14 + 14 + 2
NF = 12                   # feature-input dimension
# mean of the training task distribution minus the nominal model (payload U[0,3] kg, viscous friction U[0.1,1])
PRIOR = jax.vmap(params)(jnp.linspace(0.0, 3.0, 301), jnp.full((301, 2), 0.55)).mean(0) - A_NOM
SCB = jnp.array([1.0, 0.5, 3.0, 1.5, 1.0, 0.5, 3.0, 1.5])


def tau_u(q, dq, eta):
    # torques outside the classical regressor: Coulomb + Stribeck friction, elastic cable
    fc, fs, kc, q0 = eta[0:2], eta[2:4], eta[4:6], eta[6:8]
    t = jnp.tanh(dq / 0.05)
    return fc * t + fs * t * jnp.exp(-(dq / 0.3) ** 2) + kc * (q - q0)


def feat(qm, dqm, qt, sv):
    return jnp.concatenate([jnp.sin(qm), jnp.cos(qm), qm, dqm, qt, sv])


# ---------- pH adaptation operator with learned features ----------

def init_ph(key, gamma0=3.0, k0=4.0, ke0=1.0, scale=0.05, nh=12, nm=24):
    ks = random.split(key, 12)
    return {
        "a0": PRIOR,
        "L": jnp.diag(jnp.concatenate([jnp.sqrt(gamma0) * SC, jnp.ones(NQ)])),
        "A0": scale * random.normal(ks[0], (nh, NZ)), "b0": jnp.zeros(nh),
        "U1": -3.0 + scale * random.normal(ks[1], (nh, nh)),
        "A1": scale * random.normal(ks[2], (nh, NZ)), "b1": jnp.zeros(nh),
        "u2": -3.0 + scale * random.normal(ks[3], (nh,)),
        "W1": 0.5 * random.normal(ks[4], (nm, NF)), "c1": jnp.zeros(nm),
        "Wphi": 0.3 * random.normal(ks[5], (NQ * 2, nm)), "bphi": jnp.zeros(NQ * 2),
        "WR": scale * random.normal(ks[6], (NQ * NQ, nm)), "bR": 0.3 * jnp.eye(NQ).reshape(-1),
        "WJ": scale * random.normal(ks[7], (NQ * NQ, nm)), "bJ": jnp.zeros(NQ * NQ),
        "WK": scale * random.normal(ks[8], (4, nm)), "bK": (jnp.sqrt(k0) * jnp.eye(2)).reshape(-1),
        "vk": scale * random.normal(ks[10], (nm,)),
        "dk": jnp.log(jnp.array(max(ke0, 1e-3)) / (KE_MAX - max(ke0, 1e-3))),
    }


def H_fn(p, z, mu=1e-3):
    M = p["L"] @ p["L"].T + mu * jnp.eye(NZ)
    h1 = softplus(p["A0"] @ z + p["b0"])
    h2 = softplus(softplus(p["U1"]) @ h1 + p["A1"] @ z + p["b1"])
    return 0.5 * z @ M @ z + softplus(p["u2"]) @ h2


grad_H = jax.grad(H_fn, argnums=1)


def ph_terms(p, qm, dqm, qt, sv):
    m = jnp.tanh(p["W1"] @ feat(qm, dqm, qt, sv) + p["c1"])
    phi = (p["Wphi"] @ m + p["bphi"]).reshape(NQ, 2)
    LR = (p["WR"] @ m + p["bR"]).reshape(NQ, NQ)
    A = (p["WJ"] @ m + p["bJ"]).reshape(NQ, NQ)
    LK = (p["WK"] @ m + p["bK"]).reshape(2, 2)
    ke = KE_MAX * jax.nn.sigmoid(p["vk"] @ m + p["dk"])
    return phi, LR @ LR.T + 1e-3 * jnp.eye(NQ), A - A.T, LK @ LK.T, ke


def ph_op(p, z, sig):
    qm, dqm, qt, sv, Y, Wf, tgt = sig
    g = grad_H(p, z)
    ga, gf = g[:NP_], g[NP_:]
    ah = A_NOM + p["a0"] + ga
    phi, R, J, K, ke = ph_terms(p, qm, dqm, qt, sv)
    eps = Wf @ ah - tgt
    zdot = jnp.concatenate([-Y.T @ sv - ke / (1.0 + jnp.sum(Wf * Wf)) * (Wf.T @ eps), (J - R) @ gf - phi @ sv])
    return zdot, Y @ ah + phi.T @ gf - K @ sv


# ---------- classical baselines ----------

def sl_op(p, z, sig):
    # composite Slotine-Li with the rigid-body regressor only (prior mean as initial estimate)
    qm, dqm, qt, sv, Y, Wf, tgt = sig
    ah = A_NOM + PRIOR + z[:NP_]
    eps = Wf @ ah - tgt
    zdot = -p["gamma"] * SC ** 2 * (Y.T @ sv + p["ke"] / (1.0 + jnp.sum(Wf * Wf)) * (Wf.T @ eps))
    return jnp.concatenate([zdot, jnp.zeros(NZ - NP_)]), Y @ ah - p["k"] * sv


def phi_oracle(qm, dqm):
    t = jnp.tanh(dqm / 0.05)
    e = jnp.exp(-(dqm / 0.3) ** 2)
    z = 0.0
    return jnp.array([[t[0], t[0] * e[0], qm[0], 1.0, z, z, z, z], [z, z, z, z, t[1], t[1] * e[1], qm[1], 1.0]])


def oracle_op(p, z, sig):
    # same law plus the exact basis of the unmodeled torques (what an engineer who knew their structure would build)
    qm, dqm, qt, sv, Y, Wf, tgt = sig
    ah = A_NOM + PRIOR + z[:NP_]
    Po = phi_oracle(qm, dqm)
    eps = Wf @ ah - tgt
    zdot = -p["gamma"] * SC ** 2 * (Y.T @ sv + p["ke"] / (1.0 + jnp.sum(Wf * Wf)) * (Wf.T @ eps))
    bdot = -p["gb"] * SCB ** 2 * (Po.T @ sv)
    return jnp.concatenate([zdot, bdot]), Y @ ah + Po @ z[NP_:] - p["k"] * sv


# ---------- unstructured learned law ----------
NIN = NP_ + NF + 14 + 2


def init_un(key, gamma0=3.0, k0=4.0, scale=0.05, nh=32):
    ks = random.split(key, 8)
    A = jnp.zeros((NU, NIN)).at[:NP_, :NP_].set(-gamma0 * jnp.diag(SC ** 2))
    C = jnp.zeros((NP_, NU)).at[:, :NP_].set(jnp.eye(NP_))
    wl = jnp.zeros((2, NIN)).at[0, NP_ + 10].set(-k0).at[1, NP_ + 11].set(-k0)
    return {
        "a0": PRIOR, "A": A, "C": C, "wl": wl,
        "W2": scale * random.normal(ks[0], (NU, nh)),
        "W3": 0.3 * random.normal(ks[1], (nh, NU + NIN)), "b3": jnp.zeros(nh),
        "W4": scale * random.normal(ks[2], (NP_, nh)),
        "W5": 0.3 * random.normal(ks[3], (nh, NU + NIN)), "b5": jnp.zeros(nh),
        "W6": scale * random.normal(ks[4], (2, nh)),
        "W7": 0.3 * random.normal(ks[5], (nh, NU + NIN)), "b7": jnp.zeros(nh),
    }


def un_op(p, z, sig):
    qm, dqm, qt, sv, Y, Wf, tgt = sig
    inp = jnp.concatenate([Y.T @ sv, feat(qm, dqm, qt, sv), Wf.reshape(-1) / 10.0, tgt / 10.0])
    zi = jnp.concatenate([z, inp])
    zdot = p["A"] @ inp + p["W2"] @ jnp.tanh(p["W3"] @ zi + p["b3"])
    ah = A_NOM + p["a0"] + p["C"] @ z + p["W4"] @ jnp.tanh(p["W5"] @ zi + p["b5"])
    return zdot, Y @ ah + p["wl"] @ inp + p["W6"] @ jnp.tanh(p["W7"] @ zi + p["b7"])


# ---------- closed loop ----------

def make_rhs(op):
    def rhs(p, t, s, task, n):
        a, eta, rp = task
        q, dq = s[0:2], s[2:4]
        Yhf, Ygf, tauf, z = s[4:18].reshape(2, NP_), s[18:32].reshape(2, NP_), s[32:34], s[OFF:]
        qd, dqd, ddqd = ref(t, rp)
        qm, dqm = q + n[:2], dq + n[2:]
        qt, dqt = qm - qd, dqm - dqd
        sv = dqt + LAM_S * qt
        Y = Y_reg(qm, dqm, dqd - LAM_S * qt, ddqd - LAM_S * dqt)
        Yh = Y_h(qm, dqm)
        Wf = (Yh - Yhf) / TAUF - Ygf
        zdot, tau = op(p, z, (qm, dqm, qt, sv, Y, Wf, tauf))
        M, C, g = MCg(a, q, dq)
        tu = tau_u(q, dq, eta)
        b = tau - C @ dq - g - a[5:] * dq - tu
        ddq = jnp.array([M[1, 1] * b[0] - M[0, 1] * b[1], -M[0, 1] * b[0] + M[0, 0] * b[1]]) / (M[0, 0] * M[1, 1] - M[0, 1] ** 2)
        ds = jnp.concatenate([dq, ddq, ((Yh - Yhf) / TAUF).reshape(-1), ((Y_g(qm, dqm) - Ygf) / TAUF).reshape(-1),
                              (tau - tauf) / TAUF, zdot])
        Yt = Y_reg(q, dq, dqd - LAM_S * (q - qd), ddqd - LAM_S * (dq - dqd))
        return ds, jnp.concatenate([q - qd, tau - Yt @ a - tu])
    return rhs


def make_sim(op, nz, dt, nsteps, rho=1e-4, switch=None):
    rhs = make_rhs(op)

    def sim(p, task, noise, task2=None):
        def cur(t):
            if switch is None:
                return task
            return jax.tree_util.tree_map(lambda u, v: jnp.where(t < switch, u, v), task, task2)

        def step(s, i):
            t = i * dt
            tk = cur(t)
            n = noise[jnp.minimum((t / DT_NOISE).astype(int), noise.shape[0] - 1)]
            k1, aux = rhs(p, t, s, tk, n)
            k2, _ = rhs(p, t + dt / 2, s + dt / 2 * k1, tk, n)
            k3, _ = rhs(p, t + dt / 2, s + dt / 2 * k2, tk, n)
            k4, _ = rhs(p, t + dt, s + dt * k3, tk, n)
            return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), aux

        qd0, dqd0, _ = ref(0.0, task[2])
        s0 = jnp.concatenate([qd0, dqd0, Y_h(qd0, dqd0).reshape(-1), jnp.zeros(14 + 2 + nz)])
        _, out = lax.scan(step, s0, jnp.arange(nsteps))
        return jnp.mean(jnp.sum(out[:, 0:2] ** 2, axis=1) + rho * jnp.sum(out[:, 2:4] ** 2, axis=1)), out
    return sim


def sample_tasks(key, n, mp_hi=3.0, fr_hi=1.0, un=1.0, amp=0.6, w_hi=2.5):
    ks = random.split(key, 10)
    mp = random.uniform(ks[0], (n,), minval=0.0, maxval=mp_hi)
    fr = random.uniform(ks[1], (n, 2), minval=0.1, maxval=fr_hi)
    a = jax.vmap(params)(mp, fr)
    eta = jnp.concatenate([un * random.uniform(ks[6], (n, 2), minval=0.0, maxval=2.0),
                           un * random.uniform(ks[7], (n, 2), minval=0.0, maxval=1.0),
                           un * random.uniform(ks[8], (n, 2), minval=0.0, maxval=6.0),
                           random.uniform(ks[9], (n, 2), minval=-0.5, maxval=0.5)], axis=1)
    b = random.uniform(ks[2], (n, 2, 1), minval=-0.5, maxval=0.5)
    A = random.uniform(ks[3], (n, 2, 2), minval=0.0, maxval=amp)
    w = random.uniform(ks[4], (n, 2, 2), minval=0.5, maxval=w_hi)
    ph = random.uniform(ks[5], (n, 2, 2), minval=0.0, maxval=6.28)
    rp = jnp.concatenate([b, A[:, :, :1], w[:, :, :1], ph[:, :, :1], A[:, :, 1:], w[:, :, 1:], ph[:, :, 1:]], axis=2)
    return (a, eta, rp)


# ---------- widened class: coercive non-convex H, component networks also depend on z_f ----------

def init_ph_wide(key, nt=16, **kw):
    p = init_ph(key, **kw)
    k = random.split(key, 2)
    p["Wz"] = jnp.zeros((p["W1"].shape[0], NQ))
    p["HtW"] = 0.3 * random.normal(k[0], (nt, NZ))
    p["Htb"] = jnp.zeros(nt)
    p["Htw"] = jnp.zeros(nt)
    return p


def H_wide(p, z, mu=1e-3):
    return H_fn(p, z, mu) + p["Htw"] @ jnp.tanh(p["HtW"] @ z + p["Htb"])


grad_H_wide = jax.grad(H_wide, argnums=1)


def ph_terms_wide(p, qm, dqm, qt, sv, zf):
    m = jnp.tanh(p["W1"] @ feat(qm, dqm, qt, sv) + p["Wz"] @ zf + p["c1"])
    phi = (p["Wphi"] @ m + p["bphi"]).reshape(NQ, 2)
    LR = (p["WR"] @ m + p["bR"]).reshape(NQ, NQ)
    A = (p["WJ"] @ m + p["bJ"]).reshape(NQ, NQ)
    LK = (p["WK"] @ m + p["bK"]).reshape(2, 2)
    ke = KE_MAX * jax.nn.sigmoid(p["vk"] @ m + p["dk"])
    return phi, LR @ LR.T + 1e-3 * jnp.eye(NQ), A - A.T, LK @ LK.T, ke


def ph_wide_op(p, z, sig):
    qm, dqm, qt, sv, Y, Wf, tgt = sig
    g = grad_H_wide(p, z)
    ga, gf = g[:NP_], g[NP_:]
    ah = A_NOM + p["a0"] + ga
    phi, R, J, K, ke = ph_terms_wide(p, qm, dqm, qt, sv, z[NP_:])
    eps = Wf @ ah - tgt
    zdot = jnp.concatenate([-Y.T @ sv - ke / (1.0 + jnp.sum(Wf * Wf)) * (Wf.T @ eps), (J - R) @ gf - phi @ sv])
    return zdot, Y @ ah + phi.T @ gf - K @ sv
