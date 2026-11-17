import itertools
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import ph2 as P

J = lambda d: jax.tree_util.tree_map(jnp.asarray, d)
cl = pickle.load(open("out2_classical.pkl", "rb"))
ctrl = {"MRAC": ("cl", P.mrac_op, {"gamma": float(cl["mrac"][1]), "k": 0.0, "ke": 0.0}),
        "CMP": ("cl", P.mrac_op, {"gamma": float(cl["comp"][1]), "k": float(cl["comp"][2]), "ke": float(cl["comp"][3])}),
        "PH-lin": ("ph", P.ph_op, J(pickle.load(open("out2_phlin.pkl", "rb"))["p"])),
        "PH": ("ph", P.ph_op, J(pickle.load(open("out2_ph.pkl", "rb"))["p"]))}
DT, T = 0.002, 20.0
NS = int(T / DT)


def rates(kind, p, t, s, task):
    # dissipation rates of the ledger, recomputed from the closed-loop state (noise-free)
    theta, lam, rp = task
    x, xr, ef, uf, Phf, z = s[0], s[1], s[2], s[3], s[4:P.OFF], s[P.OFF:]
    r = P.ref_sig(t, rp)
    y = x - xr
    Ph = P.regressor(x, r)
    ell = (y - ef) / P.TAU + P.A_M * ef
    Wf = Phf.at[P.NA - 1].add(ell)
    tgt = uf - ell
    if kind == "ph":
        g = P.grad_H(p, z)
        phi, R, Jm, kap, ke = P.ph_terms(p, x, xr, r, y)
        eps = Wf @ g[:P.NA] - tgt
        return jnp.array([P.A_M * y * y, lam * kap * y * y, lam * ke / (1.0 + Wf @ Wf) * eps * eps, lam * g[P.NA:] @ R @ g[P.NA:]])
    th = z[:P.NA]
    eps = Wf @ th - tgt
    return jnp.array([P.A_M * y * y, lam * p["k"] * y * y, lam * p["ke"] / (1.0 + Wf @ Wf) * eps * eps, 0.0])


hessH = jax.hessian(P.H_fn, argnums=1)


def z_min(p, c):
    cf = jnp.concatenate([c, jnp.zeros(P.NZ - P.NA)])
    body = lambda i, z: z - jnp.linalg.solve(hessH(p, z), P.grad_H(p, z) - cf)
    return jax.lax.fori_loop(0, 60, body, jnp.zeros(P.NZ)), cf


def Vfun(kind, p, s, task, level=False):
    # energy up to the constant min term (enough for differences); level=True subtracts the minimum
    theta, lam, rp = task
    c = P.c_star(theta, lam)
    z = s[P.OFF:]
    e = s[0] - s[1]
    if kind == "ph":
        base = P.H_fn(p, z) - c @ z[:P.NA]
        if level:
            zs, cf = z_min(p, c)
            base = base - (P.H_fn(p, zs) - cf @ zs)
        return 0.5 * e * e + lam * base
    return 0.5 * e * e + lam * jnp.sum((z[:P.NA] - c) ** 2) / (2.0 * p["gamma"])


tasks, x0 = P.sample_tasks(random.PRNGKey(5), 32)
VERT = jnp.array(list(itertools.product([-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [0.0, 1.5])))
print("law      V(0)    =  int a_m e^2 + damping + pred.err + feature R + V(T)   closure | share via e | a-priori tube sqrt(2B0) vs max|e|")
for name, (kind, op, p) in ctrl.items():
    rhs = P.make_rhs(op)

    def run(theta, lam, rp, x00):
        task = (theta, lam, rp)

        def step(c, i):
            s, acc = c
            t = i * DT
            k1 = rhs(p, t, s, task, 0.0)[0]; k2 = rhs(p, t + DT / 2, s + DT / 2 * k1, task, 0.0)[0]
            k3 = rhs(p, t + DT / 2, s + DT / 2 * k2, task, 0.0)[0]; k4 = rhs(p, t + DT, s + DT * k3, task, 0.0)[0]
            sn = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            acc = acc + DT * 0.5 * (rates(kind, p, t, s, task) + rates(kind, p, t + DT, sn, task))
            return (sn, acc), jnp.abs(s[0] - s[1])
        s0 = jnp.concatenate([jnp.array([x00, x00]), jnp.zeros(P.OFF - 2 + P.NZ)])
        (sT, acc), emax = jax.lax.scan(step, (s0, jnp.zeros(4)), jnp.arange(NS))
        return Vfun(kind, p, s0, task, True), Vfun(kind, p, sT, task, True), acc, emax.max()

    V0, VT, acc, emax = map(np.asarray, jit(vmap(run))(tasks[0], tasks[1], tasks[2], x0))
    s0 = jnp.zeros(P.OFF + P.NZ)
    B0 = float(jnp.max(vmap(lambda v: Vfun(kind, p, s0, (-v[:3], 1.0 / (1.0 + v[3]), tasks[2][0]), True) * (1.0 + v[3]))(VERT)))   # lambda_max = 1
    m = lambda a: float(np.mean(a))
    tot = acc.sum(1)
    print(f"{name:7s} {m(V0):7.4f} = {m(acc[:,0]):7.4f} + {m(acc[:,1]):7.4f} + {m(acc[:,2]):7.4f} + {m(acc[:,3]):7.4f} + {m(VT):7.4f}   {m(np.abs(V0 - VT - tot)):.1e} | "
          f"{100*m(acc[:,0])/max(m(tot),1e-12):5.1f}% | {np.sqrt(2*B0):.2f} vs {emax.max():.3f} (median {np.median(emax):.3f})", flush=True)
