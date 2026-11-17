"""Design study for the Stage B constants: joint box, reachable workspace, gauge margin of targets, gravity load, directional authority (B3)."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, json, sys
from pairl.stageb import model as Mo, params
PG = 8

from pairl.stageb.kin_np import ik_all, gauge, best_margin

def workspace_stats(c, h, L=params.L_NOM, box=((0.35, 0.75), (-0.15, 0.35)), n=41, band=(0.08, 0.20), normal=0.25):
    X, Y = np.meshgrid(np.linspace(*box[0], n), np.linspace(*box[1], n)); P = np.stack([X, Y], -1)
    mg, qb = best_margin(P, L, c, h)
    mg_press = np.minimum(best_margin(P - np.array([0, 0.05]), L, c, h)[0], best_margin(P - np.array([0, 0.15]), L, c, h)[0])
    return dict(frac_normal=float(np.mean(mg >= normal)), frac_normal_with_press=float(np.mean((mg >= normal) & (mg_press >= band[0]))),
                frac_hard=float(np.mean((mg >= band[0]) & (mg < band[1]))), frac_unreachable=float(np.mean(mg < band[0])), margin_min=float(mg.min()), margin_max=float(mg.max())), mg

_ACC = jax.jit(jax.vmap(lambda q, dq, tau, xi, par, f: Mo.accel(q, dq, tau, xi, par, f)[0]))

def b3_samples(c, h, ub, k1, n, seed, ood, fbar=60.0, nu=0.8):
    """Sampled (B3) on rho(w) = 1, rho(w_q) <= 1: val = g^T ( qdd(tau = sigma_inf(-g), worst contact force in the cone) + k1 qd ).
    Contact cone: f_n in [0, fbar], |f_t| <= nu f_n, applied at the tip for every sampled posture (conservative)."""
    rng = np.random.default_rng(seed); c, h, ub = map(np.asarray, (c, h, ub))
    pars = [params.make_par(rng, ood=ood) for _ in range(n)]
    P = {k: jnp.asarray(np.stack([p[k] for p in pars])) for k in pars[0]}
    xi = (rng.random(n) < 0.5).astype(float)
    wq = rng.uniform(-1, 1, (4 * n, 3)); wq = wq[np.sum(np.abs(wq) ** PG, -1) <= 1][:n]
    w = rng.normal(size=(n, 3)); w = w / (np.sum(np.abs(w) ** PG, -1, keepdims=True) ** (1.0 / PG))
    q = c + h * wq; dq = k1 * h * (w - wq)
    grad = np.abs(w) ** (PG - 1) * np.sign(w); g = grad / h; g = g / np.linalg.norm(g, axis=-1, keepdims=True)
    tau = -g / np.max(np.abs(g) / ub, -1, keepdims=True)
    A = lambda f: np.asarray(_ACC(jnp.asarray(q), jnp.asarray(dq), jnp.asarray(tau), jnp.asarray(xi), P, jnp.asarray(np.tile(f, (n, 1)))))
    a0 = A(np.zeros(2)); val0 = np.sum(g * (a0 + k1 * dq), -1)
    cf = np.maximum(0.0, np.maximum(np.sum(g * (A(np.array([nu * fbar, fbar])) - a0), -1), np.sum(g * (A(np.array([-nu * fbar, fbar])) - a0), -1)))
    return val0, val0 + cf, dict(q=q, dq=dq, g=g, xi=xi)

def gravity_ratio(c, h, ub, n, seed, ood):
    rng = np.random.default_rng(seed); gf = jax.jit(jax.vmap(lambda q, par: Mo.gravity(q, par, 1.0)))
    pars = [params.make_par(rng, ood=ood, payload=(2.5 if ood else 1.5)) for _ in range(n)]
    P = {k: jnp.asarray(np.stack([p[k] for p in pars])) for k in pars[0]}
    wq = rng.uniform(-1, 1, (4 * n, 3)); wq = wq[np.sum(np.abs(wq) ** PG, -1) <= 1][:n]
    g = np.abs(np.asarray(gf(jnp.asarray(np.asarray(c) + np.asarray(h) * wq), P))) + 0.5
    return (g / np.asarray(ub)).max(0)

if __name__ == '__main__':
    cands = {'B': dict(c=(0.8, -1.25, -0.4), h=(1.0, 1.05, 1.3)), 'C': dict(c=(0.9, -1.3, -0.5), h=(0.9, 1.1, 1.2))}
    for name, cd in cands.items():
        for ub in ((80.0, 40.0, 15.0), (100.0, 50.0, 20.0), (120.0, 60.0, 25.0)):
            line = {}
            for k1 in (1.0, 1.5, 2.0):
                for ood in (False, True):
                    v0, v1, _ = b3_samples(cd['c'], cd['h'], ub, k1, 40000, 1, ood)
                    line[f'k1={k1} ood={int(ood)}'] = [round(float(-v0.max()), 1), round(float(-v1.max()), 1)]
            gr = [round(x, 2) for x in gravity_ratio(cd['c'], cd['h'], ub, 20000, 2, True)]
            print(name, ub, json.dumps(line), 'grav_ood', gr)
