import numpy as np, json, jax, jax.numpy as jnp
from jax import random, jit, vmap
import panda as P

# Authority of the saturated input for the 7-DOF arm with the output eta = dq + k1*q (relative degree one).
# Drift in torque units: tau_d = C dq + g + friction - k1*M*dq, so that d(eta)/dt = M^{-1}(tau - tau_d).
jax.config.update('jax_enable_x64', True)
TBAR = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
NS = 200

@jit
def model(q, dq, a, eta):
    M = P.mass_matrix(q, a) + jnp.diag(P.ARM)
    n = P.tau_r(a, q, dq, dq, jnp.zeros(P.N)) + P.tau_u(dq, eta)
    return M, n
vmodel = jit(vmap(model))

def scale_margin(A, b, iters=8, inner=400):
    # mu = min over the box boundary of v'Av / (-b'v): largest factor by which the drift can grow
    # before some error direction can no longer be restored (Dinkelbach, convex QP on each face by FISTA)
    S = A.shape[0]
    L = 2.0 * np.linalg.eigvalsh(A)[:, -1]
    fix = np.zeros((14, 7), bool); val = np.zeros((14, 7))
    for i in range(7):
        for k, sg in enumerate((1.0, -1.0)):
            fix[2 * i + k, i] = True; val[2 * i + k, i] = sg * TBAR[i]
    td = np.linalg.solve(A, b[..., None])[..., 0]                    # tau_d
    v0 = -td / np.max(np.abs(td) / TBAR, axis=1, keepdims=True)      # start: saturated input against the drift
    c = np.einsum('si,sij,sj->s', v0, A, v0) / np.maximum(-np.einsum('si,si->s', b, v0), 1e-12)
    for _ in range(iters):
        v = np.where(fix[None], val[None], 0.0) + np.zeros((S, 14, 7)); y = v.copy(); tk = 1.0
        for _ in range(inner):
            g = 2.0 * np.einsum('sij,sfj->sfi', A, y) + c[:, None, None] * b[:, None, :]
            vn = np.clip(y - g / L[:, None, None], -TBAR, TBAR)
            vn = np.where(fix[None], val[None], vn)
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y = vn + ((tk - 1.0) / tn) * (vn - v); v, tk = vn, tn
        obj = np.einsum('sfi,sij,sfj->sf', v, A, v) + c[:, None] * np.einsum('si,sfi->sf', b, v)
        j = np.argmin(obj, axis=1); vb = v[np.arange(S), j]
        num = np.einsum('si,sij,sj->s', vb, A, vb); den = -np.einsum('si,si->s', b, vb)
        c = np.where(den > 1e-12, num / np.maximum(den, 1e-12), c)
    return c, v

key = random.PRNGKey(0)
out = []
for k1, vbar in [(0.0, 0.0), (2.0, 0.5), (5.0, 0.5), (2.0, 1.0), (5.0, 1.0)]:
    key, ka, kb, kc = random.split(key, 4)
    a, eta, _ = P.sample_tasks(ka, NS)
    q = P.Q0[None] + random.uniform(kb, (NS, P.N), minval=-0.5, maxval=0.5)
    dq = vbar * random.uniform(kc, (NS, P.N), minval=-1.0, maxval=1.0)
    M, n = [np.asarray(x) for x in vmodel(q, dq, a, eta)]
    dqn = np.asarray(dq)
    td = n - k1 * np.einsum('sij,sj->si', M, dqn)
    A = np.linalg.inv(M); b = np.einsum('sij,sj->si', A, td)
    ev = np.linalg.eigvalsh(M)
    rho0 = TBAR.min() / ev[:, -1] - np.linalg.norm(b, axis=1)                      # conservative class-level form
    nrm = np.sqrt(np.einsum('si,si->s', td, b))                                     # ||tau_d|| in the M^{-1} norm
    theta = np.min(TBAR[None] / np.sqrt(np.einsum('sii->si', M)), axis=1) - nrm     # closed-form sufficient condition
    mu, vmin = scale_margin(A, b)
    # check of the certificate: the minimum over sampled boundary points must not be below mu
    rng = np.random.default_rng(1); worst = np.inf
    for _ in range(40):
        v = rng.uniform(-1, 1, (NS, 7)) * TBAR; i = rng.integers(0, 7, NS); v[np.arange(NS), i] = np.sign(v[np.arange(NS), i]) * TBAR[i]
        den = -np.einsum('si,si->s', b, v); r = np.einsum('si,sij,sj->s', v, A, v) / np.where(den > 0, den, np.nan)
        worst = min(worst, float(np.nanmin(r - mu)))
    # rate margin rho_dir = min over unit w of w'A(w/|w|_U + tau_d); candidates: face minimizers and random boundary points
    cand = np.concatenate([vmin, rng.uniform(-1, 1, (NS, 400, 7)) * TBAR], 1)
    ii = rng.integers(0, 7, (NS, 400)); sl = cand[:, 14:]
    np.put_along_axis(sl, ii[..., None], np.sign(np.take_along_axis(sl, ii[..., None], 2) + 1e-300) * TBAR[ii][..., None], 2)
    qv = np.einsum('sfi,sij,sfj->sf', cand, A, cand) + np.einsum('si,sfi->sf', b, cand)
    rho_ub = (qv / np.linalg.norm(cand, axis=2)).min(1)                      # estimate (sampled upper bound of the minimum)
    rho_lb = (np.einsum('sfi,sij,sfj->sf', vmin, A, vmin) * (1 - 1 / mu[:, None])).min(1) / np.linalg.norm(TBAR)   # certified
    # worst sampled rate w'(F + Gamma*sigma(u_ff - s w)) at finite feedback magnitude s, adversarial bounded feedforward
    UFF = 50.0; what = cand / np.linalg.norm(cand, axis=2, keepdims=True)
    Aw = np.einsum('sij,sfj->sfi', A, what); rates = {}
    ffs = [np.zeros_like(what), UFF * Aw / np.linalg.norm(Aw, axis=2, keepdims=True), UFF * what]
    for _ in range(6):
        d = rng.normal(size=what.shape); ffs.append(UFF * d / np.linalg.norm(d, axis=2, keepdims=True))
    for sfb in (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
        worst_rate = -np.inf
        for uff in ffs:
            vv = uff - sfb * what
            u = vv / np.maximum(1.0, np.max(np.abs(vv) / TBAR, axis=2, keepdims=True))
            r = np.einsum('sfi,sfi->sf', Aw, u) - np.einsum('sfi,si->sf', what, b)
            worst_rate = max(worst_rate, float(r.max()))
        rates[str(int(sfb))] = worst_rate
    res = dict(k1=k1, vbar=vbar, n=NS, rho_dir_certified_min=float(rho_lb.min()), rho_dir_sampled_min=float(rho_ub.min()), worst_rate_at_s=rates, lam_min=float(ev[:, 0].min()), lam_max=float(ev[:, -1].max()),
               rho0_pos=float(np.mean(rho0 > 0)), rho0_med=float(np.median(rho0)),
               theta_pos=float(np.mean(theta > 0)), mu_pos=float(np.mean(mu > 1)), mu_min=float(mu.min()),
               mu_q05=float(np.quantile(mu, 0.05)), mu_med=float(np.median(mu)), sampled_minus_mu_min=worst,
               tau_d_absmax=[float(x) for x in np.abs(td).max(0)])
    out.append(res)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'tau_d_absmax'}),
          '| max |tau_d| per joint:', np.round(np.abs(td).max(0), 1), flush=True)
json.dump(out, open('../records/authority_panda.json', 'w'), indent=1)
