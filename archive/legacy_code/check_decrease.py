import numpy as np
from lib_yield_core import *

# Point evaluation of the decrease inequality: for every sampled class member and controller signal with
# normalized error in [1 - m_star, 1), the exact derivative of ||w|| must be negative.
rng = np.random.default_rng(0)
N = 2_000_000

def sample_gamma(n, m, g0, g1):
    # symmetric part with eigenvalues in [g0, lam_hi], plus a skew part; spectral norm <= g1
    A = rng.normal(size=(n, m, m))
    Q, _ = np.linalg.qr(A)
    skew_max = 0.15 * g1
    lam = rng.uniform(g0, g1 - skew_max, size=(n, m))
    lam[:, 0] = g0  # weakest admissible direction always present
    S = np.einsum('nij,nj,nkj->nik', Q, lam, Q)
    B = rng.normal(size=(n, m, m)); K = B - np.swapaxes(B, 1, 2)
    K *= (rng.uniform(0, skew_max, n) / np.maximum(np.linalg.norm(K, 2, axis=(1, 2)), 1e-12))[:, None, None]
    return S + K

def unit(n, m):
    v = rng.normal(size=(n, m)); return v / np.linalg.norm(v, axis=1, keepdims=True)

worst = {}
for m_dim, ubar in [(1, np.array([2.0])), (2, np.array([2.0, 3.0])), (3, np.array([2.0, 3.0, 8.0]))]:
    for g1 in [2.0, 9.5]:
        P = dict(g0=0.5, g1=g1, umin=ubar.min(), Fbar=0.6, abar=1.0, kap0=1.0, alpha=2.0,
                 psi_low=0.05, psi_max=0.4, vbar=3.0, m_r=0.5, m_psi=0.5)
        ms, rho0 = margin_star(P)
        Gam = sample_gamma(N, m_dim, P['g0'], g1)
        what = unit(N, m_dim)
        # error direction aligned with the weakest direction in half of the samples
        ev = np.linalg.eigh(0.5 * (Gam + np.swapaxes(Gam, 1, 2)))[1][:, :, 0]
        half = rng.random(N) < 0.5
        what[half] = ev[half]
        mm = ms * rng.random(N) ** 2                         # margin in (0, m_star], dense near 0
        mm = np.maximum(mm, 1e-9)
        mm[: N // 10] = ms                                   # boundary of the claimed region
        wn = 1.0 - mm
        psi = rng.uniform(P['psi_low'], P['psi_max'], N)
        psi[N // 10: N // 5] = P['psi_max']
        kap = P['kap0'] * 10 ** rng.uniform(0, 1, N); kap[: N // 2] = P['kap0']
        # adversarial drift, feedforward and proposal: all push the error outward at full size
        F = P['Fbar'] * what
        a_dir = unit(N, m_dim); a_dir[: N // 2] = what[: N // 2]
        a = P['abar'] * a_dir
        vp = -P['vbar'] * what                                # reference runs away from the output
        psi_p = np.full(N, P['psi_low'])                      # fastest admissible contraction
        k = 1.0 / (1.0 - wn ** 2)
        xi = (k * wn / psi)[:, None] * what
        v = a - kap[:, None] * xi
        u, lam = sat_dir(v, ubar)
        gr = ramp(mm, P['m_r']); gp = ramp(mm, P['m_psi'])
        de = np.einsum('ni,ni->n', what, F + np.einsum('nij,nj->ni', Gam, u) - gr[:, None] * vp)
        dpsi = -P['alpha'] * gp * np.maximum(psi - psi_p, 0.0)
        dw = (de - wn * dpsi) / psi                           # exact d||w||/dt
        s = kap * k * wn / psi
        restoring = -np.einsum('ni,nij,nj->n', what, Gam, u)
        gap = restoring - Psi(s, P['g0'], g1, P['abar'], P['umin'])
        print(f"m={m_dim} gamma1={g1}: rho0={rho0:.3f} m_star={ms:.5f} | max d||w||/dt * psi = {np.max(dw * psi):+.4f} "
              f"(must be < 0) | min(restoring - Psi(s)) = {gap.min():+.3e} (must be >= 0) | samples {N}", flush=True)
        worst[(m_dim, g1)] = (ms, np.max(dw * psi), gap.min())

# Componentwise clipping: a class member for which the saturated feedback pushes the error outward.
Gam = 5.0 * np.array([[1.0, 0.9], [0.9, 1.0]]); ubar = np.array([2.0, 8.0])
w = np.array([1.0, -0.3]); w /= np.linalg.norm(w)
print("eig(Gamma) =", np.round(np.linalg.eigvalsh(Gam), 3), " gamma0*umin =", 0.5 * ubar.min())
print("clipping:             w^T Gamma clip(-s w), s->inf =", round(float(w @ Gam @ (-np.sign(w) * ubar)), 4))
print("direction-preserving: w^T Gamma sigma(-s w), s->inf =", round(float(w @ Gam @ (-w / gauge(w, ubar))), 4))
