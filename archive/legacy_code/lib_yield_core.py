import numpy as np

# Class data, design constants and closed-form pieces shared by the checks and the figure.

def gauge(v, ubar):
    # Minkowski gauge of the box U = {|u_i| <= ubar_i}
    return np.max(np.abs(v) / ubar, axis=-1)

def sat_dir(v, ubar):
    # direction-preserving saturation onto U
    lam = 1.0 / np.maximum(1.0, gauge(v, ubar))
    return lam[..., None] * v, lam

def sat_clip(v, ubar):
    # componentwise clipping (comparison only)
    return np.clip(v, -ubar, ubar)

def ramp(m, m0):
    # yield function: continuous, nondecreasing, zero at zero
    return np.clip(m / m0, 0.0, 1.0)

def Psi(s, g0, g1, abar, umin):
    # lower bound of the restoring rate -w^T Gamma sigma(a - s w), valid for s >= g1*abar/g0
    return np.minimum(1.0, umin / (abar + s)) * (g0 * s - g1 * abar)

def s_low(m, kap0, psi_max):
    # smallest feedback magnitude kappa*||xi|| at normalized error 1 - m
    return kap0 * (1.0 - m) / (psi_max * m * (2.0 - m))

def margin_star(P):
    # largest m such that the a priori decrease condition holds on (0, m]
    g0, g1, umin, Fbar, abar = P['g0'], P['g1'], P['umin'], P['Fbar'], P['abar']
    rho0 = g0 * umin - Fbar
    assert rho0 > 0, 'authority condition violated'
    dpsi = P['psi_max'] - P['psi_low']

    def ok(m):
        s = s_low(m, P['kap0'], P['psi_max'])
        if s < g1 * abar / g0:
            return False
        lhs = ramp(m, P['m_r']) * P['vbar'] + P['alpha'] * ramp(m, P['m_psi']) * dpsi \
            + g0 * umin - Psi(s, g0, g1, abar, umin)
        return lhs < rho0

    lo, hi = 0.0, 1.0 - 1e-9
    if ok(hi):
        return hi, rho0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo, rho0
