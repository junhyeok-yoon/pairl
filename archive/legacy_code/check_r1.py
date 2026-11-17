import numpy as np, json

# R1 check: box joint constraint under an adversarial hand torque (pushing toward the nearest corner, bounded),
# direction-preserving saturation and two constraint layers: componentwise barrier vs scalar p-norm gauge barrier.
K1, PG, KB, RON, RFULL = 2.0, 8, 20.0, 0.8, 0.9
LO, HI = np.array([-0.2, 0.2]), np.array([1.2, 1.8])
CEN, HALF = (LO + HI) / 2, (HI - LO) / 2

def sat(v, ub):
    lam = 1.0 / np.maximum(1.0, np.max(np.abs(v) / ub, axis=-1, keepdims=True))
    return lam * v

def arm(q, dq, pay):
    # planar two-link arm in a vertical plane (masses 4, 2 kg, lengths 0.4 m, payload at the tip, rotor inertia)
    m1, m2, l1, lc1, lc2 = 4.0, 2.0 + pay, 0.4, 0.2, (2.0 * 0.2 + pay * 0.4) / (2.0 + pay)
    I1, I2 = 4.0 * 0.4 ** 2 / 12, 2.0 * 0.4 ** 2 / 12 + pay * 0.0
    a1 = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2); a2 = m2 * l1 * lc2; a3 = I2 + m2 * lc2 ** 2
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    M = np.stack([np.stack([a1 + 2 * a2 * c2 + 0.3, a3 + a2 * c2], -1), np.stack([a3 + a2 * c2, a3 + 0.1 + 0 * c2], -1)], -2)
    h = a2 * s2
    cor = np.stack([-h * dq[:, 1] * (2 * dq[:, 0] + dq[:, 1]), h * dq[:, 0] ** 2], -1)
    gv = 9.81 * np.stack([(m1 * lc1 + m2 * l1) * np.cos(q[:, 0]) + m2 * lc2 * np.cos(q[:, 0] + q[:, 1]), m2 * lc2 * np.cos(q[:, 0] + q[:, 1])], -1)
    return M, cor + gv + 0.5 * dq

def coupled(q, dq, pay):
    # constant strongly coupled inertia (M^{-1} = 5 [[1, 0.9], [0.9, 1]]) with a small drift
    Minv = 5.0 * np.array([[1.0, 0.9], [0.9, 1.0]])
    M = np.broadcast_to(np.linalg.inv(Minv), (q.shape[0], 2, 2))
    return M, 0.2 * dq

def barrier(eta, kind):
    w = (eta - K1 * CEN) / (K1 * HALF)
    if kind == 'componentwise':
        a = np.clip((np.abs(w) - RON) / (RFULL - RON), 0, 1)
        return -a * KB * w / np.maximum(1 - w ** 2, 1e-9), np.max(np.abs(w), -1)
    rho = np.sum(np.abs(w) ** PG, -1) ** (1.0 / PG)
    grad = np.abs(w) ** (PG - 1) * np.sign(w) / np.maximum(rho, 1e-12)[:, None] ** (PG - 1)
    g = grad / (K1 * HALF)
    gh = g / np.linalg.norm(g, axis=-1, keepdims=True)
    a = np.clip((rho - RON) / (RFULL - RON), 0, 1)
    s = a * KB * rho / np.maximum(1 - rho ** 2, 1e-9)
    return -s[:, None] * gh, rho

def run(plant, kind, ub, push, n=100, dt=1e-4, tend=2.0, seed=0):
    r = np.random.default_rng(seed)
    q = r.uniform(LO + 0.2 * HALF, HI - 0.2 * HALF, (n, 2))
    w0 = r.normal(size=(n, 2)); w0 = w0 / (np.sum(np.abs(w0) ** PG, -1, keepdims=True) ** (1.0 / PG)) * r.uniform(0.3, 0.95, (n, 1))
    dq = K1 * CEN + w0 * K1 * HALF - K1 * q                    # eta(0) inside the gauge set (hence inside the box set)
    pay = r.uniform(0, 2, (n, 1))[:, 0] if plant is arm else None
    worst, qviol = 0.0, 0
    def f(q, dq):
        eta = dq + K1 * q
        w = (eta - K1 * CEN) / (K1 * HALF)
        tau_h = push * ub * np.sign(w)                               # adversarial hand torque toward the nearest corner
        tb, meas = barrier(eta, kind)
        tau = sat(tau_h + tb, ub)
        M, drift = plant(q, dq, pay)
        ddq = np.linalg.solve(M, (tau - drift)[..., None])[..., 0]
        return dq, ddq, meas
    for _ in range(int(tend / dt)):
        k1q, k1d, m1 = f(q, dq); k2q, k2d, _ = f(q + 0.5 * dt * k1q, dq + 0.5 * dt * k1d)
        k3q, k3d, _ = f(q + 0.5 * dt * k2q, dq + 0.5 * dt * k2d); k4q, k4d, _ = f(q + dt * k3q, dq + dt * k3d)
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q); dq = dq + dt / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        worst = max(worst, float(np.nanmax(m1)))
        qviol += int(np.sum(np.any((q < LO) | (q > HI), -1)))
    return dict(worst_measure=worst, joint_violation_samples=qviol)

def authority_margin(plant, ub, n=20000, seed=1):
    # sampled directional authority of the gauge layer: -( g^T M^{-1} sigma_inf(-g) + g^T phi ) on the gauge boundary
    r = np.random.default_rng(seed)
    w = r.normal(size=(n, 2)); w = w / (np.sum(np.abs(w) ** PG, -1, keepdims=True) ** (1.0 / PG))
    eta = K1 * CEN + w * K1 * HALF
    q = r.uniform(LO, HI, (n, 2)); dq = eta - K1 * q
    pay = r.uniform(0, 2, n) if plant is arm else None
    grad = np.abs(w) ** (PG - 1) * np.sign(w); g = grad / (K1 * HALF); gh = g / np.linalg.norm(g, axis=-1, keepdims=True)
    sig = -gh / np.max(np.abs(gh) / ub, -1, keepdims=True)
    M, drift = plant(q, dq, pay)
    Minv = np.linalg.inv(M)
    phi = -(Minv @ drift[..., None])[..., 0] + K1 * dq
    val = np.einsum('ni,nij,nj->n', gh, Minv, sig) + np.einsum('ni,ni->n', gh, phi)
    return float(-val.max()), float(np.mean(val < 0))

res = {}
for name, plant, ub in [('two_link_arm', arm, np.array([60.0, 25.0])), ('strong_coupling', coupled, np.array([2.0, 8.0])), ('weak_two_link', arm, np.array([15.0, 6.0]))]:
    res[name] = dict(authority_min_margin=None)
    mm, frac = authority_margin(plant, ub)
    res[name]['authority_min_margin'], res[name]['authority_fraction_positive'] = mm, frac
    for kind in ('componentwise', 'gauge'):
        for push in ((0.6,) if name == 'weak_two_link' else (0.3, 0.6)):
            out = run(plant, kind, ub, push)
            res[name][f'{kind}_push{push}'] = out
            print(name, kind, 'push', push, out)
    print(name, 'sampled authority margin of the gauge layer: min %.4f, fraction positive %.4f' % (mm, frac))
json.dump(res, open('../records/r1_check.json', 'w'), indent=1)
