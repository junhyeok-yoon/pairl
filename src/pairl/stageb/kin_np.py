"""NumPy inverse kinematics and gauge margin of the planar 3R arm (no JAX)."""
import numpy as np
PG = 8

def ik_all(p, L, n=721):
    """All IK solutions on a grid of the last-link angle (two elbow branches). p: (...,2). Returns q (..., 2n, 3) and validity."""
    th3 = np.linspace(-np.pi, np.pi, n)
    W = p[..., None, :] - L[2] * np.stack([np.cos(th3), np.sin(th3)], -1)
    r2 = np.sum(W ** 2, -1); c2 = (r2 - L[0] ** 2 - L[1] ** 2) / (2 * L[0] * L[1]); ok = np.abs(c2) <= 1
    a = np.arccos(np.clip(c2, -1, 1)); out, val = [], []
    for sgn in (1.0, -1.0):
        q2 = sgn * a; q1 = np.arctan2(W[..., 1], W[..., 0]) - np.arctan2(L[1] * np.sin(q2), L[0] + L[1] * np.cos(q2))
        q3 = th3 - q1 - q2
        q = np.stack([q1, q2, q3], -1); q = (q + np.pi) % (2 * np.pi) - np.pi
        out.append(q); val.append(ok)
    return np.concatenate(out, -2), np.concatenate(val, -1)

def gauge(q, c, h):
    return np.sum(np.abs((q - c) / h) ** PG, -1) ** (1.0 / PG)

def best_margin(p, L, c, h):
    q, ok = ik_all(p, L); rho = np.where(ok, gauge(q, c, h), np.inf)
    i = np.argmin(rho, -1)
    return 1.0 - np.min(rho, -1), np.take_along_axis(q, i[..., None, None], -2)[..., 0, :]

