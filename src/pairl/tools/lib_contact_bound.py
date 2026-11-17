"""Formulas of the contact force bound (note_contact): theorem (c), corollary with a speed limit and a delayed relief, and their inversion in the speed."""
import numpy as np

def fbar(v, a, mu, k, de):
    """Theorem (c): onset speed v [m/s], inward drive acceleration a [m/s^2], margin mu [1/kg], stiffness k [N/m], dissipation coefficient de [s/m]."""
    return (1 + de * v) * (a / mu + np.sqrt((a / mu) ** 2 + k * v ** 2 / mu))

def fbar_reaction(v, ar, mu, k, td, de):
    """Corollary: speed limit v, inward drive acceleration <= -ar from the delay td [s] after the onset."""
    c = mu * k; Vd = 0.5 * v ** 2 + 0.5 * c * (v * td) ** 2 + ar * v * td
    return (1 + de * v) * k * (-ar + np.sqrt(ar ** 2 + 2 * c * Vd)) / c

def speed_for(f, flim, hi=5.0):
    """Largest v in [0, hi] with f(v) <= flim for an increasing f; None if f(0) > flim."""
    if f(0.0) > flim: return None
    if f(hi) <= flim: return hi
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) <= flim else (lo, mid)
    return lo
