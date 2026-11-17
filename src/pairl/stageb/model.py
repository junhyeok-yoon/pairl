"""Stage B plant: planar 3R arm in a vertical plane, attached object, block contact at the tip.
Closed-form dynamics in absolute link angles theta = S q; all functions are pure JAX and act on one state.
The attached object is a point mass rigidly fixed to the last link at the tip-relative offset (rx, ry) in the last-link frame."""
import jax, jax.numpy as jnp

G0 = 9.81
C_V = 400.0                                             # viscous slope of the regularized friction at zero sliding speed [N s/m]
S = jnp.array([[1., 0, 0], [1, 1, 0], [1, 1, 1]])       # theta = S q
SINV = jnp.array([[1., 0, 0], [-1, 1, 0], [0, -1, 1]])  # q = SINV theta

def link_consts(par):
    """Constant matrices of the links in the absolute-angle form."""
    L, m, lc = par['L'], par['m'], par['lc']
    z = jnp.zeros(())
    c1 = jnp.stack([lc[0], z, z]); c2 = jnp.stack([L[0], lc[1], z]); c3 = jnp.stack([L[0], L[1], lc[2]])
    A = m[0] * jnp.outer(c1, c1) + m[1] * jnp.outer(c2, c2) + m[2] * jnp.outer(c3, c3)
    G = m[0] * c1 + m[1] * c2 + m[2] * c3
    return A, G

def obj_lever(par):
    """Attached object: p_obj = sum_k a_k e(theta_k + b_k), e(x) = (cos x, sin x)."""
    L = par['L']; ux = L[2] + par['rx']; uy = par['ry']
    a = jnp.stack([L[0], L[1], jnp.sqrt(ux ** 2 + uy ** 2)])
    b = jnp.stack([jnp.zeros(()), jnp.zeros(()), jnp.arctan2(uy, ux)])
    return a, b

def abs_terms(th, om, par, xi):
    """Mass matrix, velocity term and gravity in absolute angles (object included when xi = 1)."""
    A, G = link_consts(par); a, b = obj_lever(par); mo = xi * par['mo']; ph = th + b
    dth = th[:, None] - th[None, :]; dph = ph[:, None] - ph[None, :]; Ao = mo * jnp.outer(a, a)
    M = A * jnp.cos(dth) + Ao * jnp.cos(dph) + jnp.diag(par['Icm']) + par['Ir'] * (SINV.T @ SINV)
    h = (A * jnp.sin(dth)) @ om ** 2 + (Ao * jnp.sin(dph)) @ om ** 2
    g = G0 * (G * jnp.cos(th) + mo * a * jnp.cos(ph))
    return M, h, g

def tip(q, L):
    th = jnp.cumsum(q)
    return jnp.stack([jnp.sum(L * jnp.cos(th)), jnp.sum(L * jnp.sin(th))])

def obj_pos(q, par):
    """Center of mass of the object if it is attached at the offset (rx, ry)."""
    a, b = obj_lever(par); ph = jnp.cumsum(q) + b
    return jnp.stack([jnp.sum(a * jnp.cos(ph)), jnp.sum(a * jnp.sin(ph))])

def offset_of(q, c, L):
    """Tip-relative offset in the last-link frame of a point c: R(theta_3)^T (c - p(q))."""
    t3 = jnp.sum(q); v = c - tip(q, L)
    return jnp.stack([jnp.cos(t3) * v[0] + jnp.sin(t3) * v[1], -jnp.sin(t3) * v[0] + jnp.cos(t3) * v[1]])

def jac_abs(th, L):
    return jnp.stack([-L * jnp.sin(th), L * jnp.cos(th)])               # 2x3, columns d p / d theta_k

def jac(q, L):
    return jac_abs(jnp.cumsum(q), L) @ S                                # d p / d q

def mass_abs(th, par, xi):
    return abs_terms(th, jnp.zeros_like(th), par, xi)[0]

def mass(q, par, xi):
    return S.T @ mass_abs(jnp.cumsum(q), par, xi) @ S

def gravity(q, par, xi):
    th = jnp.cumsum(q)
    return S.T @ abs_terms(th, jnp.zeros_like(th), par, xi)[2]           # g(q) in joint coordinates

def energy(q, dq, par, xi):
    """Kinetic and potential energy of arm (+ attached object); E_rob = KE + PE - d^T q."""
    th, om = jnp.cumsum(q), jnp.cumsum(dq)
    _, G = link_consts(par); a, b = obj_lever(par)
    ke = 0.5 * om @ mass_abs(th, par, xi) @ om
    pe = G0 * (jnp.sum(G * jnp.sin(th)) + xi * par['mo'] * jnp.sum(a * jnp.sin(th + b)))
    return ke, pe

def contact(p, v, par):
    """Force of the block on the tip (2,) and stored elastic energy. Block: |x - xe| < we, y < ye.
    Hunt-Crossley law on the nearest face: f = ke d max(0, 1 + de d_dot), d: penetration depth; Coulomb friction on the top face,
    regularized with a slope C_V at zero sliding speed that does not grow with the normal force."""
    dx = p[0] - par['xe']; sgn = jnp.where(dx >= 0, 1.0, -1.0)
    d_top = par['ye'] - p[1]; d_side = par['we'] - jnp.abs(dx)
    inside = (d_top > 0) & (d_side > 0); top = d_top <= d_side
    fn = par['ke'] * d_top * jnp.maximum(1.0 - par['de'] * v[1], 0.0)                 # d_dot = -v_y
    fc = par['nu'] * fn                                                               # Coulomb limit
    f_top = jnp.stack([-fc * jnp.tanh(C_V * v[0] / (fc + 1e-9)), fn])
    fs = par['ke'] * d_side * jnp.maximum(1.0 - par['de'] * sgn * v[0], 0.0)          # d_dot = -sgn v_x
    f_side = jnp.stack([sgn * fs, jnp.zeros(())])
    f = jnp.where(inside, jnp.where(top, f_top, f_side), jnp.zeros(2))
    s_env = jnp.where(inside, 0.5 * par['ke'] * jnp.minimum(d_top, d_side) ** 2, 0.0)
    return f, s_env

def accel(q, dq, tau, xi, par, f_extra=None):
    """Joint acceleration and contact force. tau: applied joint torque; f_extra: additional tip force (tests)."""
    th, om = jnp.cumsum(q), jnp.cumsum(dq)
    Mabs, habs, gabs = abs_terms(th, om, par, xi)
    Ja = jac_abs(th, par['L'])
    p = jnp.stack([jnp.sum(par['L'] * jnp.cos(th)), jnp.sum(par['L'] * jnp.sin(th))])
    fe, _ = contact(p, Ja @ om, par)
    ftot = fe if f_extra is None else fe + f_extra
    Q = SINV.T @ (tau + par['d'] - par['Fv'] * dq) + Ja.T @ ftot
    return SINV @ jnp.linalg.solve(Mabs, Q - habs - gabs), fe

def attach_velocity(q, dq, par):
    """Joint velocity after a rigid attachment of the object at rest: M_plus dq_plus = M_minus dq_minus."""
    return jnp.linalg.solve(mass(q, par, 1.0), mass(q, par, 0.0) @ dq)
