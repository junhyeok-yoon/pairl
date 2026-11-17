"""v3 controller for Stage B (minimal configuration, no learning): hand (impedance + rest-gated memory), full-power tank with gate,
contact guard, gauge constraint layer, direction-preserving saturation, ledger. Uses measured signals only. Pure JAX, one episode."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, params as P

G0 = Mo.G0
SROW = np.array([[1., 0, 0], [1, 1, 0], [1, 1, 1]])          # rows: S^T e_k (joint-torque direction of an absolute-angle torque on link k)

def default_cfg():
    """Controller constants fixed in L0 (spec rev. 12): normalized memory law, tank-level gate."""
    return dict(Lq=20.0, LK=10.0, klo=np.array([50.0, 25.0, 10.0]), khi=np.array([400.0, 200.0, 80.0]), D=np.array([30.0, 12.0, 2.0]),
                tau_th=0.5, tau_max=25.0, v_r=0.3, n_r=int(round(0.1 / P.TS)), lam_s=5.0, law=2.0, Kn=np.array([200.0, 100.0, 40.0]),
                zbar=50.0, Tmin=1.0, dT=1.0, gate_sign=0.0, pm=0.0, k1=2.0, kb=2.0, ron=0.8, rfull=0.9, feats=0.0)

# memory features
def feats_class(q):
    """Class-exact basis (7 x 3) for a planar 3R arm with a point-mass load and constant joint disturbance."""
    th = jnp.cumsum(q); S = jnp.asarray(SROW, q.dtype)
    rows = [S[0] * jnp.cos(th[0]), S[1] * jnp.cos(th[1]), S[2] * jnp.cos(th[2]), S[2] * jnp.sin(th[2]), jnp.eye(3, dtype=q.dtype)[0], jnp.eye(3, dtype=q.dtype)[1], jnp.eye(3, dtype=q.dtype)[2]]
    return jnp.stack(rows)

def feats_obj(q, L):
    """Oracle object basis (3 x 3, uses the kinematics; for the generalization test only): u_obj = m f1 + (m rx) f2 + (m ry) f3."""
    th = jnp.cumsum(q); S = jnp.asarray(SROW, q.dtype)
    jy = Mo.jac(q, L)[1]
    return jnp.stack([G0 * jy, G0 * jnp.cos(th[2]) * S[2], -G0 * jnp.sin(th[2]) * S[2]])

def prior_body(q, Lnom, mnom):
    """Gravity torque of the nominal body without load (the prior of the memory)."""
    par = dict(L=Lnom, m=mnom, lc=Lnom / 2, Icm=mnom * Lnom ** 2 / 12, Ir=jnp.asarray(P.IR), mo=jnp.asarray(0.0), rx=jnp.asarray(0.0), ry=jnp.asarray(0.0))
    return Mo.gravity(q, par, 0.0)

def memory_out(q, z, cfg, Lnom, mnom):
    """u_ad = prior body gravity + G(q)^T z, and the feature matrix G (rows = features)."""
    Gc = feats_class(q); Go = feats_obj(q, Lnom)
    nz = Gc.shape[0]
    Gpad = jnp.concatenate([Go, jnp.zeros((nz - 3, 3), q.dtype)])                      # pad the oracle basis to the same size
    G = jnp.where(cfg['feats'] > 0.5, Gpad, Gc)
    return prior_body(q, Lnom, mnom) + G.T @ z, G

# constraint layer
def gauge_torque(qm, dqm, cfg):
    qc, qh = jnp.asarray(P.QC, qm.dtype), jnp.asarray(P.QH, qm.dtype)
    eta = dqm + cfg['k1'] * qm
    w = (eta - cfg['k1'] * qc) / (cfg['k1'] * qh)
    rho = jnp.sum(jnp.abs(w) ** P.PG) ** (1.0 / P.PG)
    grad = jnp.abs(w) ** (P.PG - 1) * jnp.sign(w) / jnp.maximum(rho, 1e-12) ** (P.PG - 1)
    a = jnp.clip((rho - cfg['ron']) / (cfg['rfull'] - cfg['ron']), 0.0, 1.0)
    st = a * cfg['kb'] * rho / jnp.maximum(1.0 - rho ** 2, 1e-6)
    return -st * grad / qh, rho

def barrier_potential(q, cfg, n=200):
    """U_b(q) = Psi(rho(w_q)) by quadrature of s~ (for diagnostics)."""
    wq = (q - jnp.asarray(P.QC, q.dtype)) / jnp.asarray(P.QH, q.dtype)
    r = jnp.minimum(jnp.sum(jnp.abs(wq) ** P.PG) ** (1.0 / P.PG), 1.0 - 1e-6)
    x = jnp.linspace(0.0, 1.0, n + 1) * r
    a = jnp.clip((x - cfg['ron']) / (cfg['rfull'] - cfg['ron']), 0.0, 1.0)
    f = a * cfg['kb'] * x / (1.0 - x ** 2)
    return jnp.sum(0.5 * (f[1:] + f[:-1]) * (x[1:] - x[:-1]))

# controller state and step
def init_ctrl(q0, E0, cfg, nz=7):
    z = jnp.zeros(())
    return dict(qd=q0, K=jnp.asarray(cfg['Kn'], q0.dtype), z=jnp.zeros(nz, q0.dtype), T=jnp.asarray(cfg['Tmin'], q0.dtype) + E0, nrest=jnp.zeros((), jnp.int32),
                tau_prev=jnp.zeros(3, q0.dtype), q_prev=q0, lam_acc=z, lam_n=z, n_upd=z, n_upd_fast=z, guard_on=z)

def ctrl_step(cs, obs, prop, cfg, Lnom, mnom, extra_tau=None, memfn=None):
    """One control period. obs: measured q, dq, tau_ext. prop: brain proposal (qdp, Kp). memfn(q, z) -> (u_ad, G): learned memory
    (u_ad = G^T (a_bar + z)); default: nominal body prior plus a fixed basis. Returns (tau, new state, diagnostics)."""
    qm, dqm, te = obs['q'], obs['dq'], obs['tau_ext']
    # tank pays the realized actuator power of the previous period (measured joint increment)
    T = cs['T'] - cs['tau_prev'] @ (qm - cs['q_prev'])
    # rest indicator
    still = jnp.max(jnp.abs(dqm)) <= cfg['v_r']
    nrest = jnp.where(still, cs['nrest'] + 1, 0); zeta = (nrest >= cfg['n_r']).astype(qm.dtype)
    # contact guard on the goal and stiffness filters
    qd, K = cs['qd'], cs['K']
    dqd = cfg['Lq'] * (prop['qdp'] - qd) * P.TS
    dK = cfg['LK'] * (jnp.clip(prop['Kp'], cfg['klo'], cfg['khi']) - K) * P.TS
    eq = qm - qd; spring = K * eq
    pushes = spring @ (dK * eq - K * dqd) > 0
    guard = (jnp.linalg.norm(te) >= cfg['tau_max']) & pushes
    qd = qd + jnp.where(guard, 0.0, 1.0) * dqd; K = K + jnp.where(guard, 0.0, 1.0) * dK
    eq = qm - qd; spring = K * eq
    # hand
    contact = jnp.linalg.norm(te) > cfg['tau_th']
    spring_s = jnp.where(contact, spring / jnp.maximum(1.0, jnp.linalg.norm(spring) / cfg['tau_max']), spring)
    uad, G = memory_out(qm, cs['z'], cfg, Lnom, mnom) if memfn is None else memfn(qm, cs['z'])
    tau_h = uad - spring_s - jnp.asarray(cfg['D'], qm.dtype) * dqm
    if extra_tau is not None: tau_h = tau_h + extra_tau
    # gate
    alpha_lvl = jnp.clip((T - cfg['Tmin']) / cfg['dT'], 0.0, 1.0)
    alpha = jnp.where(cfg.get('gate_sign', 0.0) > 0.5, jnp.where(dqm @ tau_h <= -cfg.get('pm', 0.0), 1.0, alpha_lvl), alpha_lvl)   # gate_sign 1: L0 sign exception (superseded)
    # constraint layer and saturation
    tau_b, rho = gauge_torque(qm, dqm, cfg)
    v = alpha * tau_h + tau_b; ub = jnp.asarray(P.UB, qm.dtype)
    lam = 1.0 / jnp.maximum(1.0, jnp.max(jnp.abs(v) / ub)); tau = lam * v
    # memory: measured mismatch with the torque applied in the period the sensed contact torque refers to
    mhat = uad - cs['tau_prev'] - te
    # law 0 (A, spec rev. 11): G K_n K^-1 m_hat; law 1 (B): G m_hat; law 2 (N, normalized): G (G^T G + eps I)^-1 m_hat, so that d u_ad/dt = -lam_s m_hat
    GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
    drive = jnp.where(cfg['law'] > 1.5, jnp.linalg.solve(GtG + eps * jnp.eye(3, dtype=qm.dtype), mhat),
                      jnp.where(cfg['law'] > 0.5, mhat, jnp.asarray(cfg['Kn'], qm.dtype) / K * mhat))
    z = jnp.clip(cs['z'] - P.TS * cfg['lam_s'] * zeta * (G @ drive), -cfg['zbar'], cfg['zbar'])
    new = dict(cs, qd=qd, K=K, z=z, T=T, nrest=nrest, tau_prev=tau, q_prev=qm, lam_acc=cs['lam_acc'] + lam, lam_n=cs['lam_n'] + 1.0,
               n_upd=cs['n_upd'] + zeta, guard_on=cs['guard_on'] + guard.astype(qm.dtype))
    diag = dict(tau=tau, tau_h=tau_h, tau_b=tau_b, alpha=alpha, lam=lam, rho=rho, zeta=zeta, uad=uad, mhat=mhat, guard=guard.astype(qm.dtype), T=T, spring=spring_s, contact=contact.astype(qm.dtype))
    return tau, new, diag

def ledger(cs, qm, cfg, Lnom, mnom, memfn=None):
    """(T, mean saturation factor over the interval, learned load G^T z at the current posture). Resets the interval accumulators."""
    uad, G = memory_out(qm, cs['z'], cfg, Lnom, mnom) if memfn is None else memfn(qm, cs['z'])
    lam_bar = jnp.where(cs['lam_n'] > 0, cs['lam_acc'] / jnp.maximum(cs['lam_n'], 1.0), 1.0)
    return dict(T=cs['T'], lam=lam_bar, uload=G.T @ cs['z']), dict(cs, lam_acc=jnp.zeros_like(cs['lam_acc']), lam_n=jnp.zeros_like(cs['lam_n']))
