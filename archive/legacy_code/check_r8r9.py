import numpy as np, json
from check_r7 import G0, LO, HI, CEN, HALF, K1, KA, RF, JF, GF, D_DAMP, LQ, LK, TMIN, TMAX, DT_GATE, KB, feats

# Numerical check of the tank-paid calibration (R8) and of liveness with calibration and relaxation (R9), one joint.
USTAR_BAR, EPS_BAR, TAU_C = 18.0, 4.0, 3.0      # class bound of |u*|, calibration intensity, target contact torque

def setup(seed, wall_k):
    r = np.random.default_rng(seed)
    p = dict(I=0.4 + 0.16 * r.uniform(0, 2), a=(1.0 + 0.4 * r.uniform(0, 2)) * G0, f=r.uniform(0.1, 0.5),
             qw=r.uniform(0.7, 1.1), ke=wall_k, be=0.5, ubar=25.0, abar=np.array([8.0, 0.0, 0.0]))
    p['zastar'] = np.linalg.solve(KA, np.array([p['a'], 0.0, 0.0]) - p['abar'])
    return p

def script(t, x, p, mode):
    # scripted brain: approach to a free-space point near the wall, optional calibration at rest, then press
    q_near, K = p['qw'] - 0.05, 40.0
    if mode == 'relax' and t < 1.5:
        return x[0], K, 0.0                          # relaxation proposal: goal at the current position
    if t < 1.5:
        return q_near, K, 0.0
    if t < 2.5:
        return q_near, K, (EPS_BAR if mode in ('calib', 'relax') else 0.0)
    return p['qw'] + TAU_C / K, K, 0.0

def parts(x, t, p, mode):
    q, dq, za, zf, qd, K, T = x[0], x[1], x[2:5], x[5:7], x[7], x[8], x[9]
    qdp, Kp, eps = script(t, x, p, mode)
    gradf = zf
    uad = feats(q) @ (p['abar'] + KA @ za) + GF @ gradf
    ustar = p['a'] * np.cos(q)
    eq = q - qd
    tau_h = uad - K * eq - D_DAMP * dq
    w = (dq + K1 * q - K1 * CEN) / (K1 * HALF)
    a = np.clip((abs(w) - 0.8) / 0.1, 0, 1)
    v = tau_h - a * KB * w / max(1 - w * w, 1e-6)
    lam = 1.0 / max(1.0, abs(v) / p['ubar'])
    tau = lam * v
    delta = max(q - p['qw'], 0.0)
    fe = -(p['ke'] * delta + p['be'] * dq) if delta > 0 else 0.0
    qd_nom, K_nom = LQ * (qdp - qd), LK * (Kp - K)
    Pnom = 0.5 * eq * K_nom * eq - eq * K * qd_nom
    gam = 1.0 if Pnom <= 0 else float(np.clip((T - TMIN) / DT_GATE, 0, 1))
    eps_r = eps * float(np.clip((T - TMIN) / DT_GATE, 0, 1))      # calibration is gated by the tank as well
    Pcal = eps_r * abs(K * eq) * (abs(uad) + USTAR_BAR)          # worst-case power of the calibration term
    Pdem, Pdiss, phi = gam * Pnom, D_DAMP * dq * dq + gradf @ RF @ gradf, (1.0 if T < TMAX else 0.0)
    Pdelta = dq * (tau - tau_h)
    ddq = (tau - p['a'] * np.cos(q) - p['f'] * dq + fe) / p['I']
    dza = -feats(q) * (dq + eps_r * K * eq)
    dzf = (JF - RF) @ gradf - GF * dq
    dT = phi * Pdiss - Pdem - Pdelta - Pcal
    cal_true = eps_r * (K * eq) * (uad - ustar)
    rhs = -p['f'] * dq * dq - (1 - phi) * Pdiss + dq * fe - (cal_true + Pcal)
    xdot = np.concatenate([[dq, ddq], dza, dzf, [gam * qd_nom, gam * K_nom, dT]])
    return xdot, dict(rhs=rhs, slack=cal_true + Pcal, mis=uad - ustar, fe=fe, contact=delta > 0)

def storage(x, p):
    q, dq, za, zf, qd, K, T = x[0], x[1], x[2:5], x[5:7], x[7], x[8], x[9]
    return 0.5 * p['I'] * dq * dq + 0.5 * K * (q - qd) ** 2 + 0.5 * (za - p['zastar']) @ KA @ (za - p['zastar']) + 0.5 * zf @ zf + T

def run(seed, wall_k, mode, T0=4.0, dt=1e-4, tend=5.0):
    p = setup(seed, wall_k)
    x = np.zeros(10); x[0], x[7], x[8], x[9] = 0.3, 0.3, 40.0, T0
    W0, acc, min_slack, Tmin_seen, qmin, qmax = storage(x, p), 0.0, np.inf, T0, x[0], x[0]
    mis_25 = None; tail = []
    for k in range(int(tend / dt)):
        t = k * dt
        k1, i1 = parts(x, t, p, mode); k2, i2 = parts(x + 0.5 * dt * k1, t + 0.5 * dt, p, mode)
        k3, i3 = parts(x + 0.5 * dt * k2, t + 0.5 * dt, p, mode); k4, i4 = parts(x + dt * k3, t + dt, p, mode)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acc += dt / 6 * (i1['rhs'] + 2 * i2['rhs'] + 2 * i3['rhs'] + i4['rhs'])
        min_slack = min(min_slack, i1['slack']); Tmin_seen = min(Tmin_seen, x[9]); qmin = min(qmin, x[0]); qmax = max(qmax, x[0])
        if mis_25 is None and t >= 2.5:
            mis_25 = abs(i1['mis'])
        if t >= tend - 0.5:
            tail.append(-i1['fe'])
    tc = float(np.mean(tail))
    return dict(seed=seed, wall_k=wall_k, mode=mode, identity_error=float(abs(storage(x, p) - W0 - acc)), min_slack=float(min_slack),
                mismatch_at_press=float(mis_25), contact_torque=tc, contact_error=abs(tc - TAU_C), reached_wall=bool(qmax > p['qw']),
                min_T=float(Tmin_seen), joint_violation=bool(qmin < LO or qmax > HI))

res = [run(s, kw, m) for s in range(4) for kw in (500.0, 5000.0) for m in ('none', 'calib')]
for r in res:
    print(r['seed'], r['wall_k'], r['mode'], 'id %.1e' % r['identity_error'], 'slack %.2e' % r['min_slack'], 'mis %.3f' % r['mismatch_at_press'],
          'tc %.3f' % r['contact_torque'], 'err %.3f' % r['contact_error'], 'wall', r['reached_wall'], 'minT %.3f' % r['min_T'], 'viol', r['joint_violation'])
for m in ('none', 'calib'):
    sub = [r for r in res if r['mode'] == m]
    print(m, 'reached %d/%d' % (sum(r['reached_wall'] for r in sub), len(sub)), 'median err %.3f' % np.median([r['contact_error'] for r in sub]),
          'max err %.3f' % max(r['contact_error'] for r in sub), 'median mis %.3f' % np.median([r['mismatch_at_press'] for r in sub]))
json.dump(res, open('../records/r8r9_check.json', 'w'), indent=1)
