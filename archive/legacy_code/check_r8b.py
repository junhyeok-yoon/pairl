import numpy as np, json
from check_r7 import G0, LO, HI, CEN, HALF, K1, KA, D_DAMP, LQ, LK, TMIN, DT_GATE, KB, feats

# Full-power tank (the tank pays the realized actuator power) with an error-driven memory that the brain switches on
# for calibration at rest: storage identity, contact-torque accuracy, wall reached, tank minimum, joint limits.
TAU_C, LAM_S, KSTIFF, T_CAL0, T_CAL1 = 3.0, 10.0, 60.0, 1.5, 3.0

def setup(seed, wall_k):
    r = np.random.default_rng(seed)
    return dict(I=0.4 + 0.16 * r.uniform(0, 2), a=(1.0 + 0.4 * r.uniform(0, 2)) * G0, f=r.uniform(0.1, 0.5),
                qw=r.uniform(0.7, 1.1), ke=wall_k, be=0.5, ubar=25.0, abar=np.array([8.0, 0.0, 0.0]))

SENSE_THR, SENSE_STD = 0.5, 0.05          # contact threshold on the sensed contact torque [Nm], sensor noise

def script(t, p, mode):
    K = KSTIFF
    if t < T_CAL0:
        return p['qw'] - 0.05, K, 0.0
    if t < T_CAL1:
        return p['qw'] - 0.05, K, (1.0 if mode in ('calib', 'calib_in_contact', 'calib_in_contact_sensed') else 0.0)
    return p['qw'] + TAU_C / K, K, (1.0 if mode in ('calib_in_contact', 'calib_in_contact_sensed') else 0.0)

def parts(x, t, p, mode, noise=0.0):
    q, dq, za, qd, K, T = x[0], x[1], x[2:5], x[5], x[6], x[7]
    qdp, Kp, kap = script(t, p, mode)
    delta0 = max(q - p['qw'], 0.0)
    tau_c_meas = (p['ke'] * delta0 + p['be'] * dq if delta0 > 0 else 0.0) + noise    # sensed contact torque
    if mode == 'calib_in_contact_sensed' and tau_c_meas > SENSE_THR:
        kap = 0.0                                                                    # memory frozen in sensed contact
    uad = feats(q) @ (p['abar'] + KA @ za)
    eq = q - qd
    tau_h = uad - K * eq - D_DAMP * dq
    w = (dq + K1 * q - K1 * CEN) / (K1 * HALF)
    act = np.clip((abs(w) - 0.8) / 0.1, 0, 1)
    tau_b = -act * KB * w / max(1 - w * w, 1e-6)
    alpha = 1.0 if dq * tau_h <= 0 else float(np.clip((T - TMIN) / DT_GATE, 0, 1))   # gate the injecting hand torque only
    v = alpha * tau_h + tau_b
    lam = 1.0 / max(1.0, abs(v) / p['ubar'])
    tau = lam * v
    delta = max(q - p['qw'], 0.0)
    fe = -(p['ke'] * delta + p['be'] * dq) if delta > 0 else 0.0
    ddq = (tau - p['a'] * np.cos(q) - p['f'] * dq + fe) / p['I']
    dza = -feats(q) * (dq + kap * LAM_S * eq)                    # error-driven memory while the brain asks for calibration
    dT = -dq * tau                                                # the tank pays the realized actuator power
    rhs = -p['f'] * dq * dq + dq * fe
    xdot = np.concatenate([[dq, ddq], dza, [LQ * (qdp - qd), LK * (Kp - K), dT]])
    return xdot, dict(rhs=rhs, mis=uad - p['a'] * np.cos(q), fe=fe)

def storage(x, p):
    return 0.5 * p['I'] * x[1] ** 2 + p['a'] * np.sin(x[0]) + x[7]

def run(seed, wall_k, mode, T0=15.0, dt=1e-4, tend=6.0):
    p = setup(seed, wall_k)
    x = np.zeros(8); x[0], x[5], x[6], x[7] = 0.3, 0.3, KSTIFF, T0
    W0, acc, Tmin_seen, qmin, qmax, mis25, tail, fmax = storage(x, p), 0.0, T0, x[0], x[0], None, [], 0.0
    rng = np.random.default_rng(1000 + seed)
    for k in range(int(tend / dt)):
        t = k * dt
        nz = rng.normal(0, SENSE_STD)
        k1, i1 = parts(x, t, p, mode, nz); k2, i2 = parts(x + 0.5 * dt * k1, t + 0.5 * dt, p, mode, nz)
        k3, i3 = parts(x + 0.5 * dt * k2, t + 0.5 * dt, p, mode, nz); k4, i4 = parts(x + dt * k3, t + dt, p, mode, nz)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acc += dt / 6 * (i1['rhs'] + 2 * i2['rhs'] + 2 * i3['rhs'] + i4['rhs'])
        Tmin_seen = min(Tmin_seen, x[7]); qmin = min(qmin, x[0]); qmax = max(qmax, x[0]); fmax = max(fmax, -i1['fe'])
        if mis25 is None and t >= T_CAL1:
            mis25 = abs(i1['mis'])
        if t >= tend - 0.5:
            tail.append(-i1['fe'])
    tc = float(np.mean(tail))
    return dict(seed=seed, wall_k=wall_k, mode=mode, identity_error=float(abs(storage(x, p) - W0 - acc)), mismatch_at_press=float(mis25),
                contact_torque=tc, contact_error=abs(tc - TAU_C), peak_contact=float(fmax), reached_wall=bool(qmax > p['qw']),
                min_T=float(Tmin_seen), joint_violation=bool(qmin < LO or qmax > HI))

if __name__ == '__main__':
    res = [run(s, 2000.0, m) for s in range(4) for m in ('none', 'calib', 'calib_in_contact', 'calib_in_contact_sensed')]
    for r in res:
        print(r['seed'], r['mode'], 'id %.1e' % r['identity_error'], 'mis %.3f' % r['mismatch_at_press'], 'tc %.3f' % r['contact_torque'],
              'err %.3f' % r['contact_error'], 'peak %.2f' % r['peak_contact'], 'wall', r['reached_wall'], 'minT %.3f' % r['min_T'], 'viol', r['joint_violation'])
    json.dump(res, open('../records/r8b_check.json', 'w'), indent=1)
