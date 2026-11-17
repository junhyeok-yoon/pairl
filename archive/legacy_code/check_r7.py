import numpy as np, json

# Numerical check of the tank-paid debt result (R7) on a one-joint arm with a joint limit, a passive wall,
# input saturation, the constraint layer, the energy gate and a port-Hamiltonian memory (z_a: 3, z_f: 2).
G0, LO, HI = 9.81, -0.2, 1.2
CEN, HALF, K1 = (LO + HI) / 2, (HI - LO) / 2, 2.0
KA, OM, RHO, GF = np.diag([5.0, 5.0, 5.0]), 4.0, 1.0, np.array([0.5, 0.0])
JF, RF = np.array([[0.0, OM], [-OM, 0.0]]), np.diag([0.0, RHO])
D_DAMP, LQ, LK, TMIN, TMAX, DT_GATE, KB = 2.0, 3.0, 3.0, 0.5, 5.0, 0.5, 20.0

def feats(q):
    return np.array([np.cos(q), np.sin(q), 1.0])

def setup(seed, ubar, wall_k, press=False):
    r = np.random.default_rng(seed)
    p = dict(I=0.4 + 0.16 * r.uniform(0, 2), a=(1.0 + 0.4 * r.uniform(0, 2)) * G0, f=r.uniform(0.1, 0.5),
             qw=r.uniform(0.7, 1.1), ke=wall_k, be=0.5, ubar=ubar, abar=np.array([8.0, 0.0, 0.0]))
    p['cstar'] = np.array([p['a'], 0.0, 0.0])
    p['zastar'] = np.linalg.solve(KA, p['cstar'] - p['abar'])
    times = np.arange(0, 5.0, 0.5)
    p['props'] = [(t, r.uniform(LO + 0.05, HI - 0.02), r.uniform(5, 60)) for t in times]
    if press:                                    # proposals behind the wall: sustained contact
        p['props'] = [(t, min(p['qw'] + 0.15, HI - 0.02), r.uniform(20, 60)) for t in times]
    return p

def parts(x, t, p):
    q, dq, za, zf, qd, K, T = x[0], x[1], x[2:5], x[5:7], x[7], x[8], x[9]
    qdp, Kp = [(a, b) for (s, a, b) in p['props'] if s <= t + 1e-12][-1]
    gradf = zf
    uad = feats(q) @ (p['abar'] + KA @ za) + GF @ gradf
    eq = q - qd
    tau_h = uad - K * eq - D_DAMP * dq
    w = (dq + K1 * q - K1 * CEN) / (K1 * HALF)
    a = np.clip((abs(w) - 0.8) / 0.1, 0, 1)
    tau_b = -a * KB * w / max(1 - w * w, 1e-6)
    v = tau_h + tau_b
    lam = 1.0 / max(1.0, abs(v) / p['ubar'])
    tau = lam * v
    delta = max(q - p['qw'], 0.0)
    fe = -(p['ke'] * delta + p['be'] * dq) if delta > 0 else 0.0
    qd_nom, K_nom = LQ * (qdp - qd), LK * (Kp - K)
    Pnom = 0.5 * eq * K_nom * eq - eq * K * qd_nom
    gam = 1.0 if Pnom <= 0 else float(np.clip((T - TMIN) / DT_GATE, 0, 1))
    Pdem = gam * Pnom
    Pdiss = D_DAMP * dq * dq + gradf @ RF @ gradf
    phi = 1.0 if T < TMAX else 0.0
    Pdelta = dq * (tau - tau_h)
    ddq = (tau - p['a'] * np.cos(q) - p['f'] * dq + fe) / p['I']
    dza = -feats(q) * dq
    dzf = (JF - RF) @ gradf - GF * dq
    dT = phi * Pdiss - Pdem - Pdelta
    rhs = -p['f'] * dq * dq - (1 - phi) * Pdiss + dq * fe
    xdot = np.concatenate([[dq, ddq], dza, dzf, [gam * qd_nom, gam * K_nom, dT]])
    return xdot, dict(rhs=rhs, Pdelta=Pdelta, active=(a > 0) or (lam < 1), lam=lam, a=a)

def storage(x, p):
    q, dq, za, zf, qd, K, T = x[0], x[1], x[2:5], x[5:7], x[7], x[8], x[9]
    E = 0.5 * p['I'] * dq * dq + 0.5 * K * (q - qd) ** 2
    S = 0.5 * (za - p['zastar']) @ KA @ (za - p['zastar']) + 0.5 * zf @ zf
    return E + S + T

def run(seed, ubar, wall_k, T0=2.0, dt=1e-4, tend=5.0, press=False):
    p = setup(seed, ubar, wall_k, press)
    x = np.zeros(10); x[0], x[7], x[8], x[9] = 0.3, 0.3, 20.0, T0
    W0, acc, debt, debt_inactive, Tmin_seen, n_act, n = storage(x, p), 0.0, 0.0, 0.0, T0, 0, 0
    n_contact, pen = 0, 0.0
    qmax = qmin = x[0]
    for k in range(int(tend / dt)):
        t = k * dt
        k1, i1 = parts(x, t, p); k2, i2 = parts(x + 0.5 * dt * k1, t + 0.5 * dt, p)
        k3, i3 = parts(x + 0.5 * dt * k2, t + 0.5 * dt, p); k4, i4 = parts(x + dt * k3, t + dt, p)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acc += dt / 6 * (i1['rhs'] + 2 * i2['rhs'] + 2 * i3['rhs'] + i4['rhs'])
        pd = dt / 6 * (i1['Pdelta'] + 2 * i2['Pdelta'] + 2 * i3['Pdelta'] + i4['Pdelta'])
        debt += max(pd, 0.0)
        if not i1['active'] and abs(i1['Pdelta']) > 1e-9:
            debt_inactive += abs(pd)
        n_act += i1['active']; n += 1; n_contact += x[0] > p['qw']; pen = max(pen, x[0] - p['qw'])
        Tmin_seen = min(Tmin_seen, x[9]); qmax = max(qmax, x[0]); qmin = min(qmin, x[0])
    W1 = storage(x, p)
    return dict(seed=seed, ubar=ubar, wall_k=wall_k, identity_error=abs((W1 - W0) - acc), W0=W0, dW=W1 - W0,
                cumulative_debt=debt, debt_while_inactive=debt_inactive, min_T=Tmin_seen, active_fraction=n_act / n, q_max=qmax, q_min=qmin,
                joint_violation=bool(qmin < LO or qmax > HI), gravity_peak=p['a'], authority_holds=bool(p['a'] < ubar),
                press=press, contact_fraction=n_contact / n, max_penetration=pen)

if __name__ == '__main__':
    cases = [(s, 25.0, kw, False) for s in range(8) for kw in (500.0, 5000.0)] + [(1, 15.0, 2000.0, False)] + [(s, 25.0, kw, True) for s in range(6) for kw in (500.0, 5000.0)]
    res = [run(s, ub, kw, T0=(4.0 if pr else 2.0), press=pr) for s, ub, kw, pr in cases]
    for r in res:
        print(r['seed'], r['ubar'], r['wall_k'], 'auth', r['authority_holds'], 'id_err %.1e' % r['identity_error'], 'debt %.4f' % r['cumulative_debt'],
              'press', r['press'], 'contact %.2f' % r['contact_fraction'], 'pen %.4f' % r['max_penetration'], 'debt_inactive %.1e' % r['debt_while_inactive'], 'minT %.4f' % r['min_T'], 'act %.3f' % r['active_fraction'], 'q [%.3f, %.3f]' % (r['q_min'], r['q_max']), 'viol', r['joint_violation'])
    json.dump([{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()} for r in res], open('../records/r7_check.json', 'w'), indent=1)
