import numpy as np, json

G0, M0, LC, L, I0, UB = 9.81, 1.5, 0.2, 0.4, 0.05, 20.0
QLO, QHI, CEN, HALF = -0.4, 1.0, 0.3, 0.7
K1, KB, RON, RFULL = 2.0, 20.0, 0.8, 0.9
TS, NSUB, NB, TEND = 1e-3, 4, 50, 10.0
QH, K0, KHI, D0, LQ, LK = 0.3, 60.0, 200.0, 3.0, 20.0, 10.0
GA, LS, ZB = 2.0, 50.0, 10.0
CENT, WID = np.linspace(QLO, QHI, 8), 0.2
TTH, TMAX, TTGT, TMIN, DTG, SIG = 0.5, 6.0, 3.0, 1.0, 1.0, 0.05

def feats(q):
    return np.concatenate([np.exp(-(q[:, None] - CENT[None, :]) ** 2 / (2 * WID ** 2)), np.ones((q.size, 1))], 1)

rng = np.random.default_rng(0)
plants = []
for pay in (0.0, 2 / 3, 4 / 3, 2.0):
    for ke in (500.0, 2000.0, 5000.0):
        for _ in range(2):
            plants.append(dict(pay=pay, ke=ke, fv=rng.uniform(0.05, 0.3), d=rng.uniform(-0.5, 0.5), be=rng.uniform(1, 5),
                               qw=rng.uniform(0.5, 0.6), bias=rng.uniform(-0.5, 0.5)))
runs = []
SPEC = []
for g, var, brain, va, T0, wall in [('E_nom', 'E', 'nom', 0.1, 20.0, 1), ('M_nom', 'M', 'nom', 0.1, 20.0, 1), ('E_adv', 'E', 'adv', 0.1, 20.0, 1),
                                    ('M_adv', 'M', 'adv', 0.1, 20.0, 1), ('E_v0.05', 'E', 'nom', 0.05, 20.0, 1), ('E_v0.2', 'E', 'nom', 0.2, 20.0, 1),
                                    ('E_v0.4', 'E', 'nom', 0.4, 20.0, 1), ('E_box', 'E', 'box', 0.1, 20.0, 0), ('E_gate', 'E', 'nom', 0.1, 1.2, 1)]:
    pass
for var in ('E', 'M', 'Mp'):
    for guard in ('block', 'yield', 'cap'):
        for brain in ('nom', 'adv'):
            SPEC.append((f'{var}_{brain}_{guard}', var, brain, 0.1, 20.0, 1, guard, 'sign'))
SPEC += [('E_gate_sign', 'E', 'nom', 0.1, 1.2, 1, 'block', 'sign'), ('E_gate_margin', 'E', 'nom', 0.1, 1.2, 1, 'block', 'margin')]
for g, var, brain, va, T0, wall, guard, gate in SPEC:
    for i, p in enumerate(plants):
        runs.append(dict(p, group=g, var=var, brain=brain, va=va, T0=T0, wall=wall, plant=i, guard=guard, gate=gate))
N = len(runs)
A = lambda k: np.array([r[k] for r in runs], dtype=float)
pay, ke, fv, dd, be, bias, va, T0 = A('pay'), A('ke'), A('fv'), A('d'), A('be'), A('bias'), A('va'), A('T0')
qw = np.where(A('wall') > 0, A('qw'), 10.0)
isM = np.array([r['var'] in ('M', 'Mp') for r in runs]); isMp = np.array([r['var'] == 'Mp' for r in runs]); gy = np.array([r['guard'] == 'yield' for r in runs]); gc = np.array([r['guard'] == 'cap' for r in runs]); gm = np.array([r['gate'] == 'margin' for r in runs]); adv = np.array([r['brain'] == 'adv' for r in runs]); box = np.array([r['brain'] == 'box' for r in runs])
Itot = I0 + M0 * LC ** 2 + pay * L ** 2; gk = G0 * (M0 * LC + pay * L)
ustar = lambda q: gk * np.cos(q) - dd
Erob = lambda q, dq: 0.5 * Itot * dq ** 2 + gk * np.sin(q) - dd * q
grid = np.linspace(QLO, QHI, 4001)
Emin = np.min(gk[:, None] * np.sin(grid[None, :]) - dd[:, None] * grid[None, :], 1)

def contact(q, dq):
    dl = q - qw
    return -np.where(dl > 0, np.maximum(ke * dl + be * dq, 0.0), 0.0)

def deriv(x, tau):
    q, dq = x[0], x[1]
    tc = contact(q, dq)
    ddq = (tau + dd + tc - fv * dq - gk * np.cos(q)) / Itot
    return np.stack([dq, ddq, fv * dq ** 2, tc * dq])

q = np.full(N, QH); dq = np.zeros(N); z = np.zeros((N, 9)); qd = np.full(N, QH); K = np.full(N, K0); T = T0.copy()
Wf = np.zeros(N); Wc = np.zeros(N); Wsens = np.zeros(N); E0 = Erob(q, dq)
qdp, Kp, kapp = np.full(N, QH), np.full(N, K0), np.zeros(N)
inc, qc, tc0, qpress = np.zeros(N, bool), np.zeros(N), np.full(N, np.inf), np.zeros(N)
tcprev, bsum, bcnt, bhat = np.zeros(N), np.zeros(N), 0, np.zeros(N)
crossed, unblk, maxinc, spr_prev = np.zeros(N, bool), np.zeros(N, int), np.zeros(N), np.zeros(N)
dr = np.zeros((N, 3)); satc, ncon = np.zeros(N), np.zeros(N); tau_prev = np.zeros(N)
first_true = np.full(N, np.inf); peak = np.zeros(N); viol = np.zeros(N, int); minT = T0.copy(); maxres = np.zeros(N); maxdel = np.zeros(N); maxdel_s = np.zeros(N)
mis05 = np.zeros(N); mis20 = np.zeros(N); acc_tc = np.zeros(N); acc_m = np.zeros(N); nacc = 0; maxrho = np.zeros(N)
trace_ids = []
tr = {i: [] for i in trace_ids}
for k in range(int(TEND / TS)):
    t = k * TS
    tau_hat = tcprev + bias + rng.normal(0, SIG, N)
    if 1.5 <= t < 2.0:
        bsum += tau_hat; bcnt += 1
    if abs(t - 2.0) < 1e-9:
        bhat = bsum / bcnt
    tared = t >= 2.0 - 1e-12
    ts = tau_hat - bhat
    if k % NB == 0:
        qdp, Kp, kapp = np.full(N, QH), np.full(N, K0), np.where((t >= 0.5) & (t < 2.0), 1.0, 0.0)
        app = t >= 2.2
        new = app & ~inc & (np.abs(ts) > TTH) & ~box
        qc = np.where(new, q, qc); tc0 = np.where(new, t, tc0); qpress = np.where(new, q + TTGT / K0, qpress); inc = inc | new
        qpress = np.where(inc & ~new & ~adv, qpress + 0.3 * (TTGT - np.abs(ts)) / K0, qpress)
        qpress = np.where(inc & ~new & adv, qpress + 0.025, qpress)
        qdp = np.where(app & ~inc, np.clip(QH + va * (t - 2.2), QLO, QHI), qdp)
        qdp = np.where(inc, np.clip(qpress, QLO, QHI), qdp)
        Kp = np.where(inc & adv, KHI, Kp)
        kapp = np.where(inc & ((isM & (t >= tc0 + 0.5)) | adv), 1.0, kapp)
        qdp = np.where(box & app, np.where(t < 6.0, QHI, QLO), qdp); Kp = np.where(box & app, KHI, Kp)
    dqd = LQ * (qdp - qd) * TS; dK = LK * (Kp - K) * TS
    blk = tared & (np.abs(ts) >= TMAX)
    dqd = np.where(blk & ~gy & (dqd * np.sign(-ts) > 0), 0.0, dqd); dK = np.where(blk & (dK > 0), 0.0, dK)
    dqd = np.where(blk & gy, -20.0 * (np.abs(ts) - TMAX) / K * np.sign(-ts) * TS, dqd)
    crossed = crossed | blk
    unblk += (crossed & ~blk & inc).astype(int)
    qd = qd + dqd; K = K + dK
    eq = q - qd
    kap = np.where(isM, kapp, kapp * (np.abs(ts) <= TTH))
    Gq = feats(q); uad = GA * np.sum(Gq * z, 1)
    c = np.where(isM & tared, np.where(isMp, (uad - tau_prev - ts) / K, eq - ts / K), eq)
    spr = np.where(gc, np.clip(K * eq, -TMAX, TMAX), K * eq)
    inc_spr = np.abs(spr) - np.abs(spr_prev); spr_prev = spr
    maxinc = np.where(crossed & ~blk & inc, np.maximum(maxinc, inc_spr), maxinc)
    tauh = uad - spr - D0 * dq + np.where(box, 0.3 * UB * np.sign(qdp - q), 0.0)
    alpha = np.where(dq * tauh > np.where(gm, -0.5, 0.0), np.clip((T - TMIN) / DTG, 0, 1), 1.0)
    eta = dq + K1 * q; w = (eta - K1 * CEN) / (K1 * HALF); rho = np.abs(w); maxrho = np.maximum(maxrho, rho)
    a = np.clip((rho - RON) / (RFULL - RON), 0, 1)
    tb = -a * KB * rho / np.maximum(1 - rho ** 2, 1e-9) * np.sign(w)
    tau = np.clip(alpha * tauh + tb, -UB, UB)
    satc += (inc & (np.abs(tau) >= UB - 1e-9)).astype(float); ncon += inc.astype(float)
    m_now = uad - ustar(q)
    if abs(t - 0.5) < 1e-9: mis05 = np.abs(m_now)
    if abs(t - 2.0) < 1e-9: mis20 = np.abs(m_now)
    if t >= 9.0:
        acc_tc += np.abs(contact(q, dq)); acc_m += np.abs(m_now); nacc += 1
    for i in trace_ids:
        if k % 10 == 0:
            tr[i].append([t, q[i], qd[i], abs(contact(q, dq)[i]), m_now[i], T[i], kap[i]])
    x = np.stack([q, dq, Wf, Wc]); h = TS / NSUB; q_old = q.copy()
    for _ in range(NSUB):
        k1 = deriv(x, tau); k2 = deriv(x + 0.5 * h * k1, tau); k3 = deriv(x + 0.5 * h * k2, tau); k4 = deriv(x + h * k3, tau)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tcs = contact(x[0], x[1]); tnow = t + (_ + 1) * h
        newc = (tcs < 0) & ~np.isfinite(first_true); first_true = np.where(newc, tnow, first_true)
        peak = np.where(np.isfinite(first_true) & (tnow <= first_true + 0.3), np.maximum(peak, -tcs), peak)
    q, dq, Wf, Wc = x
    dqk = q - q_old
    Wsens += ts * dqk
    dTk = -tau * dqk
    below = (T < TMIN) & (dTk < 0)
    cat = np.where(alpha >= 1.0, 0, np.where(alpha > 0.0, 1, 2))
    for ci in range(3): dr[:, ci] += np.where(below & (cat == ci), -dTk, 0.0)
    T = T + dTk; minT = np.minimum(minT, T); tau_prev = tau
    z = np.clip(z - feats(0.5 * (q + q_old)) * dqk[:, None] - (kap * LS * c * TS)[:, None] * Gq, -ZB, ZB)
    tcprev = contact(q, dq)
    viol += ((q < QLO) | (q > QHI)).astype(int)
    res = Erob(q, dq) + T + Wf - Wc - (E0 + T0)
    scale = np.maximum(1.0, T0 + np.abs(Erob(q, dq) - E0))
    maxres = np.maximum(maxres, np.abs(res) / scale)
    maxdel = np.maximum(maxdel, -Wc); maxdel_s = np.maximum(maxdel_s, -Wsens)
bound = E0 - Emin + T0
out = []
for i, r in enumerate(runs):
    out.append(dict(group=r['group'], plant=r['plant'], pay=r['pay'], ke=r['ke'], va=r['va'], id_res=float(maxres[i]), delivered=float(maxdel[i]),
                    delivered_sensed=float(maxdel_s[i]), bound=float(bound[i]), viol=int(viol[i]), max_rho=float(maxrho[i]),
                    mis_ratio=float(mis20[i] / max(mis05[i], 1e-12)), tc_ss=float(acc_tc[i] / nacc), mis_ss=float(acc_m[i] / nacc),
                    peak=float(peak[i]), reached=bool(np.isfinite(first_true[i])), minT=float(minT[i]), unblocked=int(unblk[i]), max_spring_inc=float(maxinc[i]),
                    drain=[float(x) for x in dr[i]], sat_frac=float(satc[i] / max(ncon[i], 1))))
json.dump(out, open('../records/m1b_fixes.json', 'w'), indent=0)

def grp(g): return [o for o in out if o['group'] == g]
S = {}
for g in sorted(set(o['group'] for o in out)):
    R = grp(g)
    S[g] = dict(tc_ss=[round(min(o['tc_ss'] for o in R), 3), round(max(o['tc_ss'] for o in R), 3)], tc_err=round(max(abs(o['tc_ss'] - 3) for o in R), 4),
                n_above_6_5=sum(o['tc_ss'] > 6.5 for o in R), mis_ss=[round(min(o['mis_ss'] for o in R), 4), round(max(o['mis_ss'] for o in R), 4)],
                n_mis_above_0_3=sum(o['mis_ss'] > 0.3 for o in R), viol=sum(o['viol'] for o in R), minT=round(min(o['minT'] for o in R), 4),
                n_minT_fail=sum(o['minT'] < 0.95 for o in R), reached=sum(o['reached'] for o in R))
    if 'adv_block' in g:
        S[g]['Q1'] = all((o['unblocked'] > 0 and o['max_spring_inc'] > 0.5) for o in R if o['tc_ss'] > 6.5)
        S[g]['unblocked'] = [min(o['unblocked'] for o in R), max(o['unblocked'] for o in R)]; S[g]['max_spring_inc'] = [round(min(o['max_spring_inc'] for o in R), 3), round(max(o['max_spring_inc'] for o in R), 3)]
    if g.startswith('M_adv'):
        S[g]['Q4_sat'] = all(o['sat_frac'] > 0 for o in R if o['mis_ss'] > 0.3)
    if 'gate' in g:
        S[g]['drain_alpha1_alpha_mid_alpha0'] = [round(sum(o['drain'][j] for o in R), 4) for j in range(3)]
    print(g, json.dumps(S[g]))
json.dump(S, open('../records/m1b_summary.json', 'w'), indent=1)
