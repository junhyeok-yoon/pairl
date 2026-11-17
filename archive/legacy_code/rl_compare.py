import os, sys, json, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax, vmap, jit
jax.config.update('jax_enable_x64', True)

# Two-link arm in a vertical plane: unknown payload and friction, reflected rotor inertia, input box, joint box.
# usage: python3 rl_compare.py ARCH SEED ITERS   (ARCH in A, B, C, auth)
K1, DT, T_EP, T_SW = 2.0, 1e-3, 3.0, 1.5
LO, HI = np.array([-0.2, 0.2]), np.array([1.2, 1.8])
CEN, HALF = 0.5 * (LO + HI), 0.5 * (HI - LO)
UBAR = np.array([60.0, 25.0]); ROT = np.array([0.3, 0.1]); G0 = 9.81
KC, GA, UFF, ALPHA, MR, C0 = 15.0, 10.0, 40.0, 4.0, 0.5, 0.3          # core and interface
PLO, PHI, PSI0, VBAR = 0.05, 0.4, 0.3, 4.0                            # action ranges
KF, ACT0, ACT1 = 20.0, 0.8, 0.9                                         # funnel safety filter of the baseline
NRF = 8
_r = np.random.default_rng(12345)
OM = _r.normal(0.0, 2.0, (NRF, 2)); PHS = _r.uniform(0, 2 * np.pi, NRF)
NF = 3 * (NRF + 1) + 2
P_POP, K_TASK, SIGMA, LR, N_TEST = 48, 3, 0.04, 0.02, 64
PMAX = 2.0
TAG = os.environ.get('RL_ROUND', '1')
if TAG == '2':                      # round 2: wider payload range; torque limits fixed by the launcher before training
    PMAX, UBAR = 4.0, np.array([float(v) for v in os.environ['RL_UBAR'].split(',')])
PREFIX = 'rl_' if TAG == '1' else 'rl2_'
FIX = [f for f in os.environ.get('RL_FIX', '').split(',') if f]      # repairs of the cause check: psi, gain, feat, target
if FIX:
    PREFIX = 'rlfix_' + '-'.join(FIX) + '_'
if 'psi' in FIX:
    PHI = 1.0

def theta_np(mp, f1, f2):
    m1, m2, l1, lc1 = 4.0, 2.0, 0.4, 0.2
    m2e = m2 + mp; lc2 = (m2 * 0.2 + mp * 0.4) / m2e
    I1 = m1 * l1 ** 2 / 12; I2 = m2 * 0.4 ** 2 / 12 + m2 * (lc2 - 0.2) ** 2 + mp * (0.4 - lc2) ** 2
    a1 = I1 + I2 + m1 * lc1 ** 2 + m2e * (l1 ** 2 + lc2 ** 2); a2 = m2e * l1 * lc2; a3 = I2 + m2e * lc2 ** 2
    return np.stack([a1, a2, a3, (m1 * lc1 + m2e * l1) * G0, m2e * lc2 * G0, f1, f2], -1)

def mass_np(q, th):
    c = np.cos(q[..., 1])
    m11 = th[..., 0] + 2 * th[..., 1] * c + ROT[0]; m12 = th[..., 2] + th[..., 1] * c; m22 = th[..., 2] + ROT[1]
    return np.stack([np.stack([m11, m12], -1), np.stack([m12, m22], -1)], -2)

def plant_acc(q, dq, u, th):
    a1, a2, a3, g1, g2, f1, f2 = th
    c, s = jnp.cos(q[1]), jnp.sin(q[1])
    m11 = a1 + 2 * a2 * c + ROT[0]; m12 = a3 + a2 * c; m22 = a3 + ROT[1]
    h = a2 * s
    n1 = -h * dq[1] * dq[0] - h * (dq[0] + dq[1]) * dq[1] + g1 * jnp.cos(q[0]) + g2 * jnp.cos(q[0] + q[1]) + f1 * dq[0]
    n2 = h * dq[0] * dq[0] + g2 * jnp.cos(q[0] + q[1]) + f2 * dq[1]
    b1, b2 = u[0] - n1, u[1] - n2
    det = m11 * m22 - m12 * m12
    return jnp.stack([(m22 * b1 - m12 * b2) / det, (-m12 * b1 + m11 * b2) / det])

def features(q, dq, a):
    # generic: random Fourier features of q, their products with the reference acceleration, and dq
    base = jnp.concatenate([jnp.ones(1), jnp.cos(jnp.asarray(OM) @ q + PHS)])
    return jnp.concatenate([base, 0.2 * jnp.outer(base, a).reshape(-1), dq])

W_PRIOR = None
if 'feat' in FIX:                    # meta-learned feature map and its mean task weights as the fast-weight prior
    from lib_meta import meta_phi
    _mf = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../records/meta_features.npz'))
    _MP = [jnp.asarray(_mf[k]) for k in sorted(k for k in _mf.files if k.startswith('p'))]
    W_PRIOR, NF = np.asarray(_mf['w_mean']), int(_mf['nf'])
    def features(q, dq, a):
        return meta_phi(_MP, q, dq, a, CEN, HALF)

def sat_dir(v):
    lam = 1.0 / jnp.maximum(1.0, jnp.max(jnp.abs(v) / UBAR))
    return lam * v, lam

def core_field(x, act, th):
    q, dq, r, psi = x[0:2], x[2:4], x[4:6], x[6]
    W = x[7:].reshape(NF, 2)
    vp, psip = act[:2], act[2]
    nu = act[3:5] if act.shape[0] == 5 else jnp.zeros(2)         # learned reflex of the spec, inside the feedforward budget
    kc = act[3] if act.shape[0] == 4 else KC                      # gain scale chosen by the policy (repair 'gain')
    e = dq + K1 * q - r; en2 = e @ e
    wn = jnp.sqrt(en2) / psi
    xi = e / jnp.maximum(psi ** 2 - en2, 1e-12)
    kap = kc * psi ** 2
    m_psi = jnp.clip(1.0 - wn, 0.0, 1.0)
    clr = jnp.min(jnp.minimum(r - K1 * LO, K1 * HI - r))
    m_c = jnp.clip((clr - psi) / C0, 0.0, 1.0)
    gr = jnp.minimum(1.0, jnp.minimum(m_psi, m_c) / MR); gp = jnp.minimum(1.0, m_psi / MR)
    dr = gr * vp
    Phi = features(q, dq, dr - K1 * dq)
    uad = Phi @ W + nu
    lff = jnp.minimum(1.0, UFF / jnp.maximum(jnp.linalg.norm(uad), 1e-12))
    u, lam = sat_dir(lff * uad - kap * xi)
    dpsi = -ALPHA * gp * jnp.maximum(psi - psip, 0.0) + ALPHA * m_c * jnp.maximum(psip - psi, 0.0)
    dW = -GA * lam * lff * jnp.outer(Phi, e)
    dx = jnp.concatenate([dq, plant_acc(q, dq, u, th), dr, dpsi[None], dW.reshape(-1)])
    return dx, (u, lam, gr, kap * jnp.sqrt(xi @ xi), wn)

def b_field(x, uRL, th):
    q, dq = x[:2], x[2:4]
    w = (dq + K1 * q - K1 * CEN) / (K1 * HALF)
    act = jnp.clip((jnp.abs(w) - ACT0) / (ACT1 - ACT0), 0.0, 1.0)
    u, lam = sat_dir(uRL - act * KF * w / jnp.maximum(1.0 - w ** 2, 1e-9))
    return jnp.concatenate([dq, plant_acc(q, dq, u, th)]), (u, lam, jnp.max(act), 0.0, jnp.max(jnp.abs(w)))

def rk4(f, x, a, th):
    k1, aux = f(x, a, th); k2, _ = f(x + 0.5 * DT * k1, a, th)
    k3, _ = f(x + 0.5 * DT * k2, a, th); k4, _ = f(x + DT * k3, a, th)
    return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), aux

def setup(arch, scripted=None):
    core = arch in ('A', 'C', 'D', 'E'); reflex = arch in ('D', 'E'); integ = arch == 'F'   # F: input policy observing the error integral
    n_in, n_out, H = (19, 4 if 'gain' in FIX else 3, 32) if core else ((8 if integ else 6), 2, 32)
    SH = [(H, n_in), (H,), (H, H), (H,), (n_out, H), (n_out,)]
    SHR = [(H, 6), (H,), (H, H), (H,), (2, H), (2,)] if reflex else []
    NP = int(sum(np.prod(s) for s in SH)); NPR = int(sum(np.prod(s) for s in SHR))
    ND = NP + NPR + (NF * 2 if core else 0)
    T_H = 0.05 if core else 0.01
    NSUB, NTICK = int(round(T_H / DT)), int(round(T_EP / T_H)); NR = int(round(0.01 / DT))
    f = core_field if core else b_field

    def policy(theta, o):
        ps, i = [], 0
        for s in SH:
            n = int(np.prod(s)); ps.append(theta[i:i + n].reshape(s)); i += n
        h = jnp.tanh(ps[0] @ o + ps[1]); h = jnp.tanh(ps[2] @ h + ps[3])
        return ps[4] @ h + ps[5]

    def reflex_net(theta, o):
        ps, i = [], NP
        for s in SHR:
            n = int(np.prod(s)); ps.append(theta[i:i + n].reshape(s)); i += n
        h = jnp.tanh(ps[0] @ o + ps[1]); h = jnp.tanh(ps[2] @ h + ps[3])
        return UFF * jnp.tanh(ps[4] @ h + ps[5])

    def episode(theta, th, q0, tgt):
        W0 = theta[NP + NPR:]
        x0 = jnp.concatenate([q0, jnp.zeros(2), K1 * q0, jnp.array([PSI0]), W0]) if core else jnp.concatenate([q0, jnp.zeros(2)])

        def tick(carry, k):
            x, prev, s, zI = carry
            t0 = k * T_H
            zI = jnp.where(jnp.abs(t0 - T_SW) < 1e-9, 0.0 * zI, zI)
            q, dq = x[:2], x[2:4]
            tq = jnp.where(t0 < T_SW - 1e-9, tgt[0], tgt[1])
            base = jnp.concatenate([(q - CEN) / HALF, dq / 2.0, (tq - CEN) / HALF])
            if core:
                r, psi = x[4:6], x[6]
                wn = jnp.linalg.norm(dq + K1 * q - r) / psi
                m_c = jnp.clip((jnp.min(jnp.minimum(r - K1 * LO, K1 * HI - r)) - psi) / C0, 0.0, 1.0)
                uff = features(q, dq, -K1 * dq) @ x[7:].reshape(NF, 2)
                rep = jnp.concatenate([jnp.array([jnp.clip(1 - wn, 0, 1), m_c, prev[0]]), (1.0 - prev[1]) * prev[3:5] / VBAR,
                                       jnp.array([(psi - prev[5]) / PHI, prev[2] / 25.0]), uff / UBAR,
                                       jnp.array([jnp.linalg.norm(x[7:] - W0) / 10.0])])
                if arch in ('C', 'E'):
                    rep = 0.0 * rep
                out = policy(theta, jnp.concatenate([base, (r / K1 - CEN) / HALF, jnp.array([psi / PHI]), rep]))
                if 'target' in FIX:          # target-level action, proportional steering of the reference
                    vp = 8.0 * (K1 * (CEN + HALF * jnp.tanh(out[:2])) - r)
                else:
                    vp = VBAR * jnp.tanh(out[:2])
                vp = vp * jnp.minimum(1.0, VBAR / jnp.maximum(jnp.linalg.norm(vp), 1e-12))
                a = jnp.concatenate([vp, jnp.array([PLO + (PHI - PLO) * jax.nn.sigmoid(out[2])])])
                if 'gain' in FIX:
                    a = jnp.concatenate([a, jnp.array([KC * 4.0 ** jax.nn.sigmoid(out[3])])])
                if scripted is not None:
                    a = scripted(x, tq)
            else:
                a = UBAR * jnp.tanh(policy(theta, jnp.concatenate([base, jnp.clip(2.0 * zI, -1.0, 1.0)]) if integ else base))

            def sub(c, j):
                x, s, l, nu, zI = c
                tq = jnp.where(t0 + j * DT < T_SW - 1e-9, tgt[0], tgt[1])
                if reflex:                                         # reflex updated at 100 Hz, as the input policy of the baseline
                    ob = jnp.concatenate([(x[:2] - CEN) / HALF, x[2:4] / 2.0, (tq - CEN) / HALF])
                    nu = lax.cond(j % NR == 0, lambda: reflex_net(theta, ob), lambda: nu)
                xn, (u, lam, y1, y2, wn) = rk4(f, x, jnp.concatenate([a, nu]) if reflex else a, th)
                qn = xn[:2]; err2 = jnp.sum((qn - tq) ** 2); zI = zI + (qn - tq) * DT
                tt = t0 + (j + 1) * DT
                inss = 1.0 * (((tt > T_SW - 0.5) & (tt <= T_SW)) | (tt > T_EP - 0.5))      # last 0.5 s of each target window
                viol = jnp.maximum(0.0, jnp.max(jnp.maximum(LO - qn, qn - HI)))
                flag = (y1 < 1.0) if core else (y1 > 0.0)
                s = s + jnp.array([-err2 - 0.01 * jnp.sum((u / UBAR) ** 2), jnp.sqrt(err2), 0.0, 0.0, lam, y1, y2, 1.0 * (lam < 1.0), 1.0 * flag, inss * jnp.sqrt(err2), inss])
                s = s.at[2].set(jnp.maximum(s[2], viol)).at[3].set(jnp.maximum(s[3], wn))
                return (xn, s, l + jnp.array([lam, y1, y2]), nu, zI), None

            (x, s, l, _, zI), _ = lax.scan(sub, (x, s, jnp.zeros(3), jnp.zeros(2), zI), jnp.arange(NSUB))
            prev = jnp.concatenate([l / NSUB, a[:2] if core else jnp.zeros(2), a[2:3] if core else jnp.zeros(1)])
            return (x, prev, s, zI), None

        prev0 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, PSI0])
        (x, _, s, _), _ = lax.scan(tick, (x0, prev0, jnp.zeros(11), jnp.zeros(2)), jnp.arange(NTICK))
        n = NSUB * NTICK
        return dict(ret=s[0] / n, err=s[1] / n, viol=s[2], wmax=s[3], lam=s[4] / n, y1=s[5] / n, eff=s[6] / n,
                    sat=s[7] / n, flag=s[8] / n, ss=s[9] / s[10])

    def init(seed):
        rng = np.random.default_rng(seed); th = []
        for s in SH:
            if len(s) == 2:
                th.append(rng.normal(0, 1 / np.sqrt(s[1]), s).ravel() * (0.1 if s is SH[4] else 1.0))
            else:
                th.append(np.zeros(s))
        for s in SHR:
            th.append(rng.normal(0, 1 / np.sqrt(s[1]), s).ravel() * (0.1 if s is SHR[4] else 1.0) if len(s) == 2 else np.zeros(s))
        th = np.concatenate(th)
        return np.concatenate([th, W_PRIOR.ravel() if (W_PRIOR is not None and core) else np.zeros(ND - NP - NPR)])

    return episode, init, ND

def sample_tasks(rng, n):
    th = theta_np(rng.uniform(0, PMAX, n), rng.uniform(0.2, 1, n), rng.uniform(0.2, 1, n))
    q0 = rng.uniform(LO + 0.15, HI - 0.15, (n, 2))
    tgt = rng.uniform(LO + 0.03, HI - 0.03, (n, 2, 2))
    for i in range(n):
        for k in range(2):
            if rng.random() < 0.5:                        # half of the targets lie next to a joint limit
                j = rng.integers(0, 2); tgt[i, k, j] = LO[j] + 0.03 if rng.random() < 0.5 else HI[j] - 0.03
    return th, q0, tgt

def authority(vbar_q, n=3000, seed=3):
    # drift-scale margin mu over q in C_q, |dq_i| <= vbar_q, all payloads and frictions of the task family
    rng = np.random.default_rng(seed)
    th = theta_np(rng.uniform(0, PMAX, n), rng.uniform(0.2, 1, n), rng.uniform(0.2, 1, n))
    q = rng.uniform(LO, HI, (n, 2)); dq = rng.uniform(-vbar_q, vbar_q, (n, 2))
    M = mass_np(q, th); c, s = np.cos(q[:, 1]), np.sin(q[:, 1]); h = th[:, 1] * s
    nv = np.stack([-h * dq[:, 1] * dq[:, 0] - h * (dq[:, 0] + dq[:, 1]) * dq[:, 1] + th[:, 3] * np.cos(q[:, 0]) + th[:, 4] * np.cos(q[:, 0] + q[:, 1]) + th[:, 5] * dq[:, 0],
                   h * dq[:, 0] ** 2 + th[:, 4] * np.cos(q[:, 0] + q[:, 1]) + th[:, 6] * dq[:, 1]], 1)
    td = nv - K1 * np.einsum('nij,nj->ni', M, dq); A = np.linalg.inv(M); b = np.einsum('nij,nj->ni', A, td)
    g = np.linspace(-1, 1, 401); V = []
    for i in range(2):
        for sg in (1.0, -1.0):
            v = np.zeros((401, 2)); v[:, i] = sg * UBAR[i]; v[:, 1 - i] = g * UBAR[1 - i]; V.append(v)
    V = np.concatenate(V)
    num = np.einsum('ki,nij,kj->nk', V, A, V); den = -np.einsum('ni,ki->nk', b, V)
    mu = np.min(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.inf), axis=1)
    return dict(vbar_q=vbar_q, mu_min=float(mu.min()), mu_q01=float(np.quantile(mu, 0.01)), mu_med=float(np.median(mu)),
                frac_mu_gt_1=float(np.mean(mu > 1)), lam_min_M=float(np.linalg.eigvalsh(M)[:, 0].min()))

if __name__ == '__main__':
    arch = sys.argv[1]
    if arch == 'auth':
        vq = VBAR / K1 + PHI + PHI                         # velocity bound of the mechanical reduction
        res = [authority(vq), authority(1.5 * vq)]
        print(json.dumps(res)); json.dump(res, open(f'../records/{PREFIX}authority.json', 'w')); sys.exit(0)
    seed, iters = int(sys.argv[2]), int(sys.argv[3])
    episode, init, ND = setup(arch)
    train = jit(vmap(vmap(episode, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None)))
    evalb = jit(vmap(episode, in_axes=(None, 0, 0, 0)))
    test = sample_tasks(np.random.default_rng(777), N_TEST)
    rng = np.random.default_rng(1000 + seed)
    theta = init(seed); m = np.zeros(ND); v2 = np.zeros(ND)
    log = open(f'../records/{PREFIX}{arch}_s{seed}.jsonl', 'w')
    n_train_ep = n_viol_ep = 0
    t_start = time.time()
    for it in range(iters + 1):
        if it % 10 == 0 or it == iters:
            ev = {k: np.asarray(v) for k, v in evalb(jnp.asarray(theta), *map(jnp.asarray, test)).items()}
            rec = dict(it=it, kind='test', ret=float(ev['ret'].mean()), err=float(ev['err'].mean()), ss=float(ev['ss'].mean()), ss_all=[float(v) for v in ev['ss']], viol_eps=int((ev['viol'] > 1e-9).sum()),
                       viol_max=float(ev['viol'].max()), wmax=float(ev['wmax'].max()), lam=float(ev['lam'].mean()), y1=float(ev['y1'].mean()),
                       eff=float(ev['eff'].mean()), sat=float(ev['sat'].mean()), flag=float(ev['flag'].mean()),
                       train_eps=n_train_ep, train_viol_eps=n_viol_ep, time=time.time() - t_start)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(json.dumps(rec), flush=True)
        if it == iters:
            break
        tasks = sample_tasks(rng, K_TASK)
        eps = rng.normal(size=(P_POP // 2, ND)); E = np.concatenate([eps, -eps])
        res = {k: np.asarray(v) for k, v in train(jnp.asarray(theta[None] + SIGMA * E), *map(jnp.asarray, tasks)).items()}
        F = res['ret'].mean(1)
        rk = np.empty(P_POP); rk[np.argsort(F)] = np.arange(P_POP); rk = rk / (P_POP - 1) - 0.5
        g = (rk[:P_POP // 2] - rk[P_POP // 2:]) @ eps / (P_POP // 2 * SIGMA)
        m = 0.9 * m + 0.1 * g; v2 = 0.999 * v2 + 0.001 * g ** 2
        theta = theta + LR * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v2 / (1 - 0.999 ** (it + 1))) + 1e-8) - 0.005 * LR * theta
        n_train_ep += res['viol'].size; n_viol_ep += int((res['viol'] > 1e-9).sum())
        log.write(json.dumps(dict(it=it, kind='train', ret=float(F.mean()), viol_eps=int((res['viol'] > 1e-9).sum()),
                                  viol_max=float(res['viol'].max()), wmax=float(res['wmax'].max()))) + '\n'); log.flush()
    np.save(f'../records/{PREFIX}{arch}_s{seed}_theta.npy', theta)
