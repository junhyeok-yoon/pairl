import jax, jax.numpy as jnp, numpy as np, json, time, sys
from jax import lax
from jax.flatten_util import ravel_pytree

G0, M0, LC, L, I0, UB = 9.81, 1.5, 0.2, 0.4, 0.05, 20.0
QLO, QHI, CEN, HALF = -0.4, 1.0, 0.3, 0.7
K1, KB, RON, RFULL = 2.0, 20.0, 0.8, 0.9
TS, NSUB, NB, TEND = 2e-3, 4, 25, 3.0
NSTEP = int(round(TEND / TS)); WST = int(round(2.0 / TS)); WTR = int(round(1.0 / TS))
KLO, KHI, D0, LQ, LK = 10.0, 200.0, 3.0, 20.0, 10.0
GA, LS, ZB = 2.0, 50.0, 10.0
TTH, TMAX, TTGT, TMIN, DTG, PM, T0 = 0.5, 6.0, 3.0, 1.0, 1.0, 0.5, 20.0
SIG = 0.05
ARMS = {'full': dict(mem=1, ledger=1, calib=1, torque=0, hand_fit='role'),
        'no_memory': dict(mem=0, ledger=1, calib=1, torque=0, hand_fit='role'),
        'no_ledger': dict(mem=1, ledger=0, calib=1, torque=0, hand_fit='role'),
        'no_calibration': dict(mem=1, ledger=1, calib=0, torque=0, hand_fit='role'),
        'single_return': dict(mem=1, ledger=1, calib=1, torque=0, hand_fit='task'),
        'torque_policy': dict(mem=0, ledger=0, calib=0, torque=1, hand_fit='none')}

def init_mlp(key, sizes, last=0.1):
    ps = []
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        sc = (last if i == len(sizes) - 2 else 1.0) / np.sqrt(sizes[i])
        ps.append((jax.random.normal(k, (sizes[i], sizes[i + 1])) * sc, jnp.zeros(sizes[i + 1])))
    return ps

def mlp(ps, x):
    for i, (W, b) in enumerate(ps):
        x = x @ W + b
        if i < len(ps) - 1:
            x = jnp.tanh(x)
    return x

def feats(hp, q):
    h = jnp.tanh(q * hp['W1'][0] + hp['b1'])
    return jnp.concatenate([jnp.tanh(h @ hp['W2'] + hp['b2']), jnp.ones(1)])

def sample_tasks(rng, n):
    return np.stack([rng.uniform(0, 2, n), np.exp(rng.uniform(np.log(500), np.log(5000), n)), rng.uniform(1, 5, n), rng.uniform(0.05, 0.3, n),
                     rng.uniform(-0.5, 0.5, n), rng.uniform(0.5, 0.6, n), rng.uniform(-0.05, 0.05, n)], 1).astype(np.float32), \
           (rng.normal(0, SIG, (n, NSTEP))).astype(np.float32)

def build(arm, key):
    A = ARMS[arm]
    kb, kh = jax.random.split(key)
    bsizes = [4, 32, 32, 1] if A['torque'] else [11, 32, 32, 3]
    bvec, ub = ravel_pytree(init_mlp(kb, bsizes))
    k1, k2 = jax.random.split(kh)
    hp = dict(W1=jax.random.normal(k1, (1, 16)) * 2.0, b1=jnp.linspace(-1.5, 1.5, 16), W2=jax.random.normal(k2, (16, 8)) / 4.0, b2=jnp.zeros(8), abar=jnp.zeros(9))
    hvec, uh = ravel_pytree(hp)

    def rollout(bv, hv, task, noise, ledger_scale):
        bp, hp = ub(bv), uh(hv)
        pay, ke, be, fv, d, qw, br = [task[i] for i in range(7)]
        It = I0 + M0 * LC ** 2 + pay * L ** 2; gk = G0 * (M0 * LC + pay * L)
        contact = lambda q, dq: -jnp.where(q > qw, jnp.maximum(ke * (q - qw) + be * dq, 0.0), 0.0)
        f = lambda q, dq, tau: (dq, (tau + d + contact(q, dq) - fv * dq - gk * jnp.cos(q)) / It)
        def step(c, inp):
            q, dq, z, qd, K, T, ts, taup, alp, qdp, Kp, kap, acc = c
            k, nz = inp
            t = k * TS
            G = feats(hp, q)
            uad = G @ (hp['abar'] + A['mem'] * GA * z)
            mhat = uad - taup - ts
            eta = dq + K1 * q; w = (eta - K1 * CEN) / (K1 * HALF); rho = jnp.abs(w)
            acon = jnp.clip((rho - RON) / (RFULL - RON), 0, 1)
            if A['torque']:
                tauh = UB * jnp.tanh(mlp(bp, jnp.array([t / TEND, q, dq, ts / 3]))[0])
            else:
                led = ledger_scale * A['ledger']
                obs = jnp.array([t / TEND, q, dq, qd, K / 100, ts / 3, led * T / T0, led * mhat / 3, led * kap, led * alp, led * acon])
                a = mlp(bp, obs)
                upd = (k % NB) == 0
                qdp = jnp.where(upd, QLO + (QHI - QLO) * jax.nn.sigmoid(a[0]), qdp)
                Kp = jnp.where(upd, KLO + (KHI - KLO) * jax.nn.sigmoid(a[1]), Kp)
                kap = jnp.where(upd, A['calib'] * jax.nn.sigmoid(a[2]), kap)
                blk = jnp.abs(ts) >= TMAX
                dqd = LQ * (qdp - qd) * TS; dqd = jnp.where(blk & (dqd * jnp.sign(-ts) > 0), 0.0, dqd)
                dK = LK * (Kp - K) * TS; dK = jnp.where(blk & (dK > 0), 0.0, dK)
                qd = qd + dqd; K = K + dK
                v = K * (q - qd); sv = jnp.where(jnp.abs(ts) > TTH, jnp.clip(v, -TMAX, TMAX), v)
                tauh = uad - sv - D0 * dq
            alp = jnp.where(dq * tauh > -PM, jnp.clip((T - TMIN) / DTG, 0, 1), 1.0)
            tb = -acon * KB * rho / jnp.maximum(1 - rho ** 2, 1e-6) * jnp.sign(w)
            tau = jnp.clip(alp * tauh + tb, -UB, UB)
            h = TS / NSUB; x0, x1 = q, dq
            for _ in range(NSUB):
                a1, b1 = f(x0, x1, tau); a2, b2 = f(x0 + 0.5 * h * a1, x1 + 0.5 * h * b1, tau)
                a3, b3 = f(x0 + 0.5 * h * a2, x1 + 0.5 * h * b2, tau); a4, b4 = f(x0 + h * a3, x1 + h * b3, tau)
                x0, x1 = x0 + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4), x1 + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            dqk = x0 - q
            T = T - tau * dqk
            if A['mem']:
                z = jnp.clip(z - feats(hp, 0.5 * (q + x0)) * dqk - (kap * LS * mhat / K * TS) * G, -ZB, ZB)
            tc = contact(x0, x1); tsn = tc + br + nz
            ust = gk * jnp.cos(q) - d
            inw = (k >= WST).astype(jnp.float32)
            first = acc[8]
            first = jnp.where((first < 0) & (tc < 0), t, first)
            pk_imp = jnp.where((first >= 0) & (t <= first + 0.3), jnp.maximum(acc[9], -tc), acc[9])
            mov = (jnp.abs(dq) > 0.1).astype(jnp.float32)
            itr = (k >= WTR).astype(jnp.float32)
            cost = jnp.where(tc < 0, ((-tc - TTGT) / TTGT) ** 2, 1.0 + jnp.maximum(qw - x0, 0.0) / 0.3)
            acc = jnp.array([acc[0] + inw * ((-tc - TTGT) / TTGT) ** 2, acc[1] + inw * (-tc), acc[2] + inw * jnp.abs(uad - ust),
                             acc[3] + inw * (0.0 if A['torque'] else sv), acc[4] + (uad - ust) ** 2 * (1 - A['torque']), acc[5] + kap * mov, acc[6] + kap,
                             jnp.maximum(acc[7], -tc), first, pk_imp, jnp.minimum(acc[10], T), acc[11] + ((x0 < QLO) | (x0 > QHI)).astype(jnp.float32), acc[12] + itr * cost])
            return (x0, x1, z, qd, K, T, tsn, tau, alp, qdp, Kp, kap, acc), None
        c0 = (jnp.float32(0.3), jnp.float32(0.0), jnp.zeros(9), jnp.float32(0.3), jnp.float32(60.0), jnp.float32(T0), jnp.float32(0.0), jnp.float32(0.0),
              jnp.float32(1.0), jnp.float32(0.3), jnp.float32(60.0), jnp.float32(0.0), jnp.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, T0, 0, 0], jnp.float32))
        c, _ = lax.scan(step, c0, (jnp.arange(NSTEP), noise))
        acc = c[-1]; nw = NSTEP - WST
        R = -acc[12] / (NSTEP - WTR) - 0.5 * jnp.maximum(acc[7] - TMAX, 0) / TMAX
        return jnp.array([R, acc[4] / NSTEP, acc[1] / nw, acc[2] / nw, acc[3] / nw, acc[5] / jnp.maximum(acc[6], 1e-6), acc[9], acc[10], acc[11], (acc[8] >= 0).astype(jnp.float32)])

    ev = jax.jit(jax.vmap(jax.vmap(rollout, in_axes=(None, None, 0, 0, None)), in_axes=(0, 0, None, None, None)))
    return bvec, hvec, ev

def ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r / (len(x) - 1) - 0.5

def train(arm, seed, iters, NP=12, NT=6, sb=0.03, sh=0.03, lrb=0.01, lrh=0.03, log=None):
    A = ARMS[arm]
    bvec, hvec, ev = build(arm, jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    th_b, th_h = np.array(bvec), np.array(hvec)
    m = [np.zeros_like(th_b), np.zeros_like(th_h)]; v = [np.zeros_like(th_b), np.zeros_like(th_h)]
    hist = []
    for it in range(iters):
        eb = rng.normal(size=(NP, th_b.size)); eh = rng.normal(size=(NP, th_h.size))
        B = np.concatenate([th_b + sb * eb, th_b - sb * eb]).astype(np.float32); H = np.concatenate([th_h + sh * eh, th_h - sh * eh]).astype(np.float32)
        tasks, noise = sample_tasks(rng, NT)
        out = np.array(ev(B, H, tasks, noise, 1.0))
        Rm, Lm = out[:, :, 0].mean(1), out[:, :, 1].mean(1)
        wb = ranks(Rm); gb = (wb[:NP] - wb[NP:]) @ eb / (2 * NP * sb)
        grads = [gb]
        if A['hand_fit'] != 'none':
            wh = ranks(-Lm if A['hand_fit'] == 'role' else Rm); grads.append((wh[:NP] - wh[NP:]) @ eh / (2 * NP * sh))
        for j, (th, g, lr) in enumerate(zip((th_b, th_h), grads, (lrb, lrh))):
            m[j] = 0.9 * m[j] + 0.1 * g; v[j] = 0.999 * v[j] + 0.001 * g * g
            th += lr * (m[j] / (1 - 0.9 ** (it + 1))) / (np.sqrt(v[j] / (1 - 0.999 ** (it + 1))) + 1e-8)
        hist.append([float(Rm.mean()), float(Lm.mean())])
        if log and it % 10 == 0:
            print(arm, seed, it, 'R %.4f L %.3f' % (Rm.mean(), Lm.mean()), file=log, flush=True)
    return th_b, th_h, ev, hist

def evaluate(th_b, th_h, ev, n=64, seed=999, ledger_scale=1.0):
    rng = np.random.default_rng(seed)
    tasks, noise = sample_tasks(rng, n)
    out = np.array(ev(th_b[None].astype(np.float32), th_h[None].astype(np.float32), tasks, noise, ledger_scale))[0]
    return tasks, out

if __name__ == '__main__':
    if sys.argv[1] == 'time':
        for arm in ('full', 'torque_policy'):
            t0 = time.time(); bvec, hvec, ev = build(arm, jax.random.PRNGKey(0)); rng = np.random.default_rng(0)
            tasks, noise = sample_tasks(rng, 6)
            B = np.repeat(np.array(bvec)[None], 24, 0); H = np.repeat(np.array(hvec)[None], 24, 0)
            out = ev(B, H, tasks, noise, 1.0); out.block_until_ready(); t1 = time.time()
            out = ev(B, H, tasks, noise, 1.0); out.block_until_ready(); t2 = time.time()
            print(arm, 'compile+run %.1f s, run %.2f s per iteration (24 members x 6 tasks)' % (t1 - t0, t2 - t1), 'R', float(out[:, :, 0].mean()), 'L', float(out[:, :, 1].mean()))
    else:
        arm, seed, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        tag = sys.argv[5] if len(sys.argv) > 5 else 'm2'
        th_b, th_h, ev, hist = train(arm, seed, iters, log=sys.stdout, **kw)
        tasks, out = evaluate(th_b, th_h, ev)
        res = dict(arm=arm, seed=seed, iters=iters, kw=kw, hist=hist, eval=out.tolist(), tasks=tasks.tolist())
        if arm != 'torque_policy':
            _, out0 = evaluate(th_b, th_h, ev, ledger_scale=0.0); res['eval_ledger_zeroed'] = out0.tolist()
        json.dump(res, open(f'../records/{tag}_{arm}_{seed}.json', 'w'))
        print('done', arm, seed)
