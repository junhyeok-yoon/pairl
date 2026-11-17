"""Offline test of the transfer term L_tr of the hand's role loss (records/l2_hand_ltr_plan.md). Fresh data from a fixed brain checkpoint and its hand
(framework of l2_hand_fresh.py); two copies of the start hand: 'role' (role loss of training) and 'role+tr' (role loss + 1.0 L_tr).
Usage: python eval/diagnostics/l2_hand_ltr.py OUT_JSON CHECKPOINT_PKL [iters=30]   (best through script/parallel_run.py)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, time, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
from pairl.learn.l2_hand_contact_core import loss_variant

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT, CK = POS[0], POS[1]; NIT = int(OPT.get('iters', 30)); LAM, NPAIR = 1.0, 64
env.COST_V = 4; cfg = C.default_cfg(); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
tm = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
ck = pickle.load(open(CK, 'rb')); l2.set_mode(ck); bp, hp0 = tm(ck['bp']), tm(ck['hp']); ah0 = tm(ck['opt']['hand']) if 'opt' in ck else l2.adam_init(hp0)
ev_nom = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); res = lambda hp: float(np.median(l2.hand_diagnostics(hp, ev_nom)['res_learned']))


def l_tr(hp, rec, key):
    """Mean over episodes of the mean squared transfer error over sampled step pairs with the same attachment state."""
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec); n = flat['val'].shape[1]
    ab = l2.abar_of(hp)
    def zstar(G, us):
        GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
        return G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), us - G.T @ ab)
    def one(r, k):
        ka, kb = jax.random.split(k); ia = jax.random.randint(ka, (NPAIR,), 0, n); ib = jax.random.randint(kb, (NPAIR,), 0, n)
        keep = r['val'][ia] * r['val'][ib] * (jnp.abs(r['xi'][ia] - r['xi'][ib]) < 0.5)
        def pair(a, b):
            Ga, Gb = l2.hand_G(hp, r['q'][a]), l2.hand_G(hp, r['q'][b])
            return jnp.sum((Gb.T @ (ab + zstar(Ga, r['ustar'][a])) - r['ustar'][b]) ** 2)
        e = jax.vmap(pair)(ia, ib)
        return jnp.sum(keep * e) / jnp.maximum(jnp.sum(keep), 1.0), jnp.sum(keep)
    v, kept = jax.vmap(one)(flat, jax.random.split(key, flat['val'].shape[0]))
    return jnp.mean(v), jnp.sum(kept)


role_fn = lambda hp, rec: loss_variant(hp, rec, 'all', cfgj)
def total(hp, rec, key):
    lr_, ltr = role_fn(hp, rec), l_tr(hp, rec, key)[0]
    return lr_ + LAM * ltr, (lr_, ltr)
g_role = jax.jit(jax.value_and_grad(role_fn)); g_tot = jax.jit(jax.value_and_grad(total, has_aux=True)); f_ltr = jax.jit(l_tr)

pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(paths.POOLS / 'l2_pool_E0_6144.json'))); batch = l2.make_batch_fn(cfg)
ARMS = ('role', 'role+tr')
st = {m: dict(hp=hp0, ah=ah0, rng=np.random.default_rng(7), curve=[(0, res(hp0))], role=[], ltr=[]) for m in ARMS}; t0 = time.time(); nstep = 0; kept = []
for it in range(1, NIT + 1):
    rng = np.random.default_rng(1000 + it); idx = rng.choice(len(pool), 128, replace=False)
    eps = [pool[i] for i in idx[:64]] + [env.restart(*pool[i], 2) for i in idx[64:96]] + [env.restart(*pool[i], 3) for i in idx[96:]]
    pb, tb = env.stack(eps); _, _, _, rec = batch(pb, tb, jax.random.split(jax.random.PRNGKey(5000 + it), 128), bp, hp0, 0.0, 0.0)
    rec = {k: rec[k] for k in ('q', 'ustar', 'xi', 'val', 'zeta', 'taup', 'te')}; rec['cf'] = jnp.zeros_like(rec['val'])
    if it == 1:
        s8 = jax.tree_util.tree_map(lambda a: a[:8], rec); chk = (float(l2.replay_loss(hp0, {k: v for k, v in s8.items() if k != 'cf'}, cfgj, 1)), float(role_fn(hp0, s8)))
        print('check: replay_loss (role weighting) and variant all', chk, flush=True); assert abs(chk[0] - chk[1]) < 1e-9 * max(1.0, abs(chk[0]))
    for _ in range(16):
        nstep += 1; key = jax.random.PRNGKey(9000 + nstep)
        for m in ARMS:
            s = st[m]; sub = s['rng'].choice(128, 32, replace=False); mb = jax.tree_util.tree_map(lambda a: a[sub], rec)
            if m == 'role':
                l, g = g_role(s['hp'], mb); lt, nk = f_ltr(s['hp'], mb, key); s['role'].append(float(l)); s['ltr'].append(float(lt))
            else:
                (l, (lr_, lt)), g = g_tot(s['hp'], mb, key); s['role'].append(float(lr_)); s['ltr'].append(float(lt)); kept.append(float(f_ltr(s['hp'], mb, key)[1]))
            s['hp'], s['ah'] = l2.adam_step(s['hp'], g, s['ah'], 1e-3, clip=1.0)
    if it % 5 == 0:
        for m in ARMS: st[m]['curve'].append((it, res(st[m]['hp'])))
        print(it, ' | '.join(f"{m}: res {st[m]['curve'][-1][1]:.3f} role {np.mean(st[m]['role'][-80:]):.3f} ltr {np.mean(st[m]['ltr'][-80:]):.3f}" for m in ARMS), f'{time.time() - t0:.0f} s', flush=True)
key_out = f"{os.path.basename(os.path.dirname(CK))}:{int(ck['it'])}"
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
A = {m: dict(res_curve=st[m]['curve'], res_mean_5_30=float(np.mean([v for i, v in st[m]['curve'] if i >= 5])), role_last80=float(np.mean(st[m]['role'][-80:])),
             ltr_last80=float(np.mean(st[m]['ltr'][-80:])), role_curve=[float(np.mean(st[m]['role'][i:i + 16])) for i in range(0, len(st[m]['role']), 16)],
             ltr_curve=[float(np.mean(st[m]['ltr'][i:i + 16])) for i in range(0, len(st[m]['ltr']), 16)]) for m in ARMS}
a, b = A['role+tr'], A['role']
R[key_out] = dict(checkpoint=CK, iters=NIT, lam=LAM, pairs_per_episode=NPAIR, kept_pairs_per_minibatch_mean=float(np.mean(kept)), arms=A, device=str(jax.devices()[0]),
                  rule_a=bool(a['res_mean_5_30'] < 1.768 and a['res_mean_5_30'] < b['res_mean_5_30']), rule_b=bool(a['role_last80'] <= 1.10 * b['role_last80']),
                  seconds=round(time.time() - t0))
R[key_out]['pass'] = R[key_out]['rule_a'] and R[key_out]['rule_b']
json.dump(R, open(OUT, 'w'), indent=1); print('RESULT', key_out, json.dumps({k: v for k, v in R[key_out].items() if k != 'arms'}))
