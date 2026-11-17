"""Offline hand experiment with fresh data (analysis, no change to any run). As l2_hand_contact.py, but every iteration draws a new batch of 128 episodes from
the training start distribution (restart share 0.5) with the fixed brain checkpoint and its hand, the same batch for every arm (pool indices and keys fixed by
the iteration). The hand of the start checkpoint (with its optimizer state) is trained as in l2_train (16 Adam steps of 32 episodes per iteration, lr 1e-3,
clip 1), one copy per role-loss variant: all steps (as trained); contact steps masked out; contact onset and offset reset the rest time of the weighting.
The median transfer residual on the 64 evaluation episodes is tracked. Usage: python3 l2_hand_fresh.py OUT_JSON RUN BRAIN_IT HAND_IT [iters=30] [modes=all,mask]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, time, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
from pairl.learn.l2_hand_contact_core import loss_variant

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT, RUN, BIT, HIT = POS[0], POS[1], int(POS[2]), int(POS[3]); NIT = int(OPT.get('iters', 30)); MODES = OPT.get('modes', 'all,mask').split(',')
env.COST_V = 4; cfg = C.default_cfg(); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
ld = lambda i: pickle.load(open(f'{RUN}/params_it{i:04d}.pkl', 'rb')); tm = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
ckb, ckh = ld(BIT), ld(HIT); l2.set_mode(ckb); bp, hpb = tm(ckb['bp']), tm(ckb['hp']); hp0 = tm(ckh['hp']); ah0 = tm(ckh['opt']['hand']) if 'opt' in ckh else l2.adam_init(hp0)
ev_nom = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); res = lambda hp: float(np.median(l2.hand_diagnostics(hp, ev_nom)['res_learned']))
grads = {m: jax.jit(jax.value_and_grad(lambda hp, rec, m=m: loss_variant(hp, rec, m, cfgj))) for m in MODES}
pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json')))); batch = l2.make_batch_fn(cfg)
st = {m: dict(hp=hp0, ah=ah0, rng=np.random.default_rng(7), curve=[(0, res(hp0))], ls=[]) for m in MODES}; shares = []; t0 = time.time()
for it in range(1, NIT + 1):
    rng = np.random.default_rng(1000 + it); idx = rng.choice(len(pool), 128, replace=False)
    eps = [pool[i] for i in idx[:64]] + [env.restart(*pool[i], 2) for i in idx[64:96]] + [env.restart(*pool[i], 3) for i in idx[96:]]
    pb, tb = env.stack(eps); _, _, _, rec = batch(pb, tb, jax.random.split(jax.random.PRNGKey(5000 + it), 128), bp, hpb, 0.0, 0.0)
    rec = dict(rec); bias = jnp.asarray(tb['bias']).reshape((128,) + (1,) * (rec['te'].ndim - 2) + (3,))
    rec['cf'] = (jnp.linalg.norm(rec['te'] - bias, axis=-1) > cfg['tau_th']).astype(rec['te'].dtype); shares.append(float((rec['cf'] * rec['val']).sum() / rec['val'].sum()))
    if it == 1:
        s8 = jax.tree_util.tree_map(lambda a: a[:8], rec); chk = (float(l2.replay_loss(hp0, {k: v for k, v in s8.items() if k != 'cf'}, cfgj, 1)), float(loss_variant(hp0, s8, 'all', cfgj)))
        print('check: replay_loss (role weighting) and variant all', chk, flush=True); assert abs(chk[0] - chk[1]) < 1e-9 * max(1.0, abs(chk[0]))
    for m in MODES:
        s = st[m]
        for _ in range(16):
            sub = s['rng'].choice(128, 32, replace=False); l, g = grads[m](s['hp'], jax.tree_util.tree_map(lambda a: a[sub], rec))
            s['hp'], s['ah'] = l2.adam_step(s['hp'], g, s['ah'], 1e-3, clip=1.0); s['ls'].append(float(l))
        if it % 5 == 0: s['curve'].append((it, res(s['hp'])))
    if it % 5 == 0: print(it, ' '.join(f"{m} res {st[m]['curve'][-1][1]:.3f} loss {np.mean(st[m]['ls'][-80:]):.3f}" for m in MODES), f'{time.time() - t0:.0f} s', flush=True)
R = dict(run=RUN.rstrip('/').split('/')[-1], brain_it=BIT, hand_it=HIT, iters=NIT, contact_share=float(np.mean(shares)),
         arms={m: dict(res_curve=st[m]['curve'], loss_last=float(np.mean(st[m]['ls'][-80:]))) for m in MODES})
json.dump(R, open(OUT, 'w'), indent=1); print(json.dumps({m: a['res_curve'][-1] for m, a in R['arms'].items()}))
