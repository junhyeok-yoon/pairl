"""Offline hand experiment (analysis, no change to any run): do contact steps in the replay pull the hand's features away from transfer? Rollouts of a fixed
brain checkpoint with its hand, from the training start distribution (restart share 0.5), are stored; the hand of a start checkpoint (with its optimizer
state) is then trained on them as in l2_train (16 Adam steps of 32 episodes per iteration, lr 1e-3, clip 1) with the role loss in three variants:
all steps (as trained); contact steps masked out; contact onset and offset reset the rest time of the weighting (as a load change). Contact: measured external
torque minus the task bias above tau_th. The median transfer residual on the 64 evaluation episodes is tracked.
Usage: python3 l2_hand_contact.py OUT_JSON RUN BRAIN_IT HAND_IT [n_batches=6] [iters=60] [modes=all,mask,reset]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, time, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT, RUN, BIT, HIT = POS[0], POS[1], int(POS[2]), int(POS[3]); NBD, NIT = int(OPT.get('n_batches', 6)), int(OPT.get('iters', 60)); MODES = OPT.get('modes', 'all,mask,reset').split(',')
env.COST_V = 4; cfg = C.default_cfg(); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
ld = lambda i: pickle.load(open(f'{RUN}/params_it{i:04d}.pkl', 'rb'))
ckb, ckh = ld(BIT), ld(HIT); l2.set_mode(ckb); tm = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
bp, hpb = tm(ckb['bp']), tm(ckb['hp']); hp0 = tm(ckh['hp']); ah0 = tm(ckh['opt']['hand']) if 'opt' in ckh else l2.adam_init(hp0)
ev_nom = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM)
res = lambda hp: float(np.median(l2.hand_diagnostics(hp, ev_nom)['res_learned']))

def loss_v(hp, rec, mode):
    def one(r):
        def step(c, x):
            z, rr, xp, cp = c
            chg = jnp.abs(x['xi'] - xp) > 0.5
            if mode == 'reset': chg = chg | (jnp.abs(x['cf'] - cp) > 0.5)
            rr = jnp.where(chg, 0.0, rr)
            G = l2.hand_G(hp, x['q']); uad = G.T @ (l2.abar_of(hp) + z); w = 1.0 - jnp.exp(-cfgj['lam_s'] * rr)
            v = x['val'] * (1.0 - x['cf']) if mode == 'mask' else x['val']
            e = v * w * jnp.sum((uad - x['ustar']) ** 2)
            mhat = uad - x['taup'] - x['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
            dz = G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mhat)
            return (jnp.clip(z - P.TS * cfgj['lam_s'] * x['zeta'] * dz, -cfgj['zbar'], cfgj['zbar']), rr + P.TS * x['zeta'], x['xi'], x['cf']), e
        _, e = jax.lax.scan(jax.checkpoint(step), (jnp.zeros(l2.NA), jnp.asarray(1e9), jnp.zeros(()), jnp.zeros(())), r)
        return jnp.sum(e) / jnp.maximum(jnp.sum(r['val'] * (1.0 - r['cf']) if mode == 'mask' else r['val']), 1.0)
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec)
    return jnp.mean(jax.vmap(one)(flat))
grads = {m: jax.jit(jax.value_and_grad(lambda hp, rec, m=m: loss_v(hp, rec, m))) for m in ('all', 'mask', 'reset')}

pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json'))))
batch = l2.make_batch_fn(cfg); rng = np.random.default_rng(31); DATA = []; t0 = time.time(); shares = []
for b in range(NBD):
    idx = rng.choice(len(pool), 128, replace=False); eps = [pool[i] for i in idx[:64]] + [env.restart(*pool[i], 2) for i in idx[64:96]] + [env.restart(*pool[i], 3) for i in idx[96:]]
    pb, tb = env.stack(eps); _, _, _, rec = batch(pb, tb, jax.random.split(jax.random.PRNGKey(5000 + b), 128), bp, hpb, 0.0, 0.0)
    rec = {k: np.asarray(v) for k, v in rec.items()}
    bias = np.asarray(tb['bias']).reshape((128,) + (1,) * (rec['te'].ndim - 2) + (3,))
    rec['cf'] = (np.linalg.norm(rec['te'] - bias, axis=-1) > cfg['tau_th']).astype(float)
    shares.append(float((rec['cf'] * rec['val']).sum() / rec['val'].sum())); DATA.append(rec)
print(f'data: {NBD} batches in {time.time() - t0:.0f} s, contact share of valid steps {np.mean(shares):.4f}', flush=True)
sub0 = jax.tree_util.tree_map(lambda a: jnp.asarray(a[:8]), {k: v for k, v in DATA[0].items() if k != 'cf'})
chk = (float(l2.replay_loss(hp0, sub0, cfgj, 1)), float(loss_v(hp0, dict(sub0, cf=jnp.asarray(DATA[0]['cf'][:8])), 'all')))
print('check: replay_loss (role weighting) and variant all', chk, flush=True); assert abs(chk[0] - chk[1]) < 1e-9 * max(1.0, abs(chk[0]))

R = dict(run=RUN.rstrip('/').split('/')[-1], brain_it=BIT, hand_it=HIT, n_batches=NBD, iters=NIT, contact_share=float(np.mean(shares)), check=chk, res_start=res(hp0), arms={})
for mode in MODES:
    hp, ah = hp0, ah0; r = np.random.default_rng(7); curve = [(0, R['res_start'])]; ls = []; t0 = time.time()
    for it in range(1, NIT + 1):
        D = DATA[(it - 1) % NBD]
        for _ in range(16):
            sub = r.choice(128, 32, replace=False); l, g = grads[mode](hp, jax.tree_util.tree_map(lambda a: jnp.asarray(a[sub]), D))
            hp, ah = l2.adam_step(hp, g, ah, 1e-3, clip=1.0); ls.append(float(l))
        if it % 10 == 0: curve.append((it, res(hp))); print(mode, it, f'res {curve[-1][1]:.3f} loss {np.mean(ls[-160:]):.3f} {time.time() - t0:.0f} s', flush=True)
    R['arms'][mode] = dict(res_curve=curve, loss_last=float(np.mean(ls[-160:])))
json.dump(R, open(OUT, 'w'), indent=1); print(json.dumps({k: v for k, v in R.items() if k != 'arms'}), {m: a['res_curve'][-1] for m, a in R['arms'].items()})
