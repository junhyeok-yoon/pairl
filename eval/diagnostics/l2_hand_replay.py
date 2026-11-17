"""Where does the hand's role loss come from (attempt 3, after iteration 200)? On exploring rollouts of a brain checkpoint with its own hand (64 episodes from the
original start and 64 from the phase-3 restart, key 77), the role-weighted squared error of each listed hand in replay, split by contact (measured external
torque minus the task bias above tau_th) and by rest. Usage: python3 l2_hand_replay.py OUT_JSON RUN BRAIN_IT HAND_IT [HAND_IT ...] [n=64]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a); N = int(OPT.get('n', 64))
OUT, RUN, BIT, HITS = POS[0], POS[1], int(POS[2]), [int(a) for a in POS[3:]]
env.COST_V = 4; cfg = C.default_cfg(); cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
ld = lambda i: pickle.load(open(f'{RUN}/params_it{i:04d}.pkl', 'rb')); ck = ld(BIT); l2.set_mode(ck)
bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp0 = jax.tree_util.tree_map(jnp.asarray, ck['hp'])
hps = {i: jax.tree_util.tree_map(jnp.asarray, ld(i)['hp']) for i in HITS}
def per_step(hp, r):
    def step(c, x):
        z, rr, xp = c
        rr = jnp.where(jnp.abs(x['xi'] - xp) > 0.5, 0.0, rr)
        G = l2.hand_G(hp, x['q']); uad = G.T @ (l2.abar_of(hp) + z); w = 1.0 - jnp.exp(-cfgj['lam_s'] * rr)
        e2 = jnp.sum((uad - x['ustar']) ** 2); mhat = uad - x['taup'] - x['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
        dz = G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mhat)
        return (jnp.clip(z - P.TS * cfgj['lam_s'] * x['zeta'] * dz, -cfgj['zbar'], cfgj['zbar']), rr + P.TS * x['zeta'], x['xi']), (w, e2, jnp.linalg.norm(z))
    return jax.lax.scan(step, (jnp.zeros(l2.NA), jnp.asarray(1e9), jnp.zeros(())), r)[1]
ps = {i: jax.jit(jax.vmap(lambda r, hp=hp: per_step(hp, r))) for i, hp in hps.items()}
pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json'))))[:N]
batch = l2.make_batch_fn(cfg); out = {}
for start in (0, 3):
    pl = pool if start == 0 else [env.restart(p, t, 3) for p, t in pool]
    pb, tb = env.stack(pl); st, m, dec, rec = batch(pb, tb, jax.random.split(jax.random.PRNGKey(77), N), bp, hp0, 0.0, 0.0)
    flat = jax.tree_util.tree_map(lambda a: jnp.asarray(a).reshape((a.shape[0], -1) + a.shape[3:]), rec)
    val = np.asarray(flat['val']) > 0.5; zeta = np.asarray(flat['zeta']) > 0.5; xi = np.asarray(flat['xi']) > 0.5
    con = np.linalg.norm(np.asarray(flat['te']) - np.asarray(tb['bias'])[:, None, :], axis=-1) > cfg['tau_th']
    R = dict(steps=int(val.sum()), contact_share_of_steps=float(con[val].mean()), contact_rest_share_of_steps=float((con & zeta)[val].mean()))
    for i, f in ps.items():
        w, e2, zn = (np.asarray(a) for a in f(flat)); we = w * e2 * val; tot = we.sum()
        R[f'hand{i}'] = dict(role_loss=float(tot / val.sum()), share_contact=float(we[con].sum() / tot), share_contact_rest=float(we[con & zeta].sum() / tot),
                             share_held=float(we[xi].sum() / tot), rmse_nocontact_rest=float(np.sqrt(np.mean(e2[val & ~con & zeta]))),
                             rmse_nocontact_motion=float(np.sqrt(np.mean(e2[val & ~con & ~zeta]))),
                             rmse_contact_rest=float(np.sqrt(np.mean(e2[val & con & zeta]))) if np.any(val & con & zeta) else None,
                             rmse_contact_motion=float(np.sqrt(np.mean(e2[val & con & ~zeta]))) if np.any(val & con & ~zeta) else None,
                             zmax_median=float(np.median(zn.max(1))))
    out[f'start{start}'] = R; print(start, json.dumps(R), flush=True)
out.update(run=RUN.rstrip('/').split('/')[-1], brain_it=BIT, hand_its=HITS); json.dump(out, open(OUT, 'w'), indent=1)
