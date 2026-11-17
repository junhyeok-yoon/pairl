"""Learning signal of the brain by phase (plan of attempt 2, amendment 6 and its correction): on exploring rollouts of a checkpoint (first 2048 episodes of the
training pool, key 4242, chunks of 256, cost of revision 14), for phase-1 decisions without the object (target: grasp point), held phase-2 decisions (target p*)
and held phase-3 decisions (target p_c): decisions per 128 episodes, and the correlation between the GAE advantage (lambda 0.95 and 1) and the projection of
the goal-tip deviation (sampled minus mean action, through the forward kinematics) on the unit vector from the mean goal tip to the target, with 95% bootstrap
intervals over episodes; held decisions near the start grasp point and near p* (amendment 7);
exploration scale of the goal by class (descriptive). start=2|3: the episodes begin from the restart states of env.restart (pilot R dissection).
Usage: python3 l2_signal_check.py OUT_JSON RUN:IT [...] [n=2048] [start=0|2|3] [cost=4]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, model as Mo

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = POS[0]; REC = json.load(open(OUT)) if os.path.exists(OUT) else {}
env.COST_V = int(OPT.get('cost', 3)); cfg = C.default_cfg(); NEP, CH, NB = int(OPT.get('n', 2048)), 256, 1000; START = int(OPT.get('start', 0))
pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json'))))[:NEP]
if START: pool = [env.restart(p, t, START) for p, t in pool]
keys = jax.random.split(jax.random.PRNGKey(4242), NEP); BF = {}
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0))); jacf = jax.jit(jax.vmap(jax.vmap(Mo.jac, in_axes=(0, None)), in_axes=(0, 0)))
gaef = {lam: jax.jit(jax.vmap(lambda r, v, d, lam=lam: l2.gae(r, v, d, 0.99, lam))) for lam in (0.95, 1.0)}
CLS = ('phase1_free_to_grasp_point', 'phase2_held_to_p_star', 'phase3_held_to_p_c')

def corr(s):
    n, sa, sp, saa, spp, sap = s; ca, cp = saa - sa * sa / n, spp - sp * sp / n
    return (sap - sa * sp / n) / np.sqrt(ca * cp)
def corr_boot(S, rng):
    """Pearson correlation from per-episode sufficient statistics, with a bootstrap over episodes (multinomial weights)."""
    W = rng.multinomial(len(S), np.full(len(S), 1.0 / len(S)), size=NB); bs = np.array([corr(w @ S) for w in W])
    return float(corr(S.sum(0))), [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]

for spec in POS[1:]:
    run, it = spec.rsplit(':', 1)
    ck = pickle.load(open(f'{run}/params_it{int(it):04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp']); l2.set_mode(ck)
    batch = BF.setdefault(l2.ACT_MODE, l2.make_batch_fn(cfg)); gol = jax.jit(jax.vmap(jax.vmap(l2.goal_of)))
    gtip = jax.jit(jax.vmap(jax.vmap(lambda uu, qq, L: jax.jacfwd(lambda a: Mo.tip(l2.goal_of(a, qq), L))(uu)[:, :3], in_axes=(0, 0, None)), in_axes=(0, 0, 0)))
    if 'log_std' in bp: bp = l2.upgrade_v1(bp)
    muf = jax.jit(jax.vmap(jax.vmap(lambda xx: l2.policy(bp, xx)[:2])))
    S = {(c, lam): [] for c in CLS for lam in (0.95, 1.0)}; dist = {c: [] for c in CLS}; pj = {c: [] for c in CLS}; nep = {c: 0 for c in CLS}; cov = np.zeros(2); ex = {c: [] for c in CLS}
    for i0 in range(0, NEP, CH):
        pb, tb = env.stack(pool[i0:i0 + CH])
        st, m, dec, rec = batch(pb, tb, keys[i0:i0 + CH], bp, hp, 0.0, 0.0)
        x, u = dec['x'], dec['u']; mu, ls = muf(x)
        qo = x[..., 0:3] * l2.QH + l2.QC; tip_s = np.asarray(tipf(gol(u, qo), pb['L'])); tip_m = np.asarray(tipf(gol(mu, qo), pb['L']))
        sdu = np.exp(np.asarray(ls[..., :3])); tf = 1 - np.tanh(np.asarray(mu[..., :3])) ** 2                  # descriptive, added after iteration 250: exploration scale
        Gm = np.asarray(gtip(mu, qo, pb['L'])); gsd = np.sqrt(np.sum((Gm * sdu[..., None, :]) ** 2, axis=(-1, -2)) / 2)          # goal-tip sd per axis from the action Jacobian (any mode)
        xn = np.asarray(x); v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); held = xn[..., 21] > 0.5
        o = xn[..., 13:15] * 0.3 + np.asarray(l2.CEN); ps = np.asarray(tb['ps'])[:, None, :] + 0 * o
        pc = np.stack([ps[..., 0], np.asarray(pb['ye'])[:, None] + 0 * ps[..., 0]], -1)
        adv = {lam: np.asarray(f(dec['r'], dec['v'], dec['valid'])) for lam, f in gaef.items()}
        p_meas = xn[..., 19:21] * 0.3 + np.asarray(l2.CEN); hv = v & held                  # amendment 7 (b): held decisions near the start grasp point and near p*
        cov += [np.sum(hv & (np.linalg.norm(p_meas - o[:, :1], axis=-1) < 0.05)), np.sum(hv & (np.linalg.norm(p_meas - ps, axis=-1) < 0.05))]
        for c, mask, tgt in zip(CLS, (v & (j == 0) & ~held, v & (j == 1) & held, v & (j == 2) & held), (o, ps, pc)):
            d = tgt - tip_m; e = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9); p = np.sum((tip_s - tip_m) * e, -1); mk = mask.astype(float)
            dist[c].append(np.linalg.norm(d, axis=-1)[mask]); pj[c].append(p[mask]); nep[c] += int(mask.any(1).sum())
            ex[c].append(np.stack([sdu.mean(-1)[mask], tf.mean(-1)[mask], gsd[mask]], 1))
            for lam in (0.95, 1.0):
                a = adv[lam]; S[(c, lam)].append(np.stack([mk.sum(1), (mk * a).sum(1), (mk * p).sum(1), (mk * a * a).sum(1), (mk * p * p).sum(1), (mk * a * p).sum(1)], 1))
    rng = np.random.default_rng(0); R = dict(episodes_total=NEP, held_near_start_grasp_point_per_128=float(cov[0] * 128 / NEP), held_near_p_star_per_128=float(cov[1] * 128 / NEP))
    for c in CLS:
        dc, pc_ = np.concatenate(dist[c]), np.concatenate(pj[c])
        out = dict(decisions_per_128_episodes=float(len(dc) * 128 / NEP), episodes=nep[c], goal_tip_to_target_median=float(np.median(dc)) if len(dc) else None,
                   deviation_projection_sd=float(pc_.std()) if len(dc) else None)
        if len(dc): xs = np.concatenate(ex[c]); out.update(sd_u_goal_median=float(np.median(xs[:, 0])), tanh_factor_median=float(np.median(xs[:, 1])), goal_tip_sd_median=float(np.median(xs[:, 2])))
        for lam in (0.95, 1.0):
            if nep[c] >= 5: cc, ci = corr_boot(np.concatenate(S[(c, lam)]), rng); out[f'corr_lambda{lam:g}'] = cc; out[f'ci95_lambda{lam:g}'] = ci
            else: out[f'corr_lambda{lam:g}'] = None; out[f'ci95_lambda{lam:g}'] = None
        R[c] = out
    R['cost_version'] = env.COST_V; REC[f"{run.rstrip('/').split('/')[-1]}:{it}" + (f':start{START}' if START else '')] = R; print(spec.split('/')[-1], json.dumps(R), flush=True)
    json.dump(REC, open(OUT, 'w'), indent=1)
