"""Tests of amendment 1 of attempt 4 (records/l2_attempt4_plan.md): the gripper exploration of arm I applies only to phase-4 decisions. CPU, the
l2_pilotG_C checkpoint at 320, 16 evaluation episodes from the original start and 16 from the phase-4 start, training mode 0, same keys.
(a) with GRIP_EPS 0.05 every decision of phases 1-3 (and everything up to the first phase-4 decision) reproduces the unmodified sampling (GRIP_EPS 0,
    the code path of arm N) bit-identically: gripper draws, actions, observations and recorded log-probabilities;
(b) on phase-4 decisions the recorded log-probability equals logp_mix(0.05) recomputed from the recorded observation, action and gripper, and the share of
    psi = 1 matches the mean of p' = 0.95 sigmoid(logit) + 0.025 (binomial 4-sigma band); on phases 1-3 it equals logp_of;
(c) l2.ppo_loss_i reproduces the recorded log-probability on every decision (approximate KL 0 and no clipped sample before an update), so the PPO
    log-probability and entropy use the distribution actually sampled.
Usage: python eval/verify/attempt4_amend_tests.py   (prints PASS/FAIL per test; exit 1 on a failure)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

ck = pickle.load(open(paths.run_dir('l2_pilotG_C') / 'params_it0320.pkl', 'rb')); l2.set_mode(ck)
bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp')); res = []
def report(name, ok, detail): res.append(ok); print(('PASS' if ok else 'FAIL'), name, '|', detail, flush=True)
env.COST_V = 4; cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 16, lim=0.7), P.E_NOM)
eps = ev + [env.restart(p, t, 4) for p, t in ev]; pb, tb = env.stack(eps); keys = jax.random.split(jax.random.PRNGKey(7), len(eps))
l2.GRIP_EPS = 0.0; _, _, d0, _ = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, 0.0, 0.0)
l2.GRIP_EPS = 0.05; _, _, d1, _ = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, 0.0, 0.0)
A = lambda d, k: np.asarray(d[k]); v = A(d1, 'valid') > 0.5; j = A(d1, 'j')
pre = np.cumsum(j == 3, 1) == 0                                    # decisions before the first phase-4 decision of each episode
ok = all(np.array_equal(A(d0, k)[pre], A(d1, k)[pre]) for k in ('psi', 'u', 'x', 'logp'))
report('(a) phases 1-3 reproduce the unmodified sampling', ok and bool(np.array_equal(A(d0, 'j'), A(d1, 'j'))),
       f'{int((pre & v).sum())} decisions before phase 4 (valid), gripper, action, observation and log-probability bit-identical')
mu, ls, lg = jax.vmap(jax.vmap(lambda x: l2.policy(bp, x)))(jnp.asarray(A(d1, 'x')))
lmix = np.asarray(jax.vmap(jax.vmap(l2.logp_mix, in_axes=(0, 0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, 0, None))(mu, ls, lg, d1['u'], d1['psi'], 0.05))
lpl = np.asarray(jax.vmap(jax.vmap(l2.logp_of))(mu, ls, lg, d1['u'], d1['psi'])); lp = A(d1, 'logp')
p4 = v & (j == 3); p13 = v & (j < 3)
e4, e13 = float(np.max(np.abs(lmix - lp)[p4])), float(np.max(np.abs(lpl - lp)[p13])); e4p = float(np.max(np.abs(lpl - lp)[p4]))
report('(b) phase-4 log-probability = logp_mix(0.05), phases 1-3 = logp_of', e4 < 1e-9 and e13 < 1e-9 and e4p > 1e-3,
       f'phase 4: max |recorded - logp_mix| {e4:.1e} over {int(p4.sum())} decisions (against logp_of {e4p:.3f}); phases 1-3: max |recorded - logp_of| {e13:.1e} over {int(p13.sum())}')
pp = np.asarray(l2.grip_prob(lg, 0.05))[p4]; ps = A(d1, 'psi')[p4]; sd = np.sqrt(np.sum(pp * (1 - pp))) / p4.sum()
report('(b) phase-4 gripper frequency = mean p\'', abs(ps.mean() - pp.mean()) < 4 * sd, f"share psi=1 {ps.mean():.4f}, mean p' {pp.mean():.4f} (4 sd {4 * sd:.4f})")
msk = v.reshape(-1); fl = lambda a: jnp.asarray(np.asarray(a).reshape((-1,) + np.asarray(a).shape[2:])[msk])
b = dict(x=fl(d1['x']), xc=fl(d1['xc']), u=fl(d1['u']), psi=fl(d1['psi']), logp=fl(d1['logp']), ret=fl(d1['v']), adv=jnp.zeros(int(msk.sum())))
_, ax = l2.ppo_loss_i(l2.add_cls_head(bp), b, 0.2, 0.5, 1e-2, 1e-2, 0.05, 0.1)
report('(c) ppo_loss_i uses the sampled distribution', abs(float(ax['kl'])) < 1e-12 and float(ax['clipfrac']) == 0.0,
       f"approximate KL {float(ax['kl']):.1e}, clipped share {float(ax['clipfrac'])} over {int(msk.sum())} decisions ({int(np.asarray(b['x'][:, l2.PH4] > 0.5).sum())} in phase 4)")
print('all passed' if all(res) else 'FAILED'); sys.exit(0 if all(res) else 1)
