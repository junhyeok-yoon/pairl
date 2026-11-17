"""Tests of the attempt-4 arm I additions (records/l2_attempt4_plan.md), CPU, on the l2_pilotG_C checkpoint at 320.
(3) class-reading head: the action distribution with the zero-initialized head equals the one without it (exactly); at initialization the class loss gives no
    gradient to the actor (the head's weights are zero), so the PPO gradient of the actor is unchanged; ppo_loss_i with eps 0, equal entropy coefficients and
    w_cls 0 equals ppo_loss (to rounding).
(5) gripper exploration: the sampler draws psi = 1 with frequency grip_prob (binomial 4-sigma band, 10^6 draws per logit); in rollouts of mode 0 with
    GRIP_EPS 0.05 the recorded log-probability equals logp_mix(eps = 0.05) recomputed from the recorded observation, action and gripper (the PPO ratio is 1 before
    the update), and the share of psi = 1 matches the mean grip_prob; in modes 1 and 2 (evaluation) the gripper is the unchanged deterministic / sigmoid one.
Usage: python eval/verify/attempt4_tests.py   (prints PASS/FAIL per test; exit 1 on a failure)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, pickle, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

ck = pickle.load(open(paths.run_dir('l2_pilotG_C') / 'params_it0320.pkl', 'rb')); l2.set_mode(ck)
bp0, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp')); res = []
def report(name, ok, detail): res.append(ok); print(('PASS' if ok else 'FAIL'), name, '|', detail, flush=True)

# minibatch from a sampled rollout (mode 0, plain gripper) of 8 episodes
env.COST_V = 4; cfg = C.default_cfg(); ev = l1.with_E0(l1.episodes(6000, 8, lim=0.7), P.E_NOM); pb, tb = env.stack(ev)
st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(3), 8), bp0, hp, 0.0, 0.0)
msk = np.asarray(dec['valid']).reshape(-1) > 0.5; fl = lambda a: jnp.asarray(np.asarray(a).reshape((-1,) + np.asarray(a).shape[2:])[msk])
adv = np.random.default_rng(0).normal(size=int(msk.sum()))
b = dict(x=fl(dec['x']), xc=fl(dec['xc']), u=fl(dec['u']), psi=fl(dec['psi']), logp=fl(dec['logp']), ret=fl(dec['v']) + 0.1, adv=jnp.asarray(adv))

# (3)
bph = l2.add_cls_head(bp0); X = b['x']
o0 = jax.vmap(lambda x: l2.policy(bp0, x))(X); o1 = jax.vmap(lambda x: l2.policy(bph, x))(X)
report('(3) head leaves the action distribution unchanged', all(bool(jnp.array_equal(a, c)) for a, c in zip(o0, o1)), f'{X.shape[0]} decisions, mean/log_std/logit bit-identical')
(l_a, ax_a), g_a = l2.ppo_grad_i(bph, b, 0.2, 0.5, 1e-2, 1e-2, 0.0, 0.1); (l_b, ax_b), g_b = l2.ppo_grad_i(bph, b, 0.2, 0.5, 1e-2, 1e-2, 0.0, 0.0)
da = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves(g_a['actor']), jax.tree_util.tree_leaves(g_b['actor'])))
gh = float(jnp.max(jnp.abs(g_a['cls']['W'])))
report('(3) class loss gives no actor gradient at initialization', da == 0.0 and gh > 0,
       f'max |actor grad (w 0.1) - (w 0)| = {da:.1e}; head gradient max {gh:.3e} (the head learns); BCE {float(ax_a["cls_bce"]):.6f} = log 2 {np.log(2):.6f}; held decisions {int(ax_a["n_hold"])}')
(l_o, _), g_o = l2.ppo_grad(bp0, b, 0.2, 0.5, 1e-2)
dl = abs(float(l_b) - float(l_o)); dg = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves({k: g_b[k] for k in ('actor', 'critic')}), jax.tree_util.tree_leaves({k: g_o[k] for k in ('actor', 'critic')})))
report('(3) ppo_loss_i reduces to ppo_loss', dl < 1e-12 and dg < 1e-10, f'|loss diff| {dl:.1e}, max |grad diff| {dg:.1e}')

# (5) sampler
for lg in (-6.0, -1.0, 0.0, 2.0, 6.0):
    pp = float(l2.grip_prob(jnp.asarray(lg), 0.05)); u = jax.random.uniform(jax.random.PRNGKey(int(lg * 10) + 100), (1_000_000,))
    f = float(jnp.mean(u < pp)); sd = np.sqrt(pp * (1 - pp) / 1e6)
    report(f'(5) sampler frequency, logit {lg:+.0f}', abs(f - pp) < 4 * sd, f"p' {pp:.6f} frequency {f:.6f} (4 sd {4 * sd:.1e}); p' in [{0.025}, {0.975}]")
# (5) rollouts
l2.GRIP_EPS = 0.05; bf = l2.make_batch_fn(cfg)
st, m, d0, _ = bf(pb, tb, jax.random.split(jax.random.PRNGKey(4), 8), bp0, hp, 0.0, 0.0)
v = np.asarray(d0['valid']) > 0.5; mu, ls, lg = jax.vmap(jax.vmap(lambda x: l2.policy(bp0, x)))(jnp.asarray(d0['x']))
lp_mix = jax.vmap(jax.vmap(l2.logp_mix, in_axes=(0, 0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, 0, None))(mu, ls, lg, d0['u'], d0['psi'], 0.05)
lp_plain = jax.vmap(jax.vmap(l2.logp_of))(mu, ls, lg, d0['u'], d0['psi'])
e = float(np.max(np.abs(np.asarray(lp_mix) - np.asarray(d0['logp']))[v])); e0 = float(np.max(np.abs(np.asarray(lp_plain) - np.asarray(d0['logp']))[v]))
report('(5) rollout log-probability = logp_mix(0.05)', e < 1e-9 and e0 > 1e-3, f'max |recorded - recomputed| {e:.1e} over {int(v.sum())} decisions; against the plain gripper term {e0:.3f}')
ppv = np.asarray(l2.grip_prob(lg, 0.05))[v]; psv = np.asarray(d0['psi'])[v]; sd = np.sqrt(np.sum(ppv * (1 - ppv))) / v.sum()
report('(5) rollout gripper frequency = mean p\'', abs(psv.mean() - ppv.mean()) < 4 * sd, f'share psi=1 {psv.mean():.4f}, mean p\' {ppv.mean():.4f} (4 sd {4 * sd:.4f})')
st, m1a, d1, _ = bf(pb, tb, jax.random.split(jax.random.PRNGKey(4), 8), bp0, hp, 1.0, 0.0)
l2.GRIP_EPS = 0.0; st, m1b, d1o, _ = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(4), 8), bp0, hp, 1.0, 0.0)
report('(5) evaluation mode 1 unchanged', bool(np.array_equal(np.asarray(m1a['cost']), np.asarray(m1b['cost']))) and bool(np.array_equal(np.asarray(d1['psi']), np.asarray(d1o['psi']))) and bool(np.array_equal(np.asarray(d1['u']), np.asarray(d1o['u']))),
       'deterministic rollouts with GRIP_EPS 0.05 and 0: identical gripper, actions and costs')
l2.GRIP_EPS = 0.05; st, m2a, d2, _ = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(4), 8), bp0, hp, 2.0, 0.0)
l2.GRIP_EPS = 0.0; st, m2b, d2o, _ = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(4), 8), bp0, hp, 2.0, 0.0)
report('(5) evaluation mode 2 unchanged', bool(np.array_equal(np.asarray(d2['psi']), np.asarray(d2o['psi']))) and bool(np.array_equal(np.asarray(m2a['cost']), np.asarray(m2b['cost']))),
       'gripper-sampled rollouts with GRIP_EPS 0.05 and 0: identical gripper draws and costs')
print('all passed' if all(res) else 'FAILED'); sys.exit(0 if all(res) else 1)
