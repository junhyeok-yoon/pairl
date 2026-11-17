"""Gripper behaviour of a checkpoint while holding (diagnostic of 2026-11-17, Part B0 of the attempt-4 request; no rule depends on it).
Evaluation set and keys of eval/diagnostics/phase_success.py (256 episodes per start, PRNG key 555), CPU.
- open probability while holding: at every valid decision with the object attached (state at the decision instant, before the gripper event), the policy's
  probability of opening, 1 - sigmoid(logit), recomputed from the recorded observation; per start and per phase: mean, median, share of decisions above
  0.5 (where the deterministic gripper opens), and the mean over episodes of the probability that a gripper sampled at those decisions keeps holding
  through the phase (product of sigmoid(logit) over the episode's holding decisions of the phase, on the deterministic trajectory);
- phase-4 start (env.restart kind 4): share released by the end of phase 4 (not attached at the last decision of phase 4), share whose tip is within 2 cm
  of p* at that decision, their joint table, and P4 of phase_success.py; deterministic and gripper-sampled (l2.rollout mode 2).
Usage: python script/parallel_run.py workers=N eval/diagnostics/l2_gripper_hold.py OUT_JSON RUN_DIR:IT"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_success as PSm
from pairl.learn import l2
from pairl.stageb import env

OUT, SPEC = sys.argv[1], sys.argv[2]; run, it = SPEC.rsplit(':', 1)
ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp'])
plogit = jax.jit(jax.vmap(jax.vmap(lambda x: l2.policy(bp, x)[2])))
out = dict(spec=SPEC, open_prob_holding={}, phase4_start={})
for start in (0, 2, 3, 4):
    eps = PSm.ev if start == 0 else [env.restart(p, t, start) for p, t in PSm.ev]; pb, tb = env.stack(eps)
    st, m, dec, rec = l2.make_batch_fn(PSm.cfg)(pb, tb, PSm.keys, jax.tree_util.tree_map(jnp.asarray, ck['bp']), jax.tree_util.tree_map(jnp.asarray, ck['hp']), 1.0, 0.0)
    lg = np.asarray(plogit(jnp.asarray(dec['x']))); keep = 1 / (1 + np.exp(-lg)); po = 1 - keep
    v, j, xi = np.asarray(dec['valid']) > 0.5, np.asarray(dec['j']), np.asarray(dec['viz']['xi']) > 0.5
    o = {}
    for ph in range(4):
        h = v & xi & (j == ph)
        if not h.any(): o[f'phase{ph + 1}'] = dict(n_decisions=0); continue
        ep = h.any(1); surv = np.where(h, keep, 1.0).prod(1)[ep]
        o[f'phase{ph + 1}'] = dict(n_decisions=int(h.sum()), n_episodes=int(ep.sum()), mean=float(po[h].mean()), median=float(np.median(po[h])),
                                   share_above_half=float((po[h] > 0.5).mean()), keep_through_phase_sampled=float(surv.mean()))
    out['open_prob_holding'][f'start{start}'] = o
for name, mode in (('deterministic', 1.0), ('gripper_sampled', 2.0)):
    tr = PSm.traces(SPEC, 4, mode); M = PSm.metrics(tr)
    w4 = tr['v'] & (tr['j'] == 3); last = np.array([np.where(w4[e])[0][-1] for e in range(PSm.N)])
    rel = ~tr['xi'][np.arange(PSm.N), last]; near = np.linalg.norm(tr['p'][np.arange(PSm.N), last] - tr['ps'], axis=-1) <= 0.02
    out['phase4_start'][name] = dict(released=float(rel.mean()), within_2cm=float(near.mean()), released_and_within=float((rel & near).mean()),
                                     released_not_within=float((rel & ~near).mean()), held_and_within=float((~rel & near).mean()), held_not_within=float((~rel & ~near).mean()),
                                     P4=float(M['p4'].mean()))
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out, indent=1))
