"""Rule (Ld)(ii) of attempt 4 (records/l2_attempt4_plan.md): arm I's final policy from the phase-3 start, deterministic, the decision-evaluation episodes
and keys (l2_pilotO_eval.py: 256 episodes, key 555, revision-15 cost), with the ledger load visible (blind 0) and masked (blind 1: the brain sees u_load = 0,
hence a zero tip load; the hand is unchanged). Both on the same platform in one call (GPU with the deterministic flag, like the decision evaluation).
Per episode: fbar, class_ok, targets, heavy. The unmasked entry must equal the decision evaluation's start3 deterministic entry (checked by the decision script).
Usage: python eval/diagnostics/l2_attempt4_masked.py OUT_JSON RUN_DIR:IT"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

OUT, SPEC = sys.argv[1:3]; run, it = SPEC.rsplit(':', 1)
ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck); bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
env.COST_V = 4; cfg = C.default_cfg(); batch = l2.make_batch_fn(cfg)
ev = l1.with_E0(l1.episodes(6000, 256, lim=0.7), P.E_NOM); keys = jax.random.split(jax.random.PRNGKey(555), 256)
eps = [env.restart(p, t, 3) for p, t in ev]; pb, tb = env.stack(eps); out = dict(spec=SPEC, device=str(jax.devices()[0]))
for name, blind in (('visible', 0.0), ('masked', 1.0)):
    st, m, dec, rec = batch(pb, tb, keys, bp, hp, 1.0, blind); s = l2.summarize(st, m)
    out[name] = dict(cost=[float(x) for x in s['cost']], fbar_ep=[float(x) for x in s['fbar']], class_ok_ep=[float(x) for x in s['class_ok']],
                     f_light=[float(x) for x in tb['f_light']], f_heavy=[float(x) for x in tb['f_heavy']], heavy=[float(x) for x in tb['heavy']])
    c = np.asarray(s['fbar']) > 0; print(name, 'top-face episodes', int(c.sum()), 'class decision', float(np.asarray(s['class_ok'])[c].mean()), flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
