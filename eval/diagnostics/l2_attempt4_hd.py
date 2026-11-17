"""Rule (Hd) of attempt 4 (records/l2_attempt4_plan.md): each arm's final hand paired with the same brain (arm N's final brain), same evaluation episodes
and keys, CPU (through script/parallel_run.py).
- transfer residual: l2.hand_diagnostics res_learned, median over the 256 original-start episodes (as the 'hand' entry of l2_pilotO_eval.py; brain-free);
- mis_hold: per episode from the phase-2 start (l2.summarize mis_hold: |u_ad - u*| over the last 40% of phase 2), deterministic, N's brain with each hand;
- phase success (eval/diagnostics/phase_success.py, deterministic) of N's brain with N's hand and with I's hand (the hybrid checkpoint is written to
  _local/tmp and removed afterwards); per-episode results are kept so that eval/diagnostics/phase_diff.diff applies.
Usage: python script/parallel_run.py workers=N eval/diagnostics/l2_attempt4_hd.py OUT_JSON N_RUN_DIR:IT I_RUN_DIR:IT"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, shutil, numpy as np, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_success as PSm
from pairl import paths
from pairl.learn import l2
from pairl.stageb import env

OUT, SN, SI = sys.argv[1:4]
load = lambda spec: pickle.load(open(os.path.join(spec.rsplit(':', 1)[0], f"params_it{int(spec.rsplit(':', 1)[1]):04d}.pkl"), 'rb'))
ckN, ckI = load(SN), load(SI); IT = int(SN.rsplit(':', 1)[1])
tmp = paths.TMP / 'a4_hd'; shutil.rmtree(tmp, ignore_errors=True)
specs = {}
for tag, hp in (('N_brain_N_hand', ckN['hp']), ('N_brain_I_hand', ckI['hp'])):
    d = tmp / tag; d.mkdir(parents=True); ck = dict(ckN, hp=hp); ck.pop('opt', None)
    pickle.dump(ck, open(d / f'params_it{IT:04d}.pkl', 'wb')); specs[tag] = f'{d}:{IT}'
out = dict(brain=SN, hands=dict(N=SN, I=SI))
for tag, spec in specs.items():
    ck = load(spec); l2.set_mode(ck); bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    hd = l2.hand_diagnostics(hp, PSm.ev); res = float(np.median(hd['res_learned']))
    eps = [env.restart(p, t, 2) for p, t in PSm.ev]; pb, tb = env.stack(eps)
    st, m, dec, rec = l2.make_batch_fn(PSm.cfg)(pb, tb, PSm.keys, bp, hp, 1.0, 0.0); s = l2.summarize(st, m)
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']); ar = np.arange(len(eps))
    held3 = xi[ar, np.argmax(v & (j == 2), 1)] > 0.5
    ps = PSm.summary({s_: PSm.metrics(PSm.traces(spec, s_, 1.0)) for s_ in (0, 2, 3, 4)})
    out[tag] = dict(residual_median=res, mis_hold_ep=[float(x) for x in s['mis_hold']], mis_hold_median=float(np.median(s['mis_hold'])),
                    held_start3=float(held3.mean()), deterministic=ps)
    print(tag, 'residual', round(res, 4), 'mis_hold median', round(float(np.median(s['mis_hold'])), 4), 'held at phase 3', float(held3.mean()),
          json.dumps({g: {k: v_['rate'] for k, v_ in ps[g].items()} for g in ('alone', 'chain')}), flush=True)
shutil.rmtree(tmp, ignore_errors=True)
json.dump(out, open(OUT, 'w'), indent=1)
