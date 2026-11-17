"""Port check of the Stage B learning code on a new machine (rollouts only, no training).
(1) Reproduction: deterministic rollouts of the 64 evaluation episodes with the checkpoint of attempt 3 at iteration 300, from the original start and from
both restart starts, compared episode by episode with the costs recorded by the evaluation of l2_train.py (records/l2_attempt3/evals.json, iteration 300).
With ref=pc the same rollouts are compared with records/port_check_pc_baseline.json (this PC, bundle code, CPU) instead; EXACT only if all 192
episode costs are bit-identical.
(2) Throughput (bench=1): sampled rollouts of N episodes from the training pool for N in sizes=..., time per batch after compilation.
Usage: python script/port_check.py [ref=cloud|pc] [bench=1] [sizes=64,256,1024,4096] [reps=2]"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, json, pickle, time, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
from pairl import paths

OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a); RUN = str(paths.run_dir('l2_attempt3')) + '/'; REF = OPT.get('ref', 'cloud'); assert REF in ('cloud', 'pc')
env.COST_V = 4; cfg = C.default_cfg(); print('jax', jax.__version__, 'devices', jax.devices(), 'x64', jax.config.jax_enable_x64, flush=True)
ck = pickle.load(open(RUN + 'params_it0300.pkl', 'rb')); l2.set_mode(ck); assert l2.ACT_MODE == 'task'
bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
rec = {e['it']: e for e in json.load(open(RUN + 'evals.json'))}[300]
ev = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); batch = l2.make_batch_fn(cfg); worst = 0.0
PCB = json.load(open(paths.RECORDS / 'port_check_pc_baseline.json'))['starts'] if REF == 'pc' else None; n_bit = 0
for start, tag in ((0, 'nom'), (2, 'r2'), (3, 'r3')):
    eps = ev if start == 0 else [env.restart(p, t, start) for p, t in ev]; pb, tb = env.stack(eps)
    t0 = time.time(); st, m, dec, rr = batch(pb, tb, jax.random.split(jax.random.PRNGKey(555), 64), bp, hp, 1.0, 0.0)
    c = np.asarray(l2.summarize(st, m)['cost']); dt = time.time() - t0
    if REF == 'pc':
        ref = np.asarray([float.fromhex(x) for x in PCB[tag]['cost_hex']]); assert PCB[tag]['start'] == start
        rel = np.abs(c - ref) / np.maximum(np.abs(ref), 1e-9); nb = int((c == ref).sum()); n_bit += nb
        print(f'start {start} vs pc baseline: mean cost {c.mean():.6f} (baseline {ref.mean():.6f}, difference {c.mean() - ref.mean():.3e}), '
              f'relative difference per episode mean {rel.mean():.2e} max {rel.max():.2e}, episodes above 1e-6: {int((rel > 1e-6).sum())}, bit-identical {nb}/{len(c)}, '
              f'time incl. compile {dt:.1f} s', flush=True)
        continue
    ref = np.asarray(rec[tag]['cost'])
    rel = np.abs(c - ref) / np.maximum(np.abs(ref), 1e-9); worst = max(worst, float(rel.max()))
    print(f'start {start}: mean cost {c.mean():.6f} (recorded {ref.mean():.6f}), max relative difference per episode {rel.max():.2e}, '
          f'episodes above 1e-6: {int((rel > 1e-6).sum())}, time incl. compile {dt:.1f} s', flush=True)
if REF == 'pc': print('REPRODUCTION vs pc baseline', 'EXACT (all 192 episode costs bit-identical)' if n_bit == 192 else f'NOT BIT-IDENTICAL ({n_bit}/192 bit-identical)', flush=True)
else: print('REPRODUCTION', 'EXACT (<=1e-6)' if worst <= 1e-6 else ('CLOSE (<=1e-3)' if worst <= 1e-3 else 'DIFFERENT (>1e-3): investigate before any training'), f'worst {worst:.2e}', flush=True)
if OPT.get('bench', '0') == '1':
    pool = l1.with_E0(l1.episodes(10000, 6144, lim=0.7), json.load(open(str(paths.POOLS / 'l2_pool_E0_6144.json')))); rng = np.random.default_rng(0)
    for n in [int(s) for s in OPT.get('sizes', '64,256,1024,4096').split(',')]:
        idx = rng.choice(len(pool), min(n, len(pool)), replace=n > len(pool)); pb, tb = env.stack([pool[i] for i in idx]); ts = []
        for r in range(1 + int(OPT.get('reps', 2))):
            t0 = time.time(); out = batch(pb, tb, jax.random.split(jax.random.PRNGKey(r), n), bp, hp, 0.0, 0.0); jax.block_until_ready(out); ts.append(time.time() - t0)
        dec = out[2]; nd = float(np.asarray(dec['valid']).sum())
        print(f'bench n={n}: first call {ts[0]:.1f} s (compile), then {np.mean(ts[1:]):.2f} s per batch, {nd / np.mean(ts[1:]):.0f} decisions/s, '
              f'{n / np.mean(ts[1:]):.1f} episodes/s', flush=True)
