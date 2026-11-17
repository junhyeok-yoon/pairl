"""Decision evaluation of pilot O (records/l2_pilotO_plan.md; also pilots L and G and attempt 4): 256 evaluation episodes per start (original, phase-2 restart, phase-3 restart), the same
episodes and keys for every checkpoint, deterministic (decision) and fully sampled (secondary, mode 0: the whole action sampled); per-episode costs,
accurate presses from the phase-3 start, hold error from the phase-2 start, the hand's median transfer residual on the 256 original-start episodes.
Usage: python eval/diagnostics/l2_pilotO_eval.py OUT_JSON RUN_DIR:IT [RUN_DIR:IT ...]
       python eval/diagnostics/l2_pilotO_eval.py compare EVAL_JSON ARM_KEY C_KEY     (paired bootstrap of the deterministic costs, arm minus C)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, time, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P
from pairl import paths

N, KEY, N_BOOT = 256, 555, 10000
STARTS = ((0, 'start0'), (2, 'start2'), (3, 'start3'))


def held_at(eps, dec, phase):
    v = np.asarray(dec['valid']) > 0.5; j = np.asarray(dec['j']); xi = np.asarray(dec['viz']['xi']); ar = np.arange(len(eps))
    return xi[ar, np.argmax(v & (j == phase), 1)] > 0.5


def evaluate(ck_path):
    ck = pickle.load(open(ck_path, 'rb')); l2.set_mode(ck); bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    env.COST_V = 4; cfg = C.default_cfg(); batch = l2.make_batch_fn(cfg)
    ev = l1.with_E0(l1.episodes(6000, N, lim=0.7), P.E_NOM); keys = jax.random.split(jax.random.PRNGKey(KEY), N); out = dict(checkpoint=str(ck_path), it=int(ck['it']))
    for start, tag in STARTS:
        eps = ev if start == 0 else [env.restart(p, t, start) for p, t in ev]; pb, tb = env.stack(eps)
        for mode, mname in ((1.0, 'deterministic'), (0.0, 'fully_sampled')):
            st, m, dec, rec = batch(pb, tb, keys, bp, hp, mode, 0.0); s = l2.summarize(st, m)
            r = dict(cost=[float(x) for x in s['cost']], mean_cost=float(np.mean(s['cost'])), grasp=float(np.mean(s['grasp'])))
            r.update(grasp_ep=[float(x) for x in s['grasp']], fbar_ep=[float(x) for x in s['fbar']], class_ok_ep=[float(x) for x in s['class_ok']],   # per episode (pilot G)
                     f_light=[float(x) for x in tb['f_light']], f_heavy=[float(x) for x in tb['f_heavy']], heavy=[float(x) for x in tb['heavy']])
            if start == 3:
                h4 = held_at(eps, dec, 3); acc = (np.asarray(s['frel']) < 0.5) & h4
                r.update(accurate_press=float(acc.mean()), held_start4=float(h4.mean()))
            if start == 2:
                h3 = held_at(eps, dec, 2); r.update(held_start3=float(h3.mean()), hold_error_median=float(np.median(np.asarray(s['pos2'])[h3])) if h3.any() else None)
            out.setdefault(tag, {})[mname] = r
            del st, m, dec, rec
    hd = l2.hand_diagnostics(hp, ev); out['hand'] = dict(res_learned_median=float(np.median(hd['res_learned'])), res_class_median=float(np.median(hd['res_class'])))
    return out


def compare(R, arm, c):
    d = np.concatenate([np.asarray(R[arm][t]['deterministic']['cost']) - np.asarray(R[c][t]['deterministic']['cost']) for _, t in STARTS])
    rng = np.random.default_rng(0); bs = rng.choice(d, (N_BOOT, d.size), replace=True).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    dacc = R[arm]['start3']['deterministic']['accurate_press'] - R[c]['start3']['deterministic']['accurate_press']
    return dict(n_pairs=int(d.size), mean_diff=float(d.mean()), ci95=[float(lo), float(hi)], accurate_press_diff=float(dacc),
                rule_cost=bool(hi < 0), rule_press=bool(dacc >= -0.05))


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        R = json.load(open(sys.argv[2])); print(json.dumps(compare(R, sys.argv[3], sys.argv[4]), indent=1)); sys.exit()
    OUT = sys.argv[1]; R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for spec in sys.argv[2:]:
        run, it = spec.rsplit(':', 1); t0 = time.time()
        r = evaluate(os.path.join(run, f'params_it{int(it):04d}.pkl')); r['device'] = str(jax.devices()[0]); r['seconds'] = round(time.time() - t0, 1)
        key = f"{run.rstrip('/').split('/')[-1]}:{it}"; R[key] = r
        print(key, json.dumps({t: {m: {k: v for k, v in x.items() if k != 'cost'} for m, x in r[t].items()} for _, t in STARTS} | dict(hand=r['hand'], seconds=r['seconds'])), flush=True)
        json.dump(R, open(OUT, 'w'), indent=1)
