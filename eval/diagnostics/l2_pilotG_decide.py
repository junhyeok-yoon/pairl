"""Decisions of pilot G (records/l2_pilotG_plan.md) from the decision evaluation (records/l2_pilotO_eval.json, revision-15 cost). Paired bootstrap over
episodes (10000 resamples, numpy default_rng(0), percentile 95% CI).
- grasp arms S, S2: grasp at the end of phase 1 from the original start, arm minus C (256 pairs); cost arm minus C (768 pairs);
- S2 against S: grasp S2 minus S;
- ledger arm B, phase-3 start: class decision (class_ok on episodes with top-face force in both arms; paired), class-gap coefficient g of
  fbar ~ a + b f_light + c f_heavy + g h (f_heavy - f_light) on episodes with top-face force (episodes resampled jointly for both arms), cost B minus C.
Usage: python eval/diagnostics/l2_pilotG_decide.py OUT_JSON C_KEY S_KEY S2_KEY B_KEY"""
import sys, json, numpy as np
from pairl import paths

OUT, KC, KS, KS2, KB = sys.argv[1:6]
R = json.load(open(paths.RECORDS / 'l2_pilotO_eval.json')); NB = 10000; ST = ('start0', 'start2', 'start3')
ci = lambda bs: [float(x) for x in np.percentile(bs, [2.5, 97.5])]
det = lambda k, s, f: np.asarray(R[k][s]['deterministic'][f], float)


def paired(d):
    d = np.asarray(d, float); bs = np.random.default_rng(0).choice(d, (NB, d.size)).mean(1); return dict(n=int(d.size), mean=float(d.mean()), ci95=ci(bs))


def cost(a, b): return paired(np.concatenate([det(a, s, 'cost') - det(b, s, 'cost') for s in ST]))
def grasp(a, b): return paired(det(a, 'start0', 'grasp_ep') - det(b, 'start0', 'grasp_ep'))


def gap(k, idx):
    fb, fl, fh, h = (det(k, 'start3', f)[idx] for f in ('fbar_ep', 'f_light', 'f_heavy', 'heavy'))
    X = np.stack([np.ones_like(fl), fl, fh, h * (fh - fl)], 1); return float(np.linalg.lstsq(X, fb, rcond=None)[0][3])


def ledger(a, b):
    ca, cb = det(a, 'start3', 'class_ok_ep'), det(b, 'start3', 'class_ok_ep'); both = (det(a, 'start3', 'fbar_ep') > 0) & (det(b, 'start3', 'fbar_ep') > 0)
    cls = paired(ca[both] - cb[both]); cls.update(arm=float(ca[both].mean()), control=float(cb[both].mean()))
    ia = np.where(det(a, 'start3', 'fbar_ep') > 0)[0]; ib = np.where(det(b, 'start3', 'fbar_ep') > 0)[0]; ga, gb = gap(a, ia), gap(b, ib)
    rng = np.random.default_rng(0); n = len(det(a, 'start3', 'cost')); bs = []
    for _ in range(NB):
        r = rng.integers(0, n, n); bs.append(gap(a, r[np.isin(r, ia)]) - gap(b, r[np.isin(r, ib)]))
    return dict(class_decision=cls, class_gap=dict(arm=ga, control=gb, diff=ga - gb, ci95=ci(bs), n_arm=int(len(ia)), n_control=int(len(ib))))


out = dict(S=dict(grasp=grasp(KS, KC), cost=cost(KS, KC)), S2=dict(grasp=grasp(KS2, KC), cost=cost(KS2, KC)), S2_vs_S=dict(grasp=grasp(KS2, KS)),
           B=dict(**ledger(KB, KC), cost=cost(KB, KC)))
for a in ('S', 'S2'): out[a]['pass'] = bool(out[a]['grasp']['ci95'][0] > 0 and out[a]['cost']['ci95'][1] < 0.5)
out['grasp_choice'] = ('S2' if out['S2']['pass'] and (not out['S']['pass'] or out['S2_vs_S']['grasp']['ci95'][0] > 0) else ('S' if out['S']['pass'] else None))
sys.path.insert(0, str(paths.ROOT / 'eval' / 'diagnostics')); import phase_diff            # amendment 1: phase success, paired with C (CPU, deterministic)
PS = json.load(open(paths.RECORDS / 'phase_success.json'))
out['phase'] = {arm: phase_diff.diff(PS, k, KC) for arm, k in (('S', KS), ('S2', KS2), ('B', KB))}
b = out['B']; b['P3c'], b['P3'] = out['phase']['B']['P3c'], out['phase']['B']['P3']
b['pass'] = bool(b['class_decision']['ci95'][0] > 0 and b['class_gap']['ci95'][0] > 0 and b['cost']['ci95'][1] < 0.5 and b['P3c']['ci95'][0] > 0 and b['P3']['ci95'][0] > 0)
out['keys'] = dict(C=KC, S=KS, S2=KS2, B=KB)
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out, indent=1))
