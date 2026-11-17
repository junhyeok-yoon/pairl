"""Decisions of attempt 4 (records/l2_attempt4_plan.md), all differences I - N, paired bootstrap over episodes (10000 resamples, numpy default_rng(0),
percentile 95% CI):
- guarantees: hist viol, nan, n_sat_below and the minimum tank_bound_min over iterations 321-920 of both runs;
- (T): phase_diff.diff on records/phase_success.json (deterministic); improves = diff >= +0.10 and CI lower bound > 0; non-inferior = diff >= -0.10;
  gripper-sampled differences descriptive;
- (Hd): records/l2_attempt4_hd.json (residual ratio <= 0.5; mis_hold I hand - N hand, CI lower bound <= 0; (T) non-inferior under the pairing);
- (Ld): class decision and class gap g from records/l2_pilotO_eval.json (start3, deterministic), as eval/diagnostics/l2_pilotG_decide.py (function ledger,
  repeated here verbatim); masking from records/l2_attempt4_masked.json (I visible minus I masked, episodes with top-face force in both);
- adoption.
Usage: python eval/diagnostics/l2_attempt4_decide.py OUT_JSON [IT=920]"""
import sys, os, json, numpy as np
from pairl import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import phase_diff

OUT = sys.argv[1]; IT = int(sys.argv[2]) if len(sys.argv) > 2 else 920; KN, KI = f'l2_a4_N:{IT}', f'l2_a4_I:{IT}'
R = json.load(open(paths.RECORDS / 'l2_pilotO_eval.json')); PS = json.load(open(paths.RECORDS / 'phase_success.json')); NB = 10000
ci = lambda bs: [float(x) for x in np.percentile(bs, [2.5, 97.5])]
det = lambda k, s, f: np.asarray(R[k][s]['deterministic'][f], float)
def paired(d):
    d = np.asarray(d, float); bs = np.random.default_rng(0).choice(d, (NB, d.size)).mean(1); return dict(n=int(d.size), mean=float(d.mean()), ci95=ci(bs))
def gap_of(fb, fl, fh, h):
    X = np.stack([np.ones_like(fl), fl, fh, h * (fh - fl)], 1); return float(np.linalg.lstsq(X, fb, rcond=None)[0][3])
def gap(k, idx): return gap_of(*(det(k, 'start3', f)[idx] for f in ('fbar_ep', 'f_light', 'f_heavy', 'heavy')))
def ledger(a, b):
    ca, cb = det(a, 'start3', 'class_ok_ep'), det(b, 'start3', 'class_ok_ep'); both = (det(a, 'start3', 'fbar_ep') > 0) & (det(b, 'start3', 'fbar_ep') > 0)
    cls = paired(ca[both] - cb[both]); cls.update(arm=float(ca[both].mean()), control=float(cb[both].mean()))
    ia = np.where(det(a, 'start3', 'fbar_ep') > 0)[0]; ib = np.where(det(b, 'start3', 'fbar_ep') > 0)[0]; ga, gb = gap(a, ia), gap(b, ib)
    rng = np.random.default_rng(0); n = len(det(a, 'start3', 'cost')); bs = []
    for _ in range(NB):
        r = rng.integers(0, n, n); bs.append(gap(a, r[np.isin(r, ia)]) - gap(b, r[np.isin(r, ib)]))
    return dict(class_decision=cls, class_gap=dict(arm=ga, control=gb, diff=ga - gb, ci95=ci(bs), n_arm=int(len(ia)), n_control=int(len(ib))))

out = dict(keys=dict(N=KN, I=KI))
# guarantees
for arm in ('N', 'I'):
    h = [x for x in json.load(open(paths.run_dir(f'l2_a4_{arm}') / 'hist.json')) if 321 <= x['it'] <= IT]
    out.setdefault('guarantees', {})[arm] = dict(iterations=len(h), viol=float(sum(x['viol'] for x in h)), nan=float(sum(x['nan'] for x in h)),
        n_sat_below=float(sum(x['n_sat_below'] for x in h)), tank_bound_min=min(x['tank_bound_min'] for x in h if x['tank_bound_min'] is not None))
G = out['guarantees']; out['guarantees']['pass'] = bool(all(G[a]['viol'] == 0 and G[a]['nan'] == 0 for a in ('N', 'I')))
# (T)
Q = ('P1', 'P2', 'P3', 'P3c', 'P4', 'chain_P1234')
def task(d):
    t = {q: dict(d[q], improves=bool(d[q]['diff'] >= 0.10 and d[q]['ci95'][0] > 0), non_inferior=bool(d[q]['diff'] >= -0.10)) for q in Q}
    return dict(quantities=t, all_non_inferior=bool(all(t[q]['non_inferior'] for q in Q)), improves_P3_P4_chain=[q for q in ('P3', 'P4', 'chain_P1234') if t[q]['improves']])
out['T'] = task(phase_diff.diff(PS, KI, KN)); out['T_gripper_sampled_descriptive'] = phase_diff.diff(PS, KI, KN, 'gripper_sampled')
out['phase_rates'] = {k: {md: {g: {q: PS[k][md][g][q]['rate'] for q in PS[k][md][g]} for g in ('alone', 'chain')} for md in ('deterministic', 'gripper_sampled')} for k in (KN, KI)}
# (Hd)
H = json.load(open(paths.RECORDS / 'l2_attempt4_hd.json')); hn, hi = H['N_brain_N_hand'], H['N_brain_I_hand']
ratio = hi['residual_median'] / hn['residual_median']; mis = paired(np.asarray(hi['mis_hold_ep']) - np.asarray(hn['mis_hold_ep']))
Hd_T = task(phase_diff.diff({'NI': dict(deterministic=hi['deterministic']), 'NN': dict(deterministic=hn['deterministic'])}, 'NI', 'NN'))
out['Hd'] = dict(residual=dict(I_hand=hi['residual_median'], N_hand=hn['residual_median'], ratio=ratio, pass_=bool(ratio <= 0.5)),
                 mis_hold=dict(mis, I_hand_median=hi['mis_hold_median'], N_hand_median=hn['mis_hold_median'], pass_=bool(mis['ci95'][0] <= 0)),
                 T_under_pairing=Hd_T, held_start3=dict(I_hand=hi['held_start3'], N_hand=hn['held_start3']))
out['Hd']['pass'] = bool(out['Hd']['residual']['pass_'] and out['Hd']['mis_hold']['pass_'] and Hd_T['all_non_inferior'])
# (Ld)
L = ledger(KI, KN); Mk = json.load(open(paths.RECORDS / 'l2_attempt4_masked.json'))
vis, msk = Mk['visible'], Mk['masked']
same = bool(vis['class_ok_ep'] == R[KI]['start3']['deterministic']['class_ok_ep'] and vis['fbar_ep'] == R[KI]['start3']['deterministic']['fbar_ep'])
both = (np.asarray(vis['fbar_ep']) > 0) & (np.asarray(msk['fbar_ep']) > 0)
mk = paired(np.asarray(vis['class_ok_ep'])[both] - np.asarray(msk['class_ok_ep'])[both])
mk.update(visible=float(np.asarray(vis['class_ok_ep'])[both].mean()), masked=float(np.asarray(msk['class_ok_ep'])[both].mean()), visible_equals_decision_eval=same)
out['Ld'] = dict(**L, masking=mk); out['Ld']['pass'] = bool(L['class_decision']['ci95'][0] > 0 and mk['ci95'][0] > 0)
# adoption
T = out['T']; out['adopted'] = bool(T['all_non_inferior'] and (len(T['improves_P3_P4_chain']) > 0 or (out['Hd']['pass'] and out['Ld']['pass'])))
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('T_gripper_sampled_descriptive', 'phase_rates')}, indent=1))
