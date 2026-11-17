"""Decision of pilot L (records/l2_pilotL_plan.md, amendment 1): per L_tr arm against C at 260 (deterministic decision evaluation on the GPU with the
deterministic XLA setting, records/l2_pilotO_eval.json): (a) transfer residual below C's and below 1.768 N m; (b) role loss (hand_loss_all, mean over
251-260) at most 1.10 times C's; (c) cost difference (768 paired episodes) CI upper bound below +0.5; (d) (G) on 201-260. Descriptive: paired phase
success differences (records/phase_success.json, CPU). Usage: python eval/diagnostics/l2_pilotL_decide.py OUT_JSON"""
import sys, json, numpy as np
from pairl import paths
sys.path.insert(0, str(paths.ROOT / 'eval' / 'diagnostics')); import phase_diff
from l2_pilotO_eval import compare

EV = json.load(open(paths.RECORDS / 'l2_pilotO_eval.json')); PS = json.load(open(paths.RECORDS / 'phase_success.json')); out = {}
H = lambda a: json.load(open(paths.run_dir(f'l2_pilotLd_{a}') / 'hist.json'))
role = lambda h: float(np.mean([x['hand_loss_all'] for x in h if 251 <= x['it'] <= 260]))
G = lambda h: dict(viol=float(sum(x['viol'] for x in h)), nan=float(sum(x['nan'] for x in h)), n_sat_below=float(sum(x['n_sat_below'] for x in h)),
                   tank_bound_min=min(x['tank_bound_min'] for x in h if x['tank_bound_min'] is not None))
hc = H('C'); rc = EV['l2_pilotLd_C:260']['hand']['res_learned_median']; out['C'] = dict(residual=rc, role=role(hc), G=G(hc))
for arm, lam in (('L1', 1.0), ('L05', 0.5)):
    h = H(arm); r = EV[f'l2_pilotLd_{arm}:260']['hand']['res_learned_median']; c = compare(EV, f'l2_pilotLd_{arm}:260', 'l2_pilotLd_C:260'); g = G(h)
    d = dict(lam=lam, residual=r, role=role(h), role_ratio=role(h) / out['C']['role'], cost=c, G=g,
             a=bool(r < rc and r < 1.768), b=bool(role(h) <= 1.10 * out['C']['role']), c=bool(c['ci95'][1] < 0.5), d=bool(g['viol'] == 0 and g['nan'] == 0 and g['tank_bound_min'] >= -0.01),
             phase_vs_C=phase_diff.diff(PS, f'l2_pilotLd_{arm}:260', 'l2_pilotLd_C:260'),
             phase_vs_C_gripper_sampled=phase_diff.diff(PS, f'l2_pilotLd_{arm}:260', 'l2_pilotLd_C:260', 'gripper_sampled'))
    d['pass'] = d['a'] and d['b'] and d['c'] and d['d']; out[arm] = d
out['adopted'] = 'L05' if out['L05']['pass'] else ('L1' if out['L1']['pass'] else None)
out['base_checkpoint'] = str(paths.run_dir(f"l2_pilotLd_{out['adopted'] or 'C'}") / 'params_it0260.pkl')
out['base_ltr'] = {None: None, 'L1': 1.0, 'L05': 0.5}[out['adopted']]
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if not kk.startswith('phase')} if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
