import json, numpy as np
ARMS = ['full', 'no_memory', 'no_ledger', 'no_calibration', 'single_return', 'torque_policy']
M = {}
for arm in ARMS:
    for seed in (0, 1, 2):
        r = json.load(open(f'../records/m2_{arm}_{seed}.json'))
        e, tasks = np.array(r['eval']), np.array(r['tasks'])
        err = np.abs(e[:, 2] - 3.0)
        d = dict(med_err=float(np.median(err)), success=float(np.mean(err <= 0.3)), mis=float(np.median(e[:, 3])), viol=int(e[:, 8].sum()), minT=float(e[:, 7].min()),
                 reached=float(e[:, 9].mean()), pd=float(np.nanmean(e[:, 5])), impact=float(np.median(e[:, 6])), final_R=r['hist'][-1][0], final_L=r['hist'][-1][1])
        d['rsi'] = float(abs(np.corrcoef(tasks[:, 0], e[:, 4])[0, 1])) if arm != 'torque_policy' else None
        if 'eval_ledger_zeroed' in r:
            z = np.array(r['eval_ledger_zeroed']); d['reliance'] = float(np.median(np.abs(z[:, 2] - 3.0)) - d['med_err'])
        M[(arm, seed)] = d
g = lambda arm, k: [M[(arm, s)][k] for s in (0, 1, 2)]
C = {}
C['M0'] = all(x <= 0.5 for x in g('full', 'med_err'))
C['V1'] = all(M[k]['viol'] == 0 and M[k]['minT'] > 0 for k in M)
C['P1'] = all(a < b for a, b in zip(g('full', 'med_err'), g('no_memory', 'med_err'))) and all(x <= 0.3 for x in g('full', 'mis'))
C['P2'] = all(a < b for a, b in zip(g('full', 'med_err'), g('no_calibration', 'med_err')))
C['P3'] = sum(a < b for a, b in zip(g('full', 'med_err'), g('no_ledger', 'med_err'))) >= 2
C['P4'] = all(a < b for a, b in zip(g('full', 'mis'), g('single_return', 'mis')))
C['P5'] = all(x <= 0.3 for x in g('full', 'rsi')) and all(x >= 0.6 for x in g('no_memory', 'rsi'))
C['P6'] = all(x >= 0.3 for x in g('full', 'reliance'))
C['E_gap_seeds'] = sum((b - a) > 0.1 for a, b in zip(g('torque_policy', 'med_err'), g('full', 'med_err')))
for arm in ARMS:
    print(arm.ljust(15), ' '.join(f"{k}={','.join('%.3f' % x if isinstance(x, float) else str(x) for x in g(arm, k))}" for k in ('med_err', 'success', 'mis', 'rsi', 'pd', 'impact', 'reliance') if M[(arm, 0)].get(k) is not None),
          'viol', sum(g(arm, 'viol')), 'minT %.2f' % min(g(arm, 'minT')))
print(json.dumps(C))
json.dump(dict(metrics={f'{a}_{s}': v for (a, s), v in M.items()}, checks=C), open('../records/m2_summary.json', 'w'), indent=1)
