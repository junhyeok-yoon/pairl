import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pre-registered cause check: per-seed joint error of each repair against round 1 -> ../records/rlfix_summary.json, ../figures/cause_check.png
base = json.load(open('../records/rl_summary.json'))
A0, B0 = base['A']['err'], base['B']['err']
fixes = [('psi', 'D1 looser tolerance'), ('gain', 'D2 learned gain'), ('target', 'D3 target action'), ('feat', 'D4 meta-learned features'),
         ('psi-gain-feat-target', 'all four')]
S = {}
for key, name in fixes:
    per = []
    for s in range(3):
        rows = [json.loads(l) for l in open(f'../records/rlfix_{key}_A_s{s}.jsonl')]
        fin = [r for r in rows if r['kind'] == 'test'][-1]; assert fin['it'] == 70
        per.append(dict(err=fin['err'], rel=(fin['err'] - A0[s]) / A0[s], viol=fin['viol_eps'] + fin['train_viol_eps'],
                        wmax=max(r['wmax'] for r in rows), y1=fin['y1']))
    S[key] = dict(name=name, per=per, supported=all(p['rel'] <= -0.2 for p in per))
half = [(a + b) / 2 for a, b in zip(A0, B0)]
S['criteria'] = dict(half_gap=half, combined_closes_half=all(p['err'] <= 0.429 for p in S['psi-gain-feat-target']['per']),
                     order=sorted([k for k, _ in fixes[:4]], key=lambda k: np.mean([p['rel'] for p in S[k]['per']])))
json.dump(S, open('../records/rlfix_summary.json', 'w'), indent=1)
for key, _ in fixes:
    print(key, [round(p['err'], 4) for p in S[key]['per']], [f"{100 * p['rel']:+.1f}%" for p in S[key]['per']], 'supported', S[key]['supported'],
          'viol', sum(p['viol'] for p in S[key]['per']), 'wmax', round(max(p['wmax'] for p in S[key]['per']), 4), 'y1', [round(p['y1'], 3) for p in S[key]['per']])
print('criteria', S['criteria'])
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for i, (key, name) in enumerate(fixes):
    rel = [100 * p['rel'] for p in S[key]['per']]
    ax[0].bar(i, np.mean(rel), color='0.7', width=0.6); ax[0].scatter([i] * 3, rel, color='k', s=12, zorder=3)
ax[0].axhline(-20, color='C3', ls='--', lw=1.2, label='pre-registered threshold (-20 %)'); ax[0].axhline(0, color='k', lw=0.8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([n.replace(' ', '\n') for _, n in fixes], fontsize=8)
ax[0].set_title('(a) change of joint error vs. round 1 [%]'); ax[0].legend(fontsize=8, loc='lower left')
labels = ['proposed\n(round 1)', 'all four\nrepairs', 'input policy\n+ filter']
vals = [A0, [p['err'] for p in S['psi-gain-feat-target']['per']], B0]
for i, v in enumerate(vals):
    ax[1].bar(i, np.mean(v), color=['C0', 'C1', 'C3'][i], width=0.6, alpha=0.8); ax[1].scatter([i] * 3, v, color='k', s=12, zorder=3)
ax[1].axhline(0.429, color='k', ls=':', lw=1.2, label='half of the gap (0.429 rad)')
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels, fontsize=8); ax[1].set_title('(b) mean joint error on test tasks [rad]'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig('../figures/cause_check.png', dpi=170)
