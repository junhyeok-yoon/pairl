import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS = ['full', 'no_memory', 'no_ledger', 'no_calibration', 'single_return', 'torque_policy']
LAB = ['full', 'no\nmemory', 'no\nledger', 'no\ncalibration', 'single\nreturn', 'torque\npolicy']
E = {(a, s): np.array(json.load(open(f'../records/m2_{a}_{s}.json'))['eval']) for a in ARMS for s in (0, 1, 2)}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cols = ['tab:blue', 'tab:orange', 'tab:green']
for i, a in enumerate(ARMS):
    for s in (0, 1, 2):
        e = E[(a, s)]
        ax[0].scatter(i + (s - 1) * 0.15, np.median(np.abs(e[:, 2] - 3)), color=cols[s], s=22, label=f'seed {s}' if i == 0 else None)
        if a != 'torque_policy':
            ax[1].scatter(i + (s - 1) * 0.15, np.median(e[:, 3]), color=cols[s], s=22)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(LAB, fontsize=8); ax[0].set_ylabel('median steady error [Nm]'); ax[0].set_title('(a) task', fontsize=10); ax[0].legend(fontsize=7)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(LAB[:5], fontsize=8); ax[1].set_yscale('log'); ax[1].axhline(0.3, ls=':', color='k', lw=1, label='criterion')
ax[1].set_ylabel('median |mismatch| [Nm]'); ax[1].set_title('(b) hand role', fontsize=10); ax[1].legend(fontsize=7)
mk = {'full': 'o', 'no_memory': 'x', 'no_ledger': 's', 'no_calibration': '^', 'single_return': 'D'}
for a in ARMS[:5]:
    for s in (0, 1, 2):
        e = E[(a, s)]
        ax[2].scatter(-e[:, 4].mean(), np.median(e[:, 3]), marker=mk[a], color=cols[s], s=30, label=a.replace('_', ' ') if s == 0 else None)
ax[2].axvline(6.0, ls='--', color='k', lw=1, label='spring limit')
ax[2].set_xlabel('mean spring torque in contact [Nm]'); ax[2].set_ylabel('median |mismatch| [Nm]'); ax[2].set_title('(c) spring torque and mismatch per run', fontsize=10); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig('../figures/m2_results.png', dpi=200)
