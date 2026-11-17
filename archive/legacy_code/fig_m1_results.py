import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(f): return json.load(open(f'../records/{f}'))
m1, m1b, m1c, m1d = load('m1_stage_a.json'), load('m1b_fixes.json'), load('m1c_guard.json'), load('m1d_confirm.json')
g = lambda R, name, key: [o[key] for o in R if o['group'] == name]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))
rng = np.random.default_rng(0)

def strip(a, cats, ylab, title, lines=(), log=False):
    for i, (lab, vals, col) in enumerate(cats):
        a.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=col, alpha=0.8)
    for y, ls, lab in lines:
        a.axhline(y, ls=ls, color='k', lw=1, label=lab)
    a.set_xticks(range(len(cats))); a.set_xticklabels([c[0] for c in cats], fontsize=8)
    a.set_ylabel(ylab); a.set_title(title, fontsize=10)
    if log: a.set_yscale('log')
    if lines: a.legend(fontsize=7, loc='upper right')

strip(ax[0], [('E,\nblock', g(m1, 'E_adv', 'tc_ss'), 'tab:red'), ('M,\nblock', g(m1, 'M_adv', 'tc_ss'), 'tab:orange'),
              ('E,\nyield', g(m1b, 'E_adv_yield', 'tc_ss'), 'tab:purple'), ("M',\nlatch", g(m1c, 'Mp_adv_latch', 'tc_ss'), 'tab:brown'),
              ("M',\ncontact cap", g(m1d, 'C_adv', 'tc_ss'), 'tab:green')],
      'steady contact torque [Nm]', '(a) adversarial brain', [(6.0, '--', 'tau_max'), (6.5, ':', 'criterion')])
strip(ax[1], [('E', g(m1, 'E_nom', 'mis_ss'), 'tab:red'), ('M', g(m1, 'M_nom', 'mis_ss'), 'tab:orange'),
              ('M,\nadversarial', g(m1, 'M_adv', 'mis_ss'), 'tab:orange'), ("M',\nadversarial", g(m1d, 'C_adv', 'mis_ss'), 'tab:green')],
      'steady |mismatch| [Nm]', '(b) calibration', [(0.3, ':', 'criterion')], log=True)
strip(ax[2], [('sign test', g(m1, 'E_gate', 'minT'), 'tab:red'), ('power margin', g(m1d, 'C_gate', 'minT'), 'tab:green')],
      'minimum tank level [J]', '(c) gate, T(0) = 1.2 J', [(1.0, '--', 'T_min'), (0.95, ':', 'criterion')])
S = load('m1d_summary.json')['peaks']
for j, (kv, col) in enumerate(((500, 'tab:blue'), (2000, 'tab:orange'), (5000, 'tab:red'))):
    v = np.array([0.05, 0.1, 0.2, 0.4]); lo = np.array([S[f'{x}_{kv}'][0] for x in v]); hi = np.array([S[f'{x}_{kv}'][1] for x in v])
    xs = v * (1 + 0.06 * (j - 1))
    ax[3].vlines(xs, lo, hi, color=col, lw=3, alpha=0.7, label=f'wall {kv} Nm/rad'); ax[3].plot(xs, hi, 'o', color=col, ms=4)
ax[3].axhline(6.0, ls='--', color='k', lw=1, label='tau_max'); ax[3].set_xscale('log'); ax[3].minorticks_off(); ax[3].set_xticks([0.05, 0.1, 0.2, 0.4]); ax[3].set_xticklabels(['0.05', '0.1', '0.2', '0.4'])
ax[3].set_xlabel('approach speed [rad/s]'); ax[3].set_ylabel('peak contact torque [Nm]'); ax[3].set_title('(d) impact, first 0.3 s', fontsize=10); ax[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig('../figures/m1_results.png', dpi=200)
