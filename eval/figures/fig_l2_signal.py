"""Learning signal of the brain by phase (plan of attempt 2, amendment 6): correlation of the advantage with the goal-tip deviation toward the target, with 95%
bootstrap intervals; decisions per 128 episodes; distance from the mean goal tip to the target; spread of the goal-tip deviation along the target direction.
Usage: python3 fig_l2_signal.py SIGNAL_JSON OUT_PNG [RUN=l2_attempt2]"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1])); OUT = sys.argv[2]; RUN = sys.argv[3] if len(sys.argv) > 3 else 'l2_attempt2'
ITS = sorted(int(k.split(':')[1]) for k in R if k.split(':')[0] == RUN)
CLS = [('phase1_free_to_grasp_point', 'phase 1, not held: toward the grasp point'), ('phase2_held_to_p_star', 'phase 2, held: toward p*'),
       ('phase3_held_to_p_c', 'phase 3, held: toward p_c')]
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6); a.set_xlabel('iteration'); a.set_xticks(ITS)
fig, ax = plt.subplots(2, 3, figsize=(14, 7.4)); ax = ax.ravel()
for ci, (c, lab) in enumerate(CLS):
    x = np.array(ITS) + (ci - 1) * 4.0; g = lambda k: [R[f'{RUN}:{i}'][c][k] for i in ITS]
    for a, lam in ((ax[0], '0.95'), (ax[1], '1')):
        r = np.array(g(f'corr_lambda{lam}'), float); lo, hi = np.array(g(f'ci95_lambda{lam}'), float).T
        a.errorbar(x, r, yerr=[r - lo, hi - r], fmt='o', color=CAT[ci], ms=5, lw=1.4, capsize=3, label=lab)
    for a, k in ((ax[2], 'decisions_per_128_episodes'), (ax[3], 'goal_tip_to_target_median'), (ax[4], 'deviation_projection_sd')):
        a.plot(ITS, g(k), color=CAT[ci], marker='o', ms=5, lw=1.6, label=lab)
for a, t in ((ax[0], '(a) correlation, advantage with lambda 0.95 (training)'), (ax[1], '(b) correlation, advantage with lambda 1')):
    a.axhline(0, color=MUTED, ls='--', lw=0.9); a.set_title(t); a.set_ylabel('correlation with the deviation toward the target'); style(a)
ax[2].set_yscale('log'); ax[2].set_title('(c) decisions per 128 episodes'); style(ax[2])
ax[3].set_title('(d) mean goal tip to target, median [m]'); ax[3].set_ylim(bottom=0); style(ax[3])
ax[4].set_title('(e) goal-tip deviation along the target direction, sd [m]'); ax[4].set_ylim(bottom=0); style(ax[4])
ax[5].axis('off'); ax[5].legend(*ax[2].get_legend_handles_labels(), loc='center', frameon=False, fontsize=9, title='decision class (exploring rollouts, 2048 episodes)', title_fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
