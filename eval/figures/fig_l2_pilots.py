"""Pilots from the same start and seed: P1 (cost version 2), P2 (version 2, gripper head reset), P3 (version 3, gripper head reset, role-loss weighting).
Usage: python3 fig_l2_pilots.py OUT_PNG P1_DIR P2_DIR [P3_DIR]"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT, DIRS = sys.argv[1], sys.argv[2:]
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6)
LAB = ['P1: cost version 2', 'P2: cost version 2, gripper head reset', 'P3: cost version 3, gripper head reset, role-loss weighting']
runs = [(LAB[i], d, CAT[i], 'os^'[i]) for i, d in enumerate(DIRS)]
fig, ax = plt.subplots(2, 3, figsize=(14, 7.2)); ax = ax.ravel()
for lab, d, c, mk in runs:
    H = json.load(open(d + '/hist.json')); E = json.load(open(d + '/evals.json')); it = [h['it'] for h in H]; ie = [e['it'] for e in E]
    nan = lambda v: np.nan if v is None else v
    ax[0].plot(it, [nan(h['p_open_att']) for h in H], color=c, lw=1.4, label=lab)
    ax[1].plot(it, [h['att3'] for h in H], color=c, lw=1.4, label=lab)
    ax[2].plot(ie, [np.median(e['hand']['res_learned']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[3].plot(it, [h['grasp'] for h in H], color=c, lw=1.4, label=lab)
    ax[4].plot(ie, [np.mean(e['nom']['grasp']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[5].plot(ie, [np.median(e['nom']['pos2']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
tt = ['(a) open probability while held, phases 2-3 (training)', '(b) episodes held in phase 3 (training)', '(c) hand: residual after calibration elsewhere [N m] (evaluation)',
      '(d) grasp at the end of phase 1 (training)', '(e) grasp at the end of phase 1 (evaluation)', '(f) hold error median [m] (evaluation)']
for a, t in zip(ax, tt):
    a.set_title(t); a.set_xlabel('iteration'); style(a)
for a in (ax[0], ax[1], ax[3], ax[4]): a.set_ylim(-0.03, 1.03)
ax[2].axhline(1.768, color=MUTED, ls='--', lw=0.9); ax[2].text(100.5, 1.83, 'class basis + nominal prior (L1 hand)', fontsize=7, color=MUTED)
ax[0].legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(OUT, dpi=170)
