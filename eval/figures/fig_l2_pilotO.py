"""Learning curves of pilot O (records/l2_pilotO_plan.md): training cost, KL and clip share, goal standard deviation, hand transfer residual and role loss,
built-in evaluation cost from the original start, for every arm present in the group.
Usage: python eval/figures/fig_l2_pilotO.py GROUP_DIR OUT_PNG"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

G, OUT = sys.argv[1:3]
ARMS = [a for a in ('C', 'N', 'K', 'H', 'D') if os.path.exists(f'{G}/l2_pilotO_{a}/hist.json')]
COL = dict(C='#3b3b3b', N='#2f6db5', K='#c0562b', H='#4f9a4a', D='#8a55a8')
INK, GRID, SURF = '#2b2b2b', '#e4e2da', '#fbfaf6'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fig, ax = plt.subplots(2, 3, figsize=(11, 6)); ax = ax.ravel()
sm = lambda y, k=5: np.convolve(np.asarray(y, float), np.ones(k) / k, mode='valid')
for a in ARMS:
    H = json.load(open(f'{G}/l2_pilotO_{a}/hist.json')); E = json.load(open(f'{G}/l2_pilotO_{a}/evals.json'))
    it = np.array([h['it'] for h in H]); g = lambda k: np.array([np.nan if h.get(k) is None else h[k] for h in H], float); c = COL[a]
    ax[0].plot(it, g('cost'), color=c, alpha=0.25, lw=0.8); ax[0].plot(it[4:], sm(g('cost')), color=c, lw=1.4, label=a)
    ax[1].plot(it, g('kl'), color=c, lw=1.0); ax[2].plot(it, g('clipfrac'), color=c, lw=1.0)
    ax[3].plot(it, g('sd_goal'), color=c, lw=1.2)
    ie = [e['it'] for e in E]
    ax[4].plot(ie, [np.median(e['hand']['res_learned']) for e in E], 'o-', color=c, ms=3, lw=1.2)
    ax[5].plot(ie, [np.mean(e['nom']['cost']) for e in E], 'o-', color=c, ms=3, lw=1.2)
ax[1].axhspan(0.01, 0.02, color=GRID, zorder=0); ax[1].set_yscale('log'); ax[4].axhline(1.768, color=INK, ls='--', lw=0.8)
for a_, t in zip(ax, ('training cost (original start; 5-iteration mean)', 'KL per iteration (band: 0.01-0.02)', 'clip share', 'goal standard deviation',
                      'hand transfer residual, median [N m] (dashed: 1.768)', 'evaluation cost, original start (64 episodes)')):
    a_.set_title(t, fontsize=8.5, color=INK, loc='left'); a_.grid(color=GRID, lw=0.6); a_.set_xlabel('iteration')
    for s in ('top', 'right'): a_.spines[s].set_visible(False)
ax[0].legend(frameon=False, ncol=len(ARMS), fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
