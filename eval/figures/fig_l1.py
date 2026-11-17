"""Figure of the pre-registered L1 runs from records/l1_results.json."""
import json, numpy as np, matplotlib
from pairl import paths
matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = json.load(open(str(paths.RECORDS / 'l1_results.json'))); A = lambda k: {m: np.asarray(v) for m, v in R[k].items()}
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, axis='y', color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6)
def bars(a, labels, vals, fmt, col=CAT[0], top=None):
    x = np.arange(len(vals)); a.bar(x, vals, width=0.62, color=col)
    for xi, v in zip(x, vals): a.annotate(fmt % v, (xi, v), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5)
    if top is not None: a.set_ylim(0, top)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0)); ax = ax.ravel()
arms = [('ID_ref', 'reference brain,\nv3 hand'), ('ID_noled', 'no ledger,\nv3 hand'), ('ID_nomem', 'reference brain,\nmemory off'), ('ID_nomem_khi', 'reference brain,\nmemory off, k_hi')]
bars(ax[0], [l for k, l in arms], [np.median(A(k)['frel']) for k, l in arms], '%.3f', top=0.75)
ax[0].set_ylabel('median relative force error'); ax[0].set_title('(a) force error in distribution (256 episodes)'); style(ax[0])
cls = []
for k, l in arms:
    a = A(k); g = a['grasp'] > 0.5; cls.append(np.mean(a['class_ok'][g]))
bars(ax[1], [l for k, l in arms], cls, '%.3f', top=1.15)
ax[1].axhline(0.5, color=MUTED, lw=0.8, ls=':'); ax[1].set_ylabel('class decision accuracy'); ax[1].set_title('(b) payload class chosen in the press'); style(ax[1])
sets = [('ID_ref', 'in distribution'), ('HEAVY_ref', 'heavy object'), ('BODY_MASS_ref', 'masses and\nobject only'), ('BODY_LEN_ref', 'lengths and\nobject only'), ('BODY_ref', 'body\n(Table 2)'), ('STIFF_ref', 'stiff block')]
bars(ax[2], [l for k, l in sets], [np.mean(A(k)['grasp']) for k, l in sets], '%.3f', top=1.15)
ax[2].set_ylabel('grasp success'); ax[2].set_title('(c) reference brain out of distribution'); style(ax[2])
ref, bud = A('ID_ref'), A('BUD_ref'); hard = np.asarray(R['ID_hard']) > 0.5
dev = [('constraint layer,\nhard episodes', np.mean(ref['cons'][hard])), ('constraint layer,\nother episodes', np.mean(ref['cons'][~hard])),
       ('gate,\nbudget episodes', np.mean(bud['gate'])), ('saturation', np.mean(ref['sat'])), ('guard', np.mean(ref['guard']))]
bars(ax[3], [l for l, v in dev], [v for l, v in dev], '%.4f', top=0.9)
ax[3].set_ylabel('fraction of time active'); ax[3].set_title('(d) guarantee devices in operation'); style(ax[3])
fig.tight_layout(); fig.savefig(str(paths.FIGURES / 'l1_results.png'), dpi=200)
