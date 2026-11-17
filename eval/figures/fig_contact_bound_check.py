"""Observed peak normal force against the bound of the theorem, one point per trajectory of check C2 (its contact interval with the largest ratio)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
try:
    from pairl import paths; RECORDS, FIGURES = paths.RECORDS, paths.FIGURES
except ImportError:
    RECORDS = Path(__file__).resolve().parents[1] / 'records'; FIGURES = Path(__file__).resolve().parents[1] / 'figures'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'; C = ['#2a78d6', '#eb6834']
D = np.load(RECORDS / 'contact_bound_check_c2.npz'); f, fb, kind = D['f'], D['fbar'], D['kind']
fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=220)
lo, hi = 0.8 * min(f.min(), fb.min()), 1.25 * max(f.max(), fb.max())
ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, zorder=1)
for j, name in enumerate(('feedback toward a point below the surface', 'random torque')):
    m = kind == j; ax.scatter(fb[m], f[m], s=9, color=C[j], alpha=0.55, linewidths=0.3, edgecolors='white', label=name, zorder=2 + j)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('bound of the theorem [N]', color=INK); ax.set_ylabel('observed peak normal force [N]', color=INK)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8, which='both', length=0)
ax.annotate('observed = bound', xy=(hi / 12, hi / 12), xytext=(hi / 300, hi / 6), color=INK2, fontsize=8, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
leg = ax.legend(loc='lower right', frameon=False, fontsize=8, markerscale=1.8, handletextpad=0.3)
for t in leg.get_texts(): t.set_color(INK)
fig.tight_layout(); fig.savefig(FIGURES / 'contact_bound_check.png'); print('points', len(f), 'max ratio', float((f / fb).max()))
