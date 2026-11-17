"""Force bound of the corollary (speed limit, relief after a delay) against the speed limit, for the three contact stiffnesses of the enlarged class, with the class constants
of records/contact_bound_constants.json; the horizontal line is the normal force of the sampled certificate of (B3)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from pairl.tools.lib_contact_bound import fbar_reaction
try:
    from pairl import paths; RECORDS, FIGURES = paths.RECORDS, paths.FIGURES
except ImportError:
    RECORDS = Path(__file__).resolve().parents[1] / 'records'; FIGURES = Path(__file__).resolve().parents[1] / 'figures'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'; C = ['#2a78d6', '#eb6834', '#1baf7a']
R = json.load(open(RECORDS / 'contact_bound_constants.json')); K = R['class_in']; F_CERT = 60.0; D = json.load(open(RECORDS / 'contact_class_design.json'))
v = np.logspace(-2, 0, 300); fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=220)
ax.axhline(F_CERT, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1); ax.text(0.97, F_CERT * 0.33, '60 N, certificate of (B3)', color=INK2, fontsize=8, ha='right')
for j, k in enumerate((2e4, 2e5, 1e6)):
    f = fbar_reaction(v, K['a_r'], K['mu_low'], k, R['td'], R['de_bar']); ax.plot(v, f, color=C[j], lw=2.0, label=f'{k:.0e} N/m'.replace('e+0', 'e'), zorder=2)
    vs = D['speed'][f'class_nu05_k{int(k)}_f60_td4']; ax.scatter([vs], [F_CERT], s=42, color=C[j], edgecolors='white', linewidths=1.2, zorder=3)
    ax.text(vs * 1.06, F_CERT * 0.62, f'{vs:.3f} m/s', color=INK, fontsize=8, ha='left')
    ax.text(v[-1] * 1.05, f[-1], f'{k:.0e} N/m'.replace('e+0', 'e'), color=INK, fontsize=8, va='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.01, 1.0); ax.set_ylim(1, 1e5)
ax.set_xlabel('speed limit of the contact point [m/s]', color=INK); ax.set_ylabel('bound on the normal force [N]', color=INK)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8, which='both', length=0)
leg = ax.legend(title='contact stiffness', loc='upper left', frameon=False, fontsize=8, title_fontsize=8)
for t in leg.get_texts(): t.set_color(INK)
leg.get_title().set_color(INK2)
fig.subplots_adjust(left=0.12, right=0.86, bottom=0.13, top=0.97); fig.savefig(FIGURES / 'force_speed_design.png')
print({k: D['speed'][f'class_nu05_k{int(k)}_f60_td4'] for k in (2e4, 2e5, 1e6)})
