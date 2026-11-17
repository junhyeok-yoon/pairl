"""Figure for the Stage B environment section: gauge margin over the workspace, one test episode, contact integration accuracy."""
import numpy as np, json, matplotlib
from pairl import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pairl.stageb import params as P
from pairl.stageb.kin_np import best_margin

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']                        # categorical slots 1-4 (validated, light surface)
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.titlesize': 9.5, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a, grid=True):
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    if grid: a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.tick_params(length=3, width=0.6)

fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0, 1.0]))

# (a) gauge margin of the best posture over the workspace
X, Y = np.meshgrid(np.linspace(*P.BOX[0], 101), np.linspace(*P.BOX[1], 126)); mg = best_margin(np.stack([X, Y], -1), P.L_NOM, P.QC, P.QH)[0]
a = ax[0]; im = a.pcolormesh(X, Y, np.clip(mg, 0, 1), cmap=SEQ, vmin=0, vmax=1, shading='auto', rasterized=True)
a.contourf(X, Y, mg, levels=[-10, P.MARG_HARD[0]], colors=[GRID])
cs = a.contour(X, Y, mg, levels=[P.MARG_HARD[0], P.MARG_HARD[1], P.MARG_NORMAL], colors=[INK], linewidths=[1.0, 1.0, 1.6], linestyles=['dashed', 'dashed', 'solid'])
a.clabel(cs, fmt={P.MARG_HARD[0]: '0.08', P.MARG_HARD[1]: '0.20', P.MARG_NORMAL: '0.25'}, fontsize=7.5, colors=INK)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label('gauge margin of the best posture'); cb.outline.set_edgecolor(AXIS)
a.set_aspect('equal'); a.set_xlabel('x [m]'); a.set_ylabel('y [m]'); a.set_title('(a) margin to the joint limits over the workspace'); style(a, grid=False)

# (b), (c) one test episode
E = np.load(str(paths.RECORDS / 'stageb_env_episode.npz')); i = 0; n = int(E['ntot'][i]); p, fn, ph = E['p'][i][:n], E['fn'][i][:n], E['ph'][i][:n]; t = np.arange(n) * P.TS
a = ax[1]; xe, ye = float(E['xe'][i]), float(E['ye'][i])
a.fill([xe - P.WE, xe + P.WE, xe + P.WE, xe - P.WE], [ye, ye, ye - 0.12, ye - 0.12], color=GRID, ec=MUTED, lw=0.8, zorder=1)
names = ['1 pick', '2 carry and hold', '3 press', '4 release, retreat']
for j in range(4):
    m = ph == j; a.plot(p[m, 0], p[m, 1], color=CAT[j], lw=2, label=names[j], zorder=3)
a.plot(*E['po'][i], marker='s', ms=8, mfc='none', mec=INK, mew=1.4, ls='none', zorder=4); a.annotate('grasp point', E['po'][i], xytext=(6, 6), textcoords='offset points', color=INK2)
a.plot(*E['ps'][i], marker='o', ms=9, mfc='none', mec=INK, mew=1.4, ls='none', zorder=4); a.annotate('target', E['ps'][i], xytext=(7, 5), textcoords='offset points', color=INK2)
a.plot(*p[0], marker='x', ms=7, color=INK, mew=1.4, ls='none', zorder=4); a.annotate('start', p[0], xytext=(6, -11), textcoords='offset points', color=INK2)
a.annotate('block', (xe, ye - 0.06), ha='center', color=INK2)
a.set_aspect('equal'); a.set_xlabel('x [m]'); a.set_ylabel('y [m]'); a.set_title('(b) tip path of one test episode'); a.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=2); style(a)
a = ax[2]; m3 = np.nonzero(ph == 2)[0]; sl = slice(max(m3[0] - 150, 0), min(m3[-1] + 150, n))
a.plot(t[sl], fn[sl], color=CAT[0], lw=2); a.axhline(float(E['fstar'][i]), color=MUTED, lw=1.0, ls='--')
a.annotate('target force', (t[sl][0], float(E['fstar'][i])), xytext=(2, 4), textcoords='offset points', color=INK2)
for k in (m3[0], m3[-1] + 1): a.axvline(k * P.TS, color=AXIS, lw=0.8)
a.annotate('press phase', (0.5 * (t[m3[0]] + t[m3[-1]]), 0.0), xytext=(0, 5), textcoords='offset points', ha='center', color=INK2)
a.set_xlabel('time [s]'); a.set_ylabel('normal force on the top face [N]'); a.set_title('(c) contact force of the same episode'); style(a)

# (d) power-balance residual against the integration step (harsh regime)
T = json.load(open(str(paths.RECORDS / 'stageb_contact_runs.json'))); a = ax[3]; subs = [2, 4, 8, 16]; h = np.array([P.TS / s for s in subs]) * 1e3
for j, ke in enumerate((2000, 20000, 200000)):
    r = [T[f'harsh_{ke}_{s}']['residual_rel'] for s in subs]
    a.plot(h, r, color=CAT[j], lw=2, marker='o', ms=6, mec=SURF, mew=1.2, label=f'{ke:,} N/m'.replace(',', ' '))
    a.annotate({2000: '2e3 N/m', 20000: '2e4 N/m', 200000: '2e5 N/m'}[ke], (h[0], r[0]), xytext=(6, 0), textcoords='offset points', va='center', color=INK2)
hh = np.array([0.125, 1.0]); a.plot(hh, 3e-12 * (hh / 0.125) ** 4, color=MUTED, lw=1.0, ls='--'); a.annotate('slope 4', (0.25, 3e-12 * 2 ** 4), xytext=(6, -11), textcoords='offset points', color=INK2)
a.axvline(0.5, color=AXIS, lw=0.8); a.annotate('chosen step', (0.5, 2e-12), xytext=(4, 0), textcoords='offset points', ha='left', color=INK2)
a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(h); a.set_xticklabels(['1', '0.5', '0.25', '0.125']); a.minorticks_off(); a.set_xlim(0.1, 2.2)
a.set_xlabel('integration step [ms]'); a.set_ylabel('relative power-balance residual'); a.set_title('(d) contact integration accuracy'); a.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=3, title='block stiffness', title_fontsize=7.5, columnspacing=1.0, handlelength=1.6); style(a)
fig.tight_layout(); fig.savefig(str(paths.FIGURES / 'stageb_env.png'), dpi=200)
print('saved')
