"""Two arms evaluated from the restart states and from the original start (pilots R and T): hand transfer residual, guarantees.
Usage: python3 fig_l2_pilotR.py OUT_PNG A_DIR B_DIR [labelA=...] [labelB=...]"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT, DA, DB = sys.argv[1:4]; OPT = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a)
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
ARMS = [(OPT.get('labelA', 'A: original start distribution'), DA, '#2a78d6', 'o'), (OPT.get('labelB', 'B: restart share 0.5 (phase-2 and phase-3 starts, object held)'), DB, '#eb6834', 's')]
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6); a.set_xlabel('iteration')
def held_med(e, tag, key, sel):
    m = np.asarray(e[tag][sel]) > 0.5
    return float(np.median(np.asarray(e[tag][key])[m])) if m.sum() >= 3 else np.nan
fig, ax = plt.subplots(2, 3, figsize=(14, 7.4)); ax = ax.ravel()
for lab, d, c, mk in ARMS:
    E = json.load(open(d + '/evals.json')); H = json.load(open(d + '/hist.json')); ie = [e['it'] for e in E]; it = [h['it'] for h in H]
    ax[0].plot(ie, [np.mean(e['r3']['top_window']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[1].plot(ie, [np.mean(e['r2']['held_start3']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[2].plot(ie, [held_med(e, 'r2', 'pos2', 'held_start3') for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[3].plot(ie, [np.mean(e['nom']['grasp']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ax[4].plot(ie, [np.median(e['hand']['res_learned']) for e in E], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    tb = [h['tank_bound_min'] if h['tank_bound_min'] is not None else np.nan for h in H]
    ax[5].plot(it, np.maximum(tb, 1e-8), color=c, lw=1.2, label=lab + ': min tank-bound margin [J]')
    ax[5].plot(it, np.array([h['viol'] for h in H]) + 1e-8, color=c, lw=1.0, ls='--', label=lab.split(':')[0] + ': joint violations (count, +1e-8)')
ax[0].axhline(0.25, color=MUTED, ls=':', lw=0.9)
ax[4].axhline(1.768, color=MUTED, ls=':', lw=0.9)
tt = ['(a) phase-3 start: held, top-face force in the force window', '(b) phase-2 start: held at the start of phase 3', '(c) phase-2 start: hold error at p*, median of the held [m]',
      '(d) original start: grasp at the end of phase 1', '(e) hand: transfer residual, median [N m] (dotted: class basis)', '(f) guarantees on every iterate (training)']
for a, t in zip(ax, tt):
    a.set_title(t); style(a)
for a in (ax[0], ax[1], ax[3]): a.set_ylim(-0.03, 1.03)
ax[2].set_ylim(bottom=0); ax[5].set_yscale('log'); ax[5].legend(frameon=False, fontsize=6.5)
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT, dpi=170)
