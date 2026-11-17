"""Figure of the L1 engineering pilot from records/l1_pilot.json."""
import json, numpy as np, matplotlib
from pairl import paths
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pairl.stageb import model as Mo

R = json.load(open(str(paths.RECORDS / 'l1_pilot.json')))
A = lambda k, f=np.mean: float(f(np.asarray(R[k]['cost'])))
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2)); ax = ax.ravel()
a = ax[0]
groups = [('hand memory\n(reference brain)', 'A_ref', 'B_ref'), ('no memory', 'A_ref_nomem', None), ('no memory,\nK = k_hi', 'A_stiff_nomem', None),
          ('no memory, brain\nintegral via q_d', 'A_intr_nomem', 'B_intr_nomem')]
w = 0.36
for i, (lab, kid, khv) in enumerate(groups):
    for j, (k, col, nm) in enumerate(((kid, CAT[0], 'in distribution'), (khv, CAT[1], 'heavy objects (1.5-2.5 kg)'))):
        if k is None: continue
        v = A(k); x = i + (j - 0.5) * w
        a.bar(x, v, width=w - 0.04, color=col, label=nm if i == 0 else None)
        a.annotate(f'{v:.2f}', (x, v), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups], fontsize=7.5); a.set_ylabel('mean episode cost')
a.set_ylim(0, 17); a.set_title('(a) who removes the load error'); a.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=2); style(a)

a = ax[1]; rhos = [0.4, 0.6, 0.8, 1.0]
for k, col, mk, nm in (('ref', CAT[0], 'o', 'reference brain'), ('slow', CAT[1], 's', 'slow brain'), ('ledJ', CAT[2], '^', 'ledger brain (T switches to slow)')):
    a.plot(rhos, [A(f'D_{k}_off{r}') for r in rhos], color=col, marker=mk, lw=1.8, ms=5, label=nm)
a.set_xlabel(r'$\rho$ in $E_0=\Delta T+\rho\,E_{\mathrm{task}}$'); a.set_ylabel('mean episode cost'); a.set_xticks(rhos)
a.set_title('(b) cost under budgets'); a.legend(frameon=False, fontsize=7.5, loc='upper right'); style(a)

a = ax[2]
er, es = np.asarray(R['A_ref']['E_task']), np.asarray(R['A_slow']['E_task']); fail = np.asarray(R['D_ref_off0.6']['grasp']) < 0.5
lo = 1e-2
a.scatter(np.maximum(er[~fail], lo), np.maximum(es[~fail], lo), s=14, color=CAT[0], alpha=0.85, label='reference brain grasps at rho = 0.6')
a.scatter(np.maximum(er[fail], lo), np.maximum(es[fail], lo), s=22, facecolors='none', edgecolors=CAT[1], lw=1.2, marker='s', label='reference brain fails at rho = 0.6')
a.plot([lo, 10], [lo, 10], color=MUTED, lw=0.8, ls='--'); a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('E_task of the reference brain [J] (values below 0.01 at 0.01)'); a.set_ylabel('E_task of the slow brain [J]')
a.set_title('(c) energy saved by moving slowly'); a.legend(frameon=False, fontsize=7, loc='upper left'); style(a)

a = ax[3]
mo = np.asarray(R['E_mo']); wh = np.asarray(R['A_ref']['w_hat']) / Mo.G0
a.axvspan(0.65, 1.05, color=GRID, alpha=0.6, lw=0); a.axhline(0.85, color=MUTED, lw=0.8, ls=':'); a.axvline(0.85, color=MUTED, lw=0.8, ls=':')
a.scatter(mo, wh, s=14, color=CAT[0], alpha=0.85)
a.plot([0.1, 1.6], [0.1, 1.6], color=MUTED, lw=0.8, ls='--')
a.set_xlabel('object mass [kg]'); a.set_ylabel('mass from the learned load [kg]')
a.set_title('(d) payload seen by the brain via the ledger'); style(a)
fig.tight_layout(); fig.savefig(str(paths.FIGURES / 'l1_pilot.png'), dpi=200)
