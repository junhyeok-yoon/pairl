import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_closed_loop import run

# One case of the closed-loop check (largest normalized error among the 64 cases of the proposed law).
out, tr = run('proposed', False, record=True)
j = int(np.argmax(out['wmax_all']))
t, wn, psi, psip, gr, lam, r1, r2, e1, e2 = tr[:, :, j].T
eps = 1.0 - out['m_star']
print(f"case {j}: max ||w|| = {out['wmax_all'][j]:.5f}, a priori bound = {eps:.5f}, "
      f"time with yield < 1: {np.mean(gr < 1):.3f}, time saturated: {np.mean(lam < 1):.3f}")

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.0))
a = ax[0, 0]
a.plot(t, wn, color='C0', lw=0.9, label=r'normalized error $\|e\|/\psi_g$')
a.axhline(1.0, color='k', lw=1.0, label='tolerance boundary')
a.axhline(eps, color='C3', ls='--', lw=1.0, label=rf'a priori bound $1-m_{{\star}}={eps:.3f}$')
a.set_ylim(0, 1.05); a.set_xlabel('time [s]'); a.legend(loc='lower right', fontsize=8); a.set_title('(a) tolerance')
a = ax[0, 1]
a.step(t, psip, where='post', color='0.45', lw=1.0, label=r'requested tolerance $\psi_p$')
a.plot(t, psi, color='C1', lw=1.2, label=r'realized tolerance $\psi_g$')
a.set_xlabel('time [s]'); a.legend(loc='upper right', fontsize=8); a.set_title('(b) requested and realized tolerance')
a = ax[1, 0]
a.plot(t, gr, color='C2', lw=0.9, label=r'yield factor $\gamma_r(m)$')
a.plot(t, lam, color='C4', lw=0.7, alpha=0.8, label=r'saturation factor $\lambda$ (1 = inactive)')
a.set_ylim(-0.02, 1.05); a.set_xlabel('time [s]'); a.legend(loc='lower right', fontsize=8); a.set_title('(c) yielding and saturation')
a = ax[1, 1]
a.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec='k', lw=1.2, label=r'constraint set $\mathcal{C}$'))
a.plot(r1, r2, color='C1', lw=1.0, label=r'realized reference $r_g$')
a.plot(e1, e2, color='C0', lw=0.6, alpha=0.8, label=r'output $\eta$')
for n, k in enumerate(range(0, len(t), 250)):
    a.add_patch(plt.Circle((r1[k], r2[k]), psi[k], fill=False, ec='C3', lw=0.6, alpha=0.7,
                           label=r'tube $\|\eta-r_g\|\leq\psi_g$ (every 2.5 s)' if n == 0 else None))
a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15); a.set_aspect('equal')
a.set_xlabel(r'$\eta_1$'); a.set_ylabel(r'$\eta_2$'); a.legend(loc='lower left', fontsize=7); a.set_title('(d) tube and constraint set')
fig.tight_layout()
fig.savefig('../figures/yield_core.png', dpi=170)
