"""Status of the learning at the end of L2 attempt 3: evaluation cost by phase, PPO step size, exploration scale, critic fit, hand loss against transfer,
cost by grasp outcome. Usage: python3 fig_l2_status.py RUN PRE TRAJ_JSON L1_RESULTS_JSON OUT_PNG"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUN, PRE, TRAJ, L1R, OUT = sys.argv[1:6]
H = [h for h in json.load(open(PRE + '/hist.json')) if h['it'] <= 100] + json.load(open(RUN + '/hist.json'))
E = [e for e in json.load(open(PRE + '/evals.json')) if e['it'] < 100] + json.load(open(RUN + '/evals.json'))
T = json.load(open(TRAJ))['start0']['deterministic']; ref = np.asarray(json.load(open(L1R))['ID_ref']['cost_ph']).mean(0)
INK, INK2, MUTED, GRID, AXIS, SURF = '#1f1f1d', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; PH = ['phase 1: reach', 'phase 2: carry', 'phase 3: press', 'phase 4: release']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a, xl='iteration'):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6); a.set_xlabel(xl)
g = lambda k: np.array([np.nan if h.get(k) is None else h[k] for h in H], float); it = g('it'); ie = np.array([e['it'] for e in E])
sm = lambda y, n=10: np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode='edge'), np.ones(n) / n, mode='valid')
fig, ax = plt.subplots(2, 3, figsize=(14, 7.4)); ax = ax.ravel()
a = ax[0]; cp = np.array([np.asarray(e['nom']['cost_ph']).mean(0) for e in E])
for k in range(4):
    a.plot(ie, cp[:, k], color=CAT[k], marker='o', ms=3, lw=1.6, label=PH[k]); a.axhline(ref[k], color=CAT[k], ls='--', lw=0.9)
a.set_yscale('log'); a.set_title('(a) evaluation cost by phase, original start (dashed: L1 reference brain)'); a.set_ylim(0.06, 200); a.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper right'); style(a)
a = ax[1]; a.axhspan(0.01, 0.02, color=GRID, lw=0); a.plot(it, g('kl'), color=CAT[0], lw=0.7, alpha=0.45); a.plot(it, sm(g('kl')), color=CAT[0], lw=1.8, label='approximate KL per iteration')
a.plot(it, sm(g('clipfrac')), color=CAT[1], lw=1.8, label='share of clipped samples'); a.set_yscale('log'); a.set_ylim(0.005, 1)
a.text(4, 0.0115, 'usual KL target 0.01-0.02', color=MUTED, fontsize=7.5); a.set_title('(b) PPO step size'); a.legend(frameon=False, fontsize=7.5, loc='lower right'); style(a)
a = ax[2]; a.plot(it, g('sd_goal'), color=CAT[0], lw=1.6, label='goal action'); a.plot(it, g('sd_K'), color=CAT[1], lw=1.6, label='stiffness action')
a.set_yscale('log'); a.set_title('(c) exploration scale (standard deviation before the squash)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[3]
for k, lab, c in (('ev_orig', 'original start', CAT[0]), ('ev_r2', 'phase-2 start', CAT[1]), ('ev_r3', 'phase-3 start', CAT[2])): a.plot(it, sm(g(k)), color=c, lw=1.6, label=lab)
a.set_ylim(0, 1.02); a.set_title('(d) critic: explained variance of the returns'); a.legend(frameon=False, fontsize=7.5, loc='lower right'); style(a)
a = ax[4]; res = np.array([np.median(e['hand']['res_learned']) for e in E]); b0 = (ie >= 100) & (ie <= 150); hb = (it >= 100) & (it <= 150)
a.plot(it, sm(g('hand_loss')) / np.nanmean(g('hand_loss')[hb]), color=CAT[0], lw=1.6, label='role loss (training)')
a.plot(ie, res / res[b0].mean(), color=CAT[1], marker='o', ms=3, lw=1.6, label='transfer residual (evaluation)')
a.axhline(1.0, color=MUTED, ls=':', lw=0.9); a.set_ylim(0, 4); a.set_ylabel('relative to the mean over 100-150'); a.set_title('(e) hand: what is trained against what is intended'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[5]; names = [('grasp_in_phase1', 'grasp in phase 1'), ('grasp_in_phase2', 'grasp in phase 2'), ('never', 'no grasp')]; left = np.zeros(3)
for k in range(4):
    v = np.array([T['outcome'][n]['cost_by_phase'][k] for n, _ in names]); a.barh(range(3), v, left=left, color=CAT[k], edgecolor=SURF, lw=1.2, height=0.55, label=PH[k]); left += v
for i, (n, lab) in enumerate(names): a.text(left[i] + 0.5, i, f"{T['outcome'][n]['episodes']} of 64 episodes, cost {T['outcome'][n]['cost']:.1f}", va='center', color=INK2, fontsize=8)
a.set_yticks(range(3)); a.set_yticklabels([lab for _, lab in names]); a.invert_yaxis(); a.set_xlim(0, 64); a.axvline(ref.sum(), color=MUTED, ls='--', lw=0.9)
a.set_title('(f) iteration 300: cost by grasp outcome (dashed: L1 reference)'); a.legend(frameon=False, fontsize=7.5, loc='upper right'); style(a, 'cost per episode'); a.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
