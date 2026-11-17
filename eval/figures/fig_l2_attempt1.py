"""Learning curves and component diagnostics of L2 attempt 1 (runs concatenated in order; a later run continues from its first checkpoint).
Usage: python3 fig_l2_attempt1.py OUT_PNG MECH_JSON RUN_DIR [RUN_DIR ...]"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT, MECH, RUNS = sys.argv[1], sys.argv[2], sys.argv[3:]
H, E, starts = [], [], []
for r in RUNS:
    h = json.load(open(r + '/hist.json')); e = json.load(open(r + '/evals.json'))
    if H:
        i0 = h[0]['it'] - 1; starts.append(i0)
        H = [a for a in H if a['it'] <= i0]; E = [a for a in E if a['it'] <= i0]; e = [a for a in e if a['it'] > i0]
    H += h; E += e
M = json.load(open(MECH))
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6)
    for i0 in starts: a.axvline(i0, color=MUTED, ls=':', lw=0.9)
L1 = dict(cost=2.39, hold=0.0014, frel=0.032)
it = np.array([h['it'] for h in H]); ie = np.array([e['it'] for e in E])
ev = lambda tag, k, f=np.mean: np.array([f(np.asarray(e[tag][k], float)) for e in E])
def cls_contact(tag, nmin=5):
    """Class decision among grasped episodes with contact in the force window; NaN with fewer than nmin such episodes."""
    out, n = [], []
    for e in E:
        c = (np.asarray(e[tag]['grasp']) > 0.5) & (np.asarray(e[tag]['fbar']) > 0); n.append(int(c.sum())); out.append(np.mean(np.asarray(e[tag]['class_ok'])[c]) if c.sum() >= nmin else np.nan)
    return np.array(out), np.array(n)
im = np.array([r['it'] for r in M]); mk = lambda ph, k: np.array([r[ph][k] if r[ph] else np.nan for r in M])

fig, ax = plt.subplots(2, 4, figsize=(18, 7.8)); ax = ax.ravel()
a = ax[0]; a.plot(it, [h['cost'] for h in H], color=CAT[0], lw=1.2, label='training (exploring)'); a.plot(ie, ev('nom', 'cost'), color=CAT[1], marker='o', ms=4, lw=1.6, label='evaluation (deterministic)')
a.axhline(L1['cost'], color=MUTED, ls='--', lw=0.9, label='L1 reference brain'); a.set_yscale('log')
for i0 in starts: a.text(i0 + 1, 0.98, 'continued with\nAmendment 3', transform=a.get_xaxis_transform(), fontsize=7, color=MUTED, va='top'); a.set_xlabel('iteration'); a.set_ylabel('mean episode cost'); a.set_title('(a) task cost'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[1]; cc, nc = cls_contact('nom'); a.plot(ie, ev('nom', 'grasp'), color=CAT[0], marker='o', ms=4, lw=1.6, label='grasp at the end of phase 1')
a.plot(ie, nc / len(E[0]['nom']['grasp']), color=CAT[2], marker='^', ms=4, lw=1.4, label='held and in contact in the force window')
if np.any(np.isfinite(cc)): a.plot(ie, cc, color=CAT[1], marker='s', ms=4, lw=1.6, label='class decision (5 or more such episodes)')
else: a.text(0.02, 0.97, f'class decision not shown: at most {nc.max()} episodes held and in contact', transform=a.transAxes, fontsize=7.5, color=INK2, va='top')
a.set_ylim(-0.05, 1.05); a.set_xlabel('iteration'); a.set_title('(b) grasp, contact, class decision (evaluation)'); a.legend(frameon=False, fontsize=7.5, loc='center left'); style(a)
a = ax[2]; a.plot(ie, ev('nom', 'pos2', np.median), color=CAT[0], marker='o', ms=4, lw=1.6, label='hold error [m]'); a.plot(ie, ev('nom', 'frel', np.median), color=CAT[1], marker='s', ms=4, lw=1.6, label='relative force error')
a.axhline(L1['hold'], color=CAT[0], ls=':', lw=0.9); a.axhline(L1['frel'], color=CAT[1], ls=':', lw=0.9); a.set_yscale('log'); a.set_xlabel('iteration'); a.set_title('(c) medians (evaluation; dotted: L1 reference)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[3]; a.plot(im, mk('phase2', 'p_open'), color=CAT[0], marker='o', ms=4, lw=1.6, label='phase 2'); a.plot(im, mk('phase3', 'p_open'), color=CAT[1], marker='s', ms=4, lw=1.6, label='phase 3')
a.set_ylim(-0.03, 1.0); a.set_xlabel('iteration'); a.set_ylabel('probability per decision'); a.set_title('(d) gripper opening while the object is held'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[4]; a.plot(im, mk('phase2', 'dV_median'), color=CAT[0], marker='o', ms=4, lw=1.6, label='phase 2'); a.plot(im, mk('phase3', 'dV_median'), color=CAT[1], marker='s', ms=4, lw=1.6, label='phase 3')
a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel('iteration'); a.set_ylabel('V(dropped) - V(held), median'); a.set_title('(e) critic: value of dropping at held states'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[5]; a.plot(it, [h['hand_loss'] for h in H], color=CAT[0], lw=1.2, label='role loss mean |u_ad - u*|^2'); a.plot(it, [h['mis_hold'] ** 2 for h in H], color=CAT[1], lw=1.0, label='hold-window mismatch squared (median)')
a.set_yscale('log'); a.set_xlabel('iteration'); a.set_ylabel('[N^2 m^2]'); a.set_title('(f) hand role loss (training rollouts)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[6]; hd = lambda k: np.array([np.median(e['hand'][k]) for e in E])
a.plot(ie, hd('res_learned'), color=CAT[0], marker='o', ms=4, lw=1.6, label='learned features: residual after calibration elsewhere'); a.plot(ie, hd('res_class'), color=CAT[0], ls='--', lw=1.0, label='class basis + nominal prior (L1 hand)')
a.plot(ie, hd('prior_learned'), color=CAT[1], marker='s', ms=4, lw=1.6, label='learned prior error at the start posture'); a.plot(ie, hd('prior_nominal'), color=CAT[1], ls='--', lw=1.0, label='nominal prior error')
a.plot(ie, hd('prior_zero'), color=CAT[1], ls=':', lw=1.0, label='zero prior error'); a.set_yscale('log'); a.set_xlabel('iteration'); a.set_ylabel('median [N m]'); a.set_title('(g) hand features (evaluation episodes)'); a.legend(frameon=False, fontsize=7); style(a)
a = ax[7]; tb = [h['tank_bound_min'] if h['tank_bound_min'] is not None else np.nan for h in H]
a.plot(it, np.maximum(tb, 1e-6), color=CAT[0], lw=1.2, label='min tank-bound margin [J]'); a.plot(it, np.array([h['viol'] for h in H]) + 1e-6, color=CAT[1], lw=1.2, label='joint violations (count, +1e-6)')
a.plot(it, [h['gate'] for h in H], color=CAT[2], lw=1.0, label='gate active (fraction of time)'); a.set_yscale('log'); a.set_xlabel('iteration'); a.set_title('(h) guarantees on every iterate (training)'); a.legend(frameon=False, fontsize=7.5); style(a)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
