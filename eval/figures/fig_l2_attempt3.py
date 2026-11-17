"""Learning curves of L2 attempt 3 (iterations 1-100: arm T of pilot T; 101 on: records/l2_attempt3): task cost, reaching and grasp, carry from the phase-2 start,
pressing from the phase-3 start, hand transfer residual, guarantees. Usage: python3 fig_l2_attempt3.py RUN_DIR PRE_DIR OUT_PNG [L1_SUMMARY_JSON]"""
import sys, glob, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUN, PRE, OUT = sys.argv[1:4]; L1 = json.load(open(sys.argv[4]))['stats'] if len(sys.argv) > 4 else None
H0, E0 = json.load(open(PRE + '/hist.json')), json.load(open(PRE + '/evals.json'))
H1, E1 = json.load(open(RUN + '/hist.json')), json.load(open(RUN + '/evals.json'))
i0 = H1[0]['it'] - 1; H = [h for h in H0 if h['it'] <= i0] + H1; E = [e for e in E0 if e['it'] < i0] + E1
DS = sorted([(int(f.split('_it')[1][:4]), json.load(open(f))) for f in glob.glob(PRE + '/dissect_it*.json') if int(f.split('_it')[1][:4]) <= i0] +
            [(int(f.split('_it')[1][:4]), json.load(open(f))) for f in glob.glob(RUN + '/dissect_it*.json')])
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6); a.set_xlabel('iteration'); a.axvline(i0, color=MUTED, ls=':', lw=0.9)
ie = np.array([e['it'] for e in E]); it = np.array([h['it'] for h in H])
sel = lambda e, tag, k, s: np.asarray(e[tag][k])[np.asarray(e[tag][s]) > 0.5]
med = lambda a: float(np.median(a)) if np.size(a) >= 3 else np.nan
fig, ax = plt.subplots(2, 3, figsize=(14, 7.4)); ax = ax.ravel()
a = ax[0]; a.plot(it, [h['cost'] for h in H], color=CAT[0], lw=1.0, label='training, original start (exploring)')
a.plot(ie, [np.mean(e['nom']['cost']) for e in E], color=CAT[1], marker='o', ms=3.5, lw=1.6, label='evaluation, original start')
if L1: a.axhline(L1['b1_cost_ref'], color=MUTED, ls='--', lw=0.9, label='L1 reference brain')
a.set_yscale('log'); a.set_title('(a) task cost (revision 15)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[1]; a.plot(ie, [np.mean(e['nom']['grasp']) for e in E], color=CAT[0], marker='o', ms=3.5, lw=1.6, label='grasp at the end of phase 1 (evaluation)')
if DS:
    di = [d[0] for d in DS]
    a.plot(di, [d[1]['reach']['frac_within_2cm'] for d in DS], color=CAT[2], marker='^', ms=4.5, lw=1.4, label='nearest distance within 2 cm (dissection)')
    a.plot(di, [d[1]['reach']['frac_slow_when_nearest'] for d in DS], color=CAT[1], marker='s', ms=4.5, lw=1.4, label='slow enough to grasp at the nearest point')
a.set_ylim(-0.03, 1.03); a.set_title('(b) reaching and grasp, original start'); a.legend(frameon=False, fontsize=7.2); style(a)
a = ax[2]; a.plot(ie, [med(sel(e, 'r2', 'pos2', 'held_start3')) for e in E], color=CAT[0], marker='o', ms=3.5, lw=1.6, label='hold error at p*, median of the held [m]')
a.set_ylim(bottom=0); a.set_title('(c) carry, phase-2 start (evaluation)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[3]
acc = lambda e: np.mean((np.asarray(e['r3']['frel']) < 0.5) & (np.asarray(e['r3']['held_start4']) > 0.5))
a.plot(ie, [np.mean(e['r3']['top_window']) for e in E], color=CAT[0], marker='o', ms=3.5, lw=1.6, label='phase-3 start: held, top-face force in the force window')
a.plot(ie, [acc(e) for e in E], color=CAT[1], marker='s', ms=3.5, lw=1.6, label='phase-3 start: accurate press (relative force error < 0.5, held)')
if DS: a.plot([d[0] for d in DS], [d[1]['contact_face']['share_episodes_held_top_force_window'] for d in DS], color=CAT[2], marker='^', ms=4.5, lw=1.4,
              label='original start: held, top-face force in the force window (dissection; stop rule)')
a.axhline(0.25, color=MUTED, ls=':', lw=0.9); a.set_ylim(-0.03, 1.03); a.set_title('(d) pressing (evaluation; dotted: stop-rule threshold)'); a.legend(frameon=False, fontsize=7, loc='upper right'); style(a)
a = ax[4]; a.plot(ie, [np.median(e['hand']['res_learned']) for e in E], color=CAT[0], marker='o', ms=3.5, lw=1.6, label='learned features: residual after calibration elsewhere')
a.axhline(1.768, color=MUTED, ls=':', lw=0.9); a.set_ylabel('median [N m]'); a.set_title('(e) hand: transfer residual (dotted: class basis)'); a.legend(frameon=False, fontsize=7.2); style(a)
a = ax[5]; tb = [h['tank_bound_min'] if h['tank_bound_min'] is not None else np.nan for h in H]
a.plot(it, np.maximum(tb, 1e-10), color=CAT[0], lw=1.0, label='min tank-bound margin [J]'); a.plot(it, np.array([h['viol'] for h in H]) + 1e-10, color=CAT[1], lw=1.0, label='joint violations (count, +1e-10)')
a.set_yscale('log'); a.set_title('(f) guarantees on every iterate (training)'); a.legend(frameon=False, fontsize=7.5); style(a)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
