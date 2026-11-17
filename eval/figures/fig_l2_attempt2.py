"""Learning curves and component diagnostics of L2 attempt 2. Usage: python3 fig_l2_attempt2.py RUN_DIR OUT_PNG [L1_SUMMARY_JSON] [pre=EARLIER_RUN_DIR]
pre: the run the main run was resumed from; its records before the resume iteration are prepended (the resume point is marked)."""
import sys, glob, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

POS = [a for a in sys.argv[1:] if '=' not in a]; OPT = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
RUN, OUT = POS[0], POS[1]
H = json.load(open(RUN + '/hist.json')); E = json.load(open(RUN + '/evals.json')); RES = []
DS = sorted((int(f.split('_it')[1][:4]), json.load(open(f))) for f in glob.glob(RUN + '/dissect_it*.json'))
if 'pre' in OPT:
    i0 = H[0]['it'] - 1; RES = [i0]; PRE = OPT['pre']
    H = [h for h in json.load(open(PRE + '/hist.json')) if h['it'] <= i0] + H; E = [e for e in json.load(open(PRE + '/evals.json')) if e['it'] < i0] + E
    DS = sorted([(int(f.split('_it')[1][:4]), json.load(open(f))) for f in glob.glob(PRE + '/dissect_it*.json') if int(f.split('_it')[1][:4]) <= i0] + DS)
L1 = dict(hold=0.0014, frel=0.032, cost=None)
if len(POS) > 2:
    S = json.load(open(POS[2]))['stats']; L1.update(cost=S['b1_cost_ref'], hold=S['a4_pos2_med'], frel=S['a3_frel_med'])
INK2, MUTED, GRID, AXIS, SURF = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6); a.set_xlabel('iteration')
    for r in RES: a.axvline(r, color=MUTED, ls=':', lw=0.9)
it = np.array([h['it'] for h in H]); ie = np.array([e['it'] for e in E]); nan = lambda v: np.nan if v is None else v
ev = lambda tag, k, f=np.mean: np.array([f(np.asarray(e[tag][k], float)) for e in E])
def cls_contact(tag, nmin=5):
    out, n = [], []
    for e in E:
        c = (np.asarray(e[tag]['grasp']) > 0.5) & (np.asarray(e[tag]['fbar']) > 0); n.append(int(c.sum())); out.append(np.mean(np.asarray(e[tag]['class_ok'])[c]) if c.sum() >= nmin else np.nan)
    return np.array(out), np.array(n)

fig, ax = plt.subplots(2, 4, figsize=(18, 7.8)); ax = ax.ravel()
a = ax[0]; a.plot(it, [h['cost'] for h in H], color=CAT[0], lw=1.2, label='training (exploring)'); a.plot(ie, ev('nom', 'cost'), color=CAT[1], marker='o', ms=3.5, lw=1.6, label='evaluation (deterministic)')
if L1['cost']: a.axhline(L1['cost'], color=MUTED, ls='--', lw=0.9, label='L1 reference brain')
a.set_yscale('log'); a.set_ylabel('mean episode cost'); a.set_title('(a) task cost (revision 14)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[1]; cc, nc = cls_contact('nom'); cb, nb = cls_contact('blind')
a.plot(ie, ev('nom', 'grasp'), color=CAT[0], marker='o', ms=3.5, lw=1.6, label='grasp at the end of phase 1'); a.plot(ie, nc / 64, color=CAT[2], marker='^', ms=3.5, lw=1.4, label='held, contact in the force window')
a.plot(ie, cc, color=CAT[1], marker='s', ms=3.5, lw=1.6, label='class decision (5 or more such episodes)'); a.plot(ie, cb, color=CAT[3], marker='D', ms=3, lw=1.2, label='class decision, learned load hidden')
a.set_ylim(-0.05, 1.05); a.set_title('(b) grasp, contact, class decision (evaluation)'); a.legend(frameon=False, fontsize=7.2); style(a)
a = ax[2]; a.plot(ie, ev('nom', 'pos2', np.median), color=CAT[0], marker='o', ms=3.5, lw=1.6, label='hold error [m]'); a.plot(ie, ev('nom', 'frel', np.median), color=CAT[1], marker='s', ms=3.5, lw=1.6, label='relative force error')
a.axhline(L1['hold'], color=CAT[0], ls=':', lw=0.9); a.axhline(L1['frel'], color=CAT[1], ls=':', lw=0.9); a.set_yscale('log'); a.set_title('(c) medians (evaluation; dotted: L1 reference)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[3]; a.plot(it, [nan(h.get('p_open_att')) for h in H], color=CAT[0], lw=1.2, label='open probability while held (phases 2-3)'); a.plot(it, [h.get('att3', np.nan) for h in H], color=CAT[1], lw=1.2, label='episodes held in phase 3')
a.plot(it, [h.get('rel_before4', np.nan) for h in H], color=CAT[2], lw=1.0, label='episodes releasing before phase 4'); a.set_ylim(-0.03, 1.03); a.set_title('(d) gripper and holding (training)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[4]; a.plot(it, [h['hand_loss'] for h in H], color=CAT[0], lw=1.2, label='weighted role loss (objective)'); a.plot(it, [h.get('hand_loss_all', np.nan) for h in H], color=CAT[1], lw=1.0, label='unweighted role loss')
a.plot(it, [h['mis_hold'] ** 2 for h in H], color=CAT[2], lw=1.0, label='hold-window mismatch squared (median)'); a.set_yscale('log'); a.set_ylabel('[N^2 m^2]'); a.set_title('(e) hand role loss (training)'); a.legend(frameon=False, fontsize=7.5); style(a)
a = ax[5]; hd = lambda k: np.array([np.median(e['hand'][k]) for e in E])
a.plot(ie, hd('res_learned'), color=CAT[0], marker='o', ms=3.5, lw=1.6, label='learned features: residual after calibration elsewhere'); a.plot(ie, hd('res_class'), color=CAT[0], ls='--', lw=1.0, label='class basis + nominal prior (L1 hand)')
a.plot(ie, hd('prior_learned'), color=CAT[1], marker='s', ms=3.5, lw=1.6, label='learned prior error at the start posture'); a.plot(ie, hd('prior_nominal'), color=CAT[1], ls='--', lw=1.0, label='nominal prior error')
a.set_yscale('log'); a.set_ylabel('median [N m]'); a.set_title('(f) hand features (evaluation)'); a.legend(frameon=False, fontsize=7); style(a)
a = ax[6]
if DS:
    di = [d[0] for d in DS]; g = lambda f: [nan(f(d[1])) for d in DS]
    a.plot(di, g(lambda d: d['channel_reference_brain']['learned_hand']['class_given_contact']), color=CAT[0], marker='o', ms=4, lw=1.6, label='channel test: class decision (reference brain, learned hand)')
    a.plot(di, g(lambda d: d['contact_face']['top_share_held_phase3']), color=CAT[2], marker='^', ms=4, lw=1.4, label='held phase-3 decisions with top-face force')
    a.plot(di, g(lambda d: d['contact_face']['side_share_held_phase3']), color=CAT[1], marker='s', ms=4, lw=1.4, label='held phase-3 decisions with side force only')
    a.axhline(0.95, color=CAT[0], ls=':', lw=0.9); a.axhline(0.25, color=CAT[2], ls=':', lw=0.9)
a.set_ylim(-0.03, 1.03); a.set_title('(g) dissections (dotted: criterion H2, stop-rule level)'); a.legend(frameon=False, fontsize=7); style(a)
a = ax[7]; tb = [h['tank_bound_min'] if h['tank_bound_min'] is not None else np.nan for h in H]
a.plot(it, np.maximum(tb, 1e-8), color=CAT[0], lw=1.2, label='min tank-bound margin [J]'); a.plot(it, np.array([h['viol'] for h in H]) + 1e-8, color=CAT[1], lw=1.2, label='joint violations (count, +1e-8)')
a.plot(it, [h['gate'] for h in H], color=CAT[2], lw=1.0, label='gate active (fraction of time)'); a.set_yscale('log'); a.set_title('(h) guarantees on every iterate (training)'); a.legend(frameon=False, fontsize=7.5); style(a)
fig.tight_layout(); fig.savefig(OUT, dpi=170)
