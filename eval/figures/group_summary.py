"""Visual summary of a run group (rule of 2026-11-16): writes, at the top level of data/runs/<timestamp>/,
  summary.png                 learning curves of all runs of the group,
  eval_<run>_it<NNNN>.png     decision evaluation of each run's final checkpoint, if it is in records/l2_pilotO_eval.json (pilot O and later),
  decision.png                pilots with a control arm: paired difference arm minus C with the bootstrap 95% CI,
  summary.md                  purpose, plan, status, key numbers and links; data/runs/INDEX.md links to it.
GIFs (<run>_start{0,2,3}_it<NNNN>.gif) are made by script/group_gifs.sh. Usage: python eval/figures/group_summary.py <timestamp>"""
import sys, os, json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pairl import paths

TS = sys.argv[1]; GD = paths.RUNS / TS; M = json.load(open(GD / 'manifest.json'))
RUNS = [r for r in M['runs'] if (GD / r / 'hist.json').exists()]
INK, GRID, SURF = '#2b2b2b', '#e4e2da', '#fbfaf6'
PAL = ['#3b3b3b', '#2f6db5', '#c0562b', '#4f9a4a', '#8a55a8', '#b08a1e', '#2a9d9d']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
style = lambda a, t: (a.set_title(t, fontsize=8, color=INK, loc='left'), a.grid(color=GRID, lw=0.6), [a.spines[s].set_visible(False) for s in ('top', 'right')])
sm = lambda y, k=5: np.convolve(np.asarray(y, float), np.ones(k) / k, mode='valid') if len(y) >= k else np.asarray(y, float)
label = lambda r: r.replace('l2_pilotO_', '').replace('l2_', '')
EVR = json.load(open(paths.RECORDS / 'l2_pilotO_eval.json')) if (paths.RECORDS / 'l2_pilotO_eval.json').exists() else {}


def load(r):
    H = json.load(open(GD / r / 'hist.json')); E = json.load(open(GD / r / 'evals.json')) if (GD / r / 'evals.json').exists() else []
    return H, E


fig, ax = plt.subplots(3, 3, figsize=(12, 9)); ax = ax.ravel()
for i, r in enumerate(RUNS):
    H, E = load(r); c = PAL[i % len(PAL)]; it = np.array([h['it'] for h in H])
    g = lambda k: np.array([np.nan if h.get(k) is None else h[k] for h in H], float)
    ax[0].plot(it, g('cost'), color=c, alpha=0.2, lw=0.7); ax[0].plot(it[len(it) - len(sm(g('cost'))):], sm(g('cost')), color=c, lw=1.3, label=label(r))
    ie = [e['it'] for e in E]
    for k, (tag, ls) in enumerate((('nom', '-'), ('r2', '--'), ('r3', ':'))):
        if E and tag in E[0]: ax[1].plot(ie, [np.mean(e[tag]['cost']) for e in E], ls, color=c, lw=1.1, marker='o', ms=2)
    if 'kl' in H[0]: ax[2].plot(it, g('kl'), color=c, lw=0.9)
    if 'clipfrac' in H[0]: ax[3].plot(it, g('clipfrac'), color=c, lw=0.9)
    if 'sd_goal' in H[0]: ax[4].plot(it, g('sd_goal'), color=c, lw=1.1)
    if 'sd_K' in H[0]: ax[5].plot(it, g('sd_K'), color=c, lw=1.1)
    if E and 'hand' in E[0]: ax[6].plot(ie, [np.median(e['hand']['res_learned']) for e in E], 'o-', color=c, ms=2.5, lw=1.1)
    ax[7].plot(it, g('tank_bound_min'), color=c, lw=0.9)
    if 'hand_loss_all' in H[0]: ax[8].plot(it, g('hand_loss_all'), color=c, lw=0.9)
if any(ax[2].lines): ax[2].set_yscale('log'); ax[2].axhspan(0.01, 0.02, color=GRID, zorder=0)
ax[6].axhline(1.768, color=INK, ls='--', lw=0.7); ax[7].axhline(-0.01, color='#b33', ls='--', lw=0.7); ax[7].set_yscale('symlog', linthresh=1e-4)
for a, t in zip(ax, ('training cost (5-iteration mean)', 'built-in evaluation cost (64 ep.): original -, phase-2 --, phase-3 :', 'KL per iteration (band 0.01-0.02)',
                     'clip share', 'goal standard deviation', 'stiffness standard deviation', 'hand transfer residual, median [N m] (dashed 1.768)',
                     'minimum tank-bound margin [J] (dashed -0.01)', 'role loss, unweighted (hand_loss_all)')):
    style(a, t); a.set_xlabel('iteration')
ax[0].legend(frameon=False, fontsize=7, ncol=min(4, len(RUNS)))
fig.suptitle(f"{TS}: {M['purpose']}", fontsize=9, color=INK, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(GD / 'summary.png', dpi=130); plt.close(fig)

eval_pngs, finals = [], {}
for i, r in enumerate(RUNS):
    H, E = load(r); last = H[-1]['it']; finals[r] = last; key = f'{r}:{last}'
    if key not in EVR: continue
    R = EVR[key]; ref = EVR.get('l2_attempt3:140') if 'pilotO' in r else None
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    st = ('start0', 'start2', 'start3'); x = np.arange(3)
    for k, (mode, c) in enumerate((('deterministic', '#3b3b3b'), ('fully_sampled', '#9a9890'))):
        ax[0].bar(x + (k - 0.5) * 0.36, [R[s][mode]['mean_cost'] for s in st], 0.34, color=c, label=mode.replace('_', ' '))
    if ref: ax[0].plot(x, [ref[s]['deterministic']['mean_cost'] for s in st], 'o', color='#c0562b', ms=4, label='start checkpoint 140 (det.)')
    ax[0].set_xticks(x, ['original', 'phase-2', 'phase-3']); ax[0].legend(frameon=False, fontsize=6.5)
    vals = [('grasp (original)', R['start0']['deterministic']['grasp'], R['start0']['fully_sampled']['grasp'], ref and ref['start0']['deterministic']['grasp']),
            ('accurate press (phase-3)', R['start3']['deterministic']['accurate_press'], R['start3']['fully_sampled']['accurate_press'], ref and ref['start3']['deterministic']['accurate_press'])]
    for a, (t, d, s_, rv) in zip(ax[1:3], vals):
        a.bar([0, 1], [d, s_], 0.5, color=['#3b3b3b', '#9a9890']); a.set_xticks([0, 1], ['det.', 'sampled']); a.set_ylim(0, max(0.05, 1.2 * max(d, s_, rv or 0)))
        if rv is not None: a.axhline(rv, color='#c0562b', lw=1, ls='--')
        style(a, t)
    he = [R['start2'][m]['hold_error_median'] for m in ('deterministic', 'fully_sampled')]
    ax[3].bar([0, 1], he, 0.5, color=['#3b3b3b', '#9a9890']); ax[3].set_xticks([0, 1], ['det.', 'sampled'])
    if ref: ax[3].axhline(ref['start2']['deterministic']['hold_error_median'], color='#c0562b', lw=1, ls='--')
    style(ax[0], 'mean cost per start (256 episodes each)'); style(ax[3], 'hold error at p* (phase-2), median [m]')
    fig.suptitle(f"{r} at {last}: decision evaluation (hand residual {R['hand']['res_learned_median']:.3f} N m); dashed / orange: start checkpoint 140", fontsize=8.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(); n = f'eval_{r}_it{last:04d}.png'; fig.savefig(GD / n, dpi=130); plt.close(fig); eval_pngs.append(n)

dec_rows = []
ctrl = next((r for r in RUNS if r.endswith('_C')), None)
if ctrl and f'{ctrl}:{finals[ctrl]}' in EVR:
    sys.path.insert(0, str(paths.ROOT / 'eval' / 'diagnostics')); import l2_pilotO_eval as PE
    cH = load(ctrl)[0]; cE = EVR[f'{ctrl}:{finals[ctrl]}']
    for r in RUNS:
        if r == ctrl or f'{r}:{finals[r]}' not in EVR: continue
        arm = r.split('_')[-1]; ref = {'NH': next((x for x in RUNS if x.endswith('_N')), ctrl)}.get(arm, ctrl)   # amendment 1: NH is decided against N
        if f'{ref}:{finals[ref]}' not in EVR: continue
        H = load(r)[0]; d = PE.compare(EVR, f'{r}:{finals[r]}', f'{ref}:{finals[ref]}'); R = EVR[f'{r}:{finals[r]}']; cH = load(ref)[0]; cE = EVR[f'{ref}:{finals[ref]}']
        row = dict(arm=arm if ref == ctrl else f'{arm} vs {ref.split("_")[-1]}', **d, kl_median=float(np.median([h['kl'] for h in H if h['it'] >= H[0]['it'] + 10])),
                   res=R['hand']['res_learned_median'], res_c=cE['hand']['res_learned_median'], role=H[-1]['hand_loss_all'], role_c=cH[-1]['hand_loss_all'])
        role10 = lambda HH: float(np.mean([h['hand_loss_all'] for h in HH[-10:]]))
        if arm.startswith('L'):                                      # pilot L: hand rule, paired with the control at the last iteration, and brain non-inferiority
            row.update(role=role10(H), role_c=role10(cH)); row['adopt'] = bool(row['res'] < row['res_c'] and row['res'] < 1.768 and row['role'] <= 1.10 * row['role_c'] and d['ci95'][1] < 0.5)
        elif arm == 'D': row['adopt'] = bool(row['res'] < 1.768 and row['res'] < row['res_c'] and row['role'] <= 1.10 * row['role_c'])
        else: row['adopt'] = bool(d['rule_cost'] and d['rule_press'] and (arm != 'K' or 0.01 <= row['kl_median'] <= 0.02))   # NH: same rule against N
        dec_rows.append(row)
    if dec_rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        y = np.arange(len(dec_rows))[::-1]
        for yy, rw in zip(y, dec_rows):
            c = '#2f6db5' if rw['adopt'] else '#9a9890'
            ax[0].plot(rw['ci95'], [yy, yy], color=c, lw=2.2); ax[0].plot(rw['mean_diff'], yy, 'o', color=c)
        ax[0].axvline(0, color='#b33', lw=1, ls='--'); ax[0].set_yticks(y, [rw['arm'] for rw in dec_rows])
        style(ax[0], 'cost difference vs reference (C; NH vs N), 95% CI; adopt if CI < 0')
        ax[1].barh(y, [rw['accurate_press_diff'] for rw in dec_rows], color=['#2f6db5' if rw['adopt'] else '#9a9890' for rw in dec_rows])
        ax[1].axvline(-0.05, color='#b33', lw=1, ls='--'); ax[1].set_yticks(y, [rw['arm'] for rw in dec_rows])
        style(ax[1], 'accurate-press difference (threshold -0.05)')
        fig.tight_layout(); fig.savefig(GD / 'decision.png', dpi=130); plt.close(fig)

gifs = sorted(os.path.basename(p) for p in glob.glob(str(GD / '*.gif')))
L = [f"# {TS}: {M['purpose']}", '', f"- Plan: `{M['plan']}`" + (f" ({M['plan_part']})" if M.get('plan_part') else '') + f", sha256 prefix {M['plan_sha256_prefix']}",
     f"- Status: {M['status']}; start {M['start']}, end {M.get('end')}", f"- Platform: {M['platform'].get('device')}, {M['platform'].get('machine') or M['platform'].get('cpu')}",
     f"- Code: {M['code'].get('source') or ('commit ' + str(M['code'].get('git_commit')) + (', dirty, diff ' + M['code']['diff_sha256'][:12] if M['code'].get('dirty') else ''))}"]
if M.get('key_result'): L.append(f"- Key result: {M['key_result']}")
L += ['', '## Runs', '', '| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |', '|---|---|---|---|---|---|']
for r in RUNS:
    H, E = load(r); tb = [h['tank_bound_min'] for h in H if h.get('tank_bound_min') is not None]
    L.append(f"| {r} | {H[0]['it']}-{H[-1]['it']} | {np.mean([h['cost'] for h in H[-5:]]):.2f} | {np.mean(E[-1]['nom']['cost']) if E else float('nan'):.2f} | "
             f"{np.median(E[-1]['hand']['res_learned']) if E and 'hand' in E[-1] else float('nan'):.3f} | {sum(h['viol'] for h in H):.0f} / {sum(h['nan'] for h in H):.0f} / {min(tb) if tb else float('nan'):.2e} |")
if eval_pngs:
    L += ['', '## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)', '', '| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |', '|---|---|---|---|---|---|---|---|']
    keys = ([('l2_attempt3:140', 'start checkpoint 140')] if 'l2_attempt3:140' in EVR and any('pilotO' in r for r in RUNS) else []) + [(f'{r}:{finals[r]}', f'{r}:{finals[r]}') for r in RUNS if f'{r}:{finals[r]}' in EVR]
    for k, lab in keys:
        R = EVR[k]; f = lambda s: f"{R[s]['deterministic']['mean_cost']:.3f} / {R[s]['fully_sampled']['mean_cost']:.3f}"
        L.append(f"| {lab} | {f('start0')} | {f('start2')} | {f('start3')} | {R['start0']['deterministic']['grasp']:.3f} | {R['start3']['deterministic']['accurate_press']:.3f} | "
                 f"{R['start2']['deterministic']['hold_error_median']:.4f} | {R['hand']['res_learned_median']:.3f} |")
if dec_rows:
    L += ['', '## Decision (records/l2_pilotO_plan.md)', '', '| arm | mean diff vs reference | 95% CI | accurate-press diff | KL median | residual arm / C | role loss arm / C | adopted |', '|---|---|---|---|---|---|---|---|']
    for rw in dec_rows:
        L.append(f"| {rw['arm']} | {rw['mean_diff']:+.3f} | [{rw['ci95'][0]:+.3f}, {rw['ci95'][1]:+.3f}] | {rw['accurate_press_diff']:+.3f} | {rw['kl_median']:.4f} | "
                 f"{rw['res']:.3f} / {rw['res_c']:.3f} | {rw['role']:.3f} / {rw['role_c']:.3f} | {'yes' if rw['adopt'] else 'no'} |")
L += ['', '## Images', '', '- [summary.png](summary.png)'] + [f'- [{n}]({n})' for n in eval_pngs] + (['- [decision.png](decision.png)'] if dec_rows else []) + [f'- [{n}]({n})' for n in gifs]
if (GD / 'gifs.md').exists(): L += ['', open(GD / 'gifs.md').read().strip()]
open(GD / 'summary.md', 'w').write('\n'.join(L) + '\n')
idx = paths.RUNS / 'INDEX.md'; s = open(idx).read().split('\n')
s = [l if not l.startswith(f'| {TS} |') or '[summary]' in l else l.replace(f'| {TS} |', f'| [{TS}]({TS}/summary.md) |', 1) for l in s]
open(idx, 'w').write('\n'.join(s))
print('summary written:', GD, 'eval pngs', len(eval_pngs), 'decision rows', len(dec_rows), 'gifs', len(gifs))
