"""Success report of a run group (standing rule of 2026-11-16): writes at the top level of data/runs/<timestamp>/
  success_table.md   rows P1, P2, P3, P3', P4 alone (deterministic; P4 also gripper-sampled), the full chain and the late grasp; columns the L1 reference,
                     attempt 3 at 300, the previous best (outside this group) and the final checkpoint of every run of the group; '*' marks a paired
                     difference against the previous best whose bootstrap 95% CI excludes 0 (eval/diagnostics/phase_diff.py);
  funnel.png         chained funnel (P1, P1&P2, P1-P3, all four) of each final checkpoint next to the L1 reference and the previous best;
  success_curve.png  phase success rates over iterations (records/phase_success_curve.json, deterministic);
and puts the success table at the head of summary.md with links to these files and the videos; the INDEX.md line gets P1 and the full chain.
All phase success numbers come from this PC's CPU (records/phase_success.json). Usage: python eval/figures/success_report.py <timestamp>"""
import sys, os, json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pairl import paths
from pairl.tools import runs
sys.path.insert(0, str(paths.ROOT / 'eval' / 'diagnostics')); import phase_diff

TS = sys.argv[1]; GD = paths.RUNS / TS; M = json.load(open(GD / 'manifest.json'))
PS = json.load(open(paths.RECORDS / 'phase_success.json')); CU = json.load(open(paths.RECORDS / 'phase_success_curve.json')) if (paths.RECORDS / 'phase_success_curve.json').exists() else {}
INK, GRID, SURF = '#2b2b2b', '#e4e2da', '#fbfaf6'; PAL = ['#3b3b3b', '#2f6db5', '#c0562b', '#4f9a4a', '#8a55a8', '#b08a1e']
group_runs = [r for r in M['runs'] if (GD / r / 'hist.json').exists()]
final = {r: f"{r}:{json.load(open(GD / r / 'hist.json'))[-1]['it']}" for r in group_runs}
cur = [final[r] for r in group_runs if final[r] in PS]
rate = lambda k, g, m='P1', md='deterministic': PS[k][md][g][m]['rate']
score = lambda k: (rate(k, 'chain', 'P1234'), rate(k, 'chain', 'P123'), rate(k, 'chain', 'P12'), rate(k, 'alone', 'P1'))
gts = lambda k: paths.run_dir(k.split(':')[0]).parent.name                 # previous best: checkpoints of earlier groups only
others = [k for k in PS if not k.startswith('_') and k != 'L1_reference' and k.split(':')[0] not in group_runs and gts(k) < TS]
prev = max(others, key=score) if others else None
cols = [('L1 reference', 'L1_reference'), ('attempt 3 @300', 'l2_attempt3:300')] + ([('previous best: ' + prev, prev)] if prev and prev != 'l2_attempt3:300' else []) + [(k, k) for k in cur]
cols = [(n, k) for i, (n, k) in enumerate(cols) if k in PS and k not in [x for _, x in cols[:i]]]
ROWS = [('P1 grasp', 'alone', 'P1', 'deterministic'), ('P2 carry (alone)', 'alone', 'P2', 'deterministic'), ('P3 press (alone)', 'alone', 'P3', 'deterministic'),
        ("P3' class (alone, contact)", 'alone', 'P3c', 'deterministic'), ('P4 release (alone)', 'alone', 'P4', 'deterministic'),
        ('P4 release (alone, gripper sampled)', 'alone', 'P4', 'gripper_sampled'), ('chain P1&P2', 'chain', 'P12', 'deterministic'),
        ('chain P1-P3', 'chain', 'P123', 'deterministic'), ('**full chain P1-P4**', 'chain', 'P1234', 'deterministic')]
dkey = {'P1': 'P1', 'P2': 'P2', 'P3': 'P3', 'P3c': 'P3c', 'P4': 'P4', 'P12': 'chain_P12', 'P123': 'chain_P123', 'P1234': 'chain_P1234'}
L = ['# Success table: ' + TS, '', f"Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against "
     f"the previous best ({prev}) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.", '', '| | ' + ' | '.join(n for n, _ in cols) + ' |', '|---' * (len(cols) + 1) + '|']
for name, g, m, md in ROWS:
    cells = []
    for _, k in cols:
        X = PS[k].get(md)
        if X is None: cells.append('-'); continue
        d = X[g][m]; c = f"{d['rate']:.3f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]" if d['rate'] is not None else '-'
        if prev and k in cur and PS[prev].get(md) is not None:
            df = phase_diff.diff(PS, k, prev, md)[dkey[m]]
            if df['ci95'][0] > 0 or df['ci95'][1] < 0: c += f" * ({df['diff']:+.3f})"
        cells.append(c)
    L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
L.append('| late grasp (any time) | ' + ' | '.join(f"{PS[k]['deterministic']['late_grasp']['rate']:.3f}" for _, k in cols) + ' |')
L.append("| P3' contact episodes | " + ' | '.join(str(PS[k]['deterministic']['P3c_contact_episodes']) for _, k in cols) + ' |')
open(GD / 'success_table.md', 'w').write('\n'.join(L) + '\n')

plt.rcParams.update({'font.size': 8, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, ax = plt.subplots(figsize=(9, 3.6)); labs = ['P1', 'P1&P2', 'P1-P3', 'P1-P4']; fk = [k for _, k in cols if k != 'l2_attempt3:300' or k == prev]
w = 0.8 / max(len(fk), 1)
for i, k in enumerate(fk):
    v = [PS[k]['deterministic']['chain'][m]['rate'] for m in ('P1', 'P12', 'P123', 'P1234')]; lo = [PS[k]['deterministic']['chain'][m]['ci95'] for m in ('P1', 'P12', 'P123', 'P1234')]
    x = np.arange(4) + (i - (len(fk) - 1) / 2) * w; ax.bar(x, v, w * 0.95, color=PAL[i % len(PAL)], label=k)
    ax.errorbar(x, v, yerr=[[a - c[0] for a, c in zip(v, lo)], [c[1] - a for a, c in zip(v, lo)]], fmt='none', ecolor=INK, lw=0.7)
ax.set_xticks(range(4), labs); ax.set_ylim(0, 1.05); ax.set_ylabel('success rate (first start, chained)'); ax.grid(axis='y', color=GRID); ax.legend(frameon=False, fontsize=7)
ax.set_title(f'{TS}: chained phase success (deterministic, 256 episodes, 95% CI)', fontsize=8.5, loc='left'); fig.tight_layout(); fig.savefig(GD / 'funnel.png', dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 6, figsize=(16, 3.0), sharey=True)
for i, r in enumerate(group_runs):
    pts = sorted((int(k.split(':')[1]), v) for k, v in CU.items() if k.split(':')[0] == r)
    if not pts: continue
    it = [p[0] for p in pts]
    for a, (g, m) in zip(axs, (('alone', 'P1'), ('alone', 'P2'), ('alone', 'P3'), ('alone', 'P3c'), ('alone', 'P4'), ('chain', 'P1234'))):
        a.plot(it, [p[1]['deterministic'][g][m]['rate'] for p in pts], 'o-', color=PAL[i % len(PAL)], ms=3, lw=1.1, label=r)
TM = {'P1 grasp': ('alone', 'P1'), 'P2 carry': ('alone', 'P2'), 'P3 press': ('alone', 'P3'), "P3' class": ('alone', 'P3c'), 'P4 release': ('alone', 'P4'), 'full chain': ('chain', 'P1234')}
for a, t in zip(axs, TM):
    a.set_title(t, fontsize=8.5, loc='left'); a.grid(color=GRID); a.set_xlabel('iteration'); a.set_ylim(-0.02, 1.02)
    if 'L1_reference' in PS: a.axhline(PS['L1_reference']['deterministic'][TM[t][0]][TM[t][1]]['rate'], color='#b33', ls='--', lw=0.7)
axs[0].legend(frameon=False, fontsize=6.5); axs[0].set_ylabel('success rate (deterministic)')
fig.suptitle(f'{TS}: phase success over iterations (dashed: L1 reference brain)', fontsize=9, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(GD / 'success_curve.png', dpi=120); plt.close(fig)

sm = GD / 'summary.md'; body = open(sm).read() if sm.exists() else ''
if '<!-- success -->' in body: body = body.split('<!-- /success -->', 1)[1]
vids = sorted(os.path.basename(p) for p in glob.glob(str(GD / '*.mp4')))
head = ['<!-- success -->', open(GD / 'success_table.md').read().strip(), '', '- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)']
if vids: head += ['- videos (MP4, GIF preview next to each): ' + ', '.join(f'[{v}]({v})' for v in vids)]
open(sm, 'w').write('\n'.join(head) + '\n<!-- /success -->\n\n' + body.lstrip())
best = max(cur, key=score) if cur else None
if best:
    M['success'] = dict(checkpoint=best, P1=rate(best, 'alone', 'P1'), full_chain=rate(best, 'chain', 'P1234')); json.dump(M, open(GD / 'manifest.json', 'w'), indent=1); runs._write_index(M)
print('success report written:', GD, 'columns', [k for _, k in cols], 'previous best', prev)
