import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Aggregates the RL comparison logs into one record and the figure fig -> ../figures/rl_comparison.png
ARCH = {'A': 'proposed', 'B': 'input policy with funnel filter', 'C': 'proposed without reports',
        'D': 'proposed with learned reflex', 'E': 'proposed with learned reflex, without reports'}
COL = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', 'E': 'C4'}

def load(a, s):
    p = f'../records/rl_{a}_s{s}.jsonl'
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    test = [r for r in rows if r['kind'] == 'test']; train = [r for r in rows if r['kind'] == 'train']
    return test, train

summary = {}
for a in 'ABCDE':
    runs = [x for x in (load(a, s) for s in range(3)) if x is not None and x[0][-1]['it'] == 70]
    if not runs:
        continue
    fin = [t[-1] for t, _ in runs]
    summary[a] = dict(name=ARCH[a], seeds=len(runs),
                      ret=[f['ret'] for f in fin], err=[f['err'] for f in fin],
                      ret0=[t[0]['ret'] for t, _ in runs], err0=[t[0]['err'] for t, _ in runs],
                      test_viol_eps=[f['viol_eps'] for f in fin], test_viol_max=[f['viol_max'] for f in fin],
                      train_eps=[f['train_eps'] for f in fin], train_viol_eps=[f['train_viol_eps'] for f in fin],
                      train_viol_max=[max(r['viol_max'] for r in tr) for _, tr in runs],
                      wmax=[max(r['wmax'] for r in tr + te) for te, tr in runs],
                      train_iters_w_over_1=[sum(r['wmax'] >= 1 for r in tr) for _, tr in runs],
                      y1=[f['y1'] for f in fin], flag=[f['flag'] for f in fin], sat=[f['sat'] for f in fin], eff=[f['eff'] for f in fin],
                      curve_it=[r['it'] for r in runs[0][0]], curve_ret=[[r['ret'] for r in t] for t, _ in runs],
                      curve_err=[[r['err'] for r in t] for t, _ in runs])
json.dump(summary, open('../records/rl_summary.json', 'w'), indent=1)
for a, d in summary.items():
    print(a, d['name'], '| seeds', d['seeds'], '| test ret', np.round(d['ret'], 4), 'mean', round(float(np.mean(d['ret'])), 4),
          '| err', np.round(d['err'], 4), '| test viol', d['test_viol_eps'], '| train viol', d['train_viol_eps'], 'of', d['train_eps'],
          '| max viol', np.round(d['train_viol_max'], 5), '| wmax', np.round(d['wmax'], 4), '| iters w>=1', d['train_iters_w_over_1'], '| y1', np.round(d['y1'], 3), '| flag', np.round(d['flag'], 3))

if summary:
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for a, d in summary.items():
        it = np.array(d['curve_it'])
        for k, (key, axi) in enumerate([('curve_ret', ax[0]), ('curve_err', ax[1])]):
            Y = np.array(d[key])
            for y in Y:
                axi.plot(it, y, color=COL[a], lw=0.6, alpha=0.4)
            axi.plot(it, Y.mean(0), color=COL[a], lw=2.0, label=f"{d['name']} ({d['seeds']} seeds)")
    ax[0].set_title('(a) return on 64 test tasks'); ax[0].set_xlabel('iteration'); ax[0].legend(fontsize=8, loc='lower right')
    ax[1].set_title('(b) mean joint error on 64 test tasks [rad]'); ax[1].set_xlabel('iteration')
    fig.tight_layout(); fig.savefig('../figures/rl_comparison.png', dpi=170)

# Round 2 (pre-registered): steady-state error per seed, split by whether a test task has a target next to a joint limit.
import rl_compare as RC
th_, q0_, tgt_ = RC.sample_tasks(np.random.default_rng(777), RC.N_TEST)
near = ((np.abs(tgt_ - RC.LO) < 0.031) | (np.abs(tgt_ - RC.HI) < 0.031)).any(-1).sum(-1)       # 0, 1 or 2 per task
R2 = {}
for a in ['A', 'C', 'B', 'F']:
    per = []
    for s in range(3):
        p = f'../records/rl2_{a}_s{s}.jsonl'
        if not os.path.exists(p):
            continue
        fin = [json.loads(l) for l in open(p) if '"kind": "test"' in l][-1]
        if fin['it'] != 70:
            continue
        ss = np.array(fin['ss_all'])
        per.append(dict(seed=s, ss=fin['ss'], err=fin['err'], ret=fin['ret'], viol_eps=fin['viol_eps'], train_viol_eps=fin['train_viol_eps'],
                        train_eps=fin['train_eps'], y1=fin['y1'], ss_interior=float(ss[near == 0].mean()), ss_near=float(ss[near > 0].mean()),
                        wmax=max(json.loads(l)['wmax'] for l in open(p))))
    R2[a] = per
pred = dict(P1=all(len(R2[k]) == 3 for k in 'ABF') and all(R2['A'][i]['ss'] < min(R2['F'][i]['ss'], R2['B'][i]['ss']) for i in range(3)),
            P2=not all(R2['A'][i]['ss'] < R2['C'][i]['ss'] for i in range(min(len(R2['A']), len(R2['C'])))),
            P3=all(r['viol_eps'] == 0 and r['train_viol_eps'] == 0 for k in 'AC' for r in R2[k]))
json.dump(dict(round2=R2, predictions=pred, n_tasks_interior=int((near == 0).sum()), n_tasks_near=int((near > 0).sum())),
          open('../records/rl2_summary.json', 'w'), indent=1)
for a, per in R2.items():
    print('round2', a, [(r['seed'], round(r['ss'], 4), round(r['ss_interior'], 4), round(r['ss_near'], 4), round(r['err'], 4), r['viol_eps'], r['train_viol_eps'], round(r['y1'], 3), round(r['wmax'], 4)) for r in per])
print('predictions', pred, '| tasks interior/near:', int((near == 0).sum()), int((near > 0).sum()))
if summary and all(len(v) == 3 for v in R2.values()):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a, d in summary.items():
        it = np.array(d['curve_it']); Y = np.array(d['curve_err'])
        ax[0].plot(it, Y.mean(0), color=COL[a], lw=2.0, label=d['name'])
        for y in Y:
            ax[0].plot(it, y, color=COL[a], lw=0.5, alpha=0.35)
    ax[0].set_title('(a) round 1: mean joint error on test tasks [rad]'); ax[0].set_xlabel('iteration'); ax[0].legend(fontsize=7, loc='upper right')
    names = {'A': 'proposed', 'C': 'proposed,\nno reports', 'B': 'input policy\n+ filter', 'F': 'input policy\n+ filter + integral'}
    for i, a in enumerate(['A', 'C', 'B', 'F']):
        for j, (key, mk, lab) in enumerate([('ss_interior', 'o', 'targets away from limits'), ('ss_near', 's', 'a target next to a limit')]):
            vals = [r[key] for r in R2[a]]
            ax[1].bar(i + (j - 0.5) * 0.36, np.mean(vals), width=0.34, color=['0.75', '0.45'][j], label=lab if i == 0 else None)
            ax[1].scatter(np.full(3, i + (j - 0.5) * 0.36), vals, color='k', s=10, zorder=3, marker=mk)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([names[a] for a in ['A', 'C', 'B', 'F']], fontsize=8)
    ax[1].set_title('(b) round 2: steady-state joint error [rad]'); ax[1].legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig('../figures/rl_comparison.png', dpi=170)
