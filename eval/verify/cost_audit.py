"""Audit of the cost rate (writes records/cost_audit.json) on states built from random postures (one evaluation episode's plant): (i) dominance of holding over dropping at the same pose
without top-face force, phases 2 and 3; (ii) slope of the held phase-3 rate with depth along the side wall of the block (zero slope = flat valley); (iii) with top-face force in
phase 3, the share of states where dropping at the same pose is cheaper (revision 14: overpressing)."""
import jax; jax.config.update('jax_enable_x64', True)
import sys, numpy as np, jax.numpy as jnp
from pairl import paths
from pairl.learn import l1
from pairl.stageb import env, params as P, model as Mo
par, task = l1.with_E0(l1.episodes(6000, 1, lim=0.7), P.E_NOM)[0]
pj = {k: jnp.asarray(v) for k, v in par.items()}; tj = {k: jnp.asarray(v) for k, v in task.items()}
st0 = env.init_state(pj, tj); e = np.cumsum(task['nph']); rng = np.random.default_rng(0)
Q = jnp.asarray(np.asarray(P.QC) + np.asarray(P.QH) * rng.uniform(-1, 1, (200000, 3)))
def states(q, k, xi):
    p = Mo.tip(q, pj['L'])
    return dict(st0, q=q, dq=jnp.zeros(3), k=jnp.asarray(k), xi=jnp.asarray(xi), obj_psi=jnp.asarray(0.0), obj_c=p - pj['Lobj'] * jnp.array([1.0, 0.0]))
P_ = np.asarray(jax.vmap(lambda q: Mo.tip(q, pj['L']))(Q)); FN = np.asarray(jax.vmap(lambda q: env.top_force(states(q, 0, 1.0), pj)[0])(Q))
xe, ye, we = par['xe'], par['ye'], par['we']; free = FN <= 0
side = (np.abs(P_[:, 0] - xe) >= we) & (np.abs(P_[:, 0] - xe) <= we + 0.01) & (P_[:, 1] < ye) & (P_[:, 1] > ye - 0.3)
REC = {}
top = FN > 0
for v in (1, 2, 3, 4):
    env.COST_V = v
    f = jax.jit(jax.vmap(lambda q, k, xi: env.cost_rate(states(q, k, xi), pj, tj), in_axes=(0, None, None)))
    out = [f'cost version {v}:']; REC[v] = {}
    for j, k in ((2, int(e[0]) + 5), (3, int(e[1]) + 5)):
        d = np.asarray(f(Q, k, 0.0)) - np.asarray(f(Q, k, 1.0))
        REC[v][f'phase{j}'] = dict(min=float(d[free].min()), share_negative=float(np.mean(d[free] < 0)))
        out.append(f'phase {j}: dropped - held at the same pose (no top force) min {d[free].min():+.3f}, share negative {np.mean(d[free] < 0):.3f}')
    h3 = np.asarray(f(Q, int(e[1]) + 5, 1.0)); dep = ye - P_[side, 1]
    slope = np.polyfit(dep, h3[side], 1)[0]
    out.append(f'held phase-3 rate along the side wall ({side.sum()} states): slope with depth {slope:+.2f} per m, range {h3[side].min():.3f}-{h3[side].max():.3f}')
    REC[v]['side'] = dict(n=int(side.sum()), slope=float(slope), lo=float(h3[side].min()), hi=float(h3[side].max()))
    d3 = np.asarray(f(Q, int(e[1]) + 5, 0.0)) - h3
    REC[v]['top'] = dict(n=int(top.sum()), min=float(d3[top].min()), share_negative=float(np.mean(d3[top] < 0)), force_ratio_median=float(np.median(FN[top] / task['fstar'])))
    out.append(f"top-face force ({top.sum()} states, f/f* median {np.median(FN[top] / task['fstar']):.1f}): dropped - held min {d3[top].min():+.3f}, share negative {np.mean(d3[top] < 0):.3f}")
    print(' | '.join(out))
import json; json.dump(REC, open(str(paths.RECORDS / 'cost_audit.json'), 'w'), indent=1)
