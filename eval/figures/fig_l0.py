"""Figure data and figure for the L0 section: calibration over time (law N vs spec law, three stiffnesses), minimum tank level under small budgets (two gates), generalization across postures."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, sys, json, matplotlib
from pairl import paths
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pairl.stageb import env, params as P, controller as C, model as Mo
from pairl.learn import l0
def cfg_(**kw):
    c = C.default_cfg(); c.update(law=2.0, lam_s=5.0, v_r=0.3, klo=np.array([50.0, 25.0, 10.0]), khi=np.array([400.0, 200.0, 80.0]), kb=2.0, gate_sign=0.0); c.update(kw); return c
eps = [l0.prep(*env.sample_episode(np.random.default_rng(s), v=1)) for s in range(2600, 2632)]
D = {}
for law, lam, ln in ((0.0, 2.0, 'spec'), (2.0, 5.0, 'N')):
    for Kname in ('klo', 'Kn', 'khi'):
        c = cfg_(law=law, lam_s=lam)
        st, cs, m, lg = l0.batch(eps, jax.random.PRNGKey(19), 2500, c, 'calib', dict(hold=3.0, K=c[Kname]), log_every=1)
        mis, xi = np.asarray(lg['mis']), np.asarray(lg['xi']); g = np.argmax(xi > 0.5, axis=1)
        curves = np.stack([mis[i, g[i]:g[i] + 1450] / mis[i, g[i]] for i in range(len(eps))])
        D[f'{ln}_{Kname}'] = curves
budget = {}
for gs, gn in ((1.0, 'sign'), (0.0, 'level')):
    vals = []
    for i, E0 in enumerate((0.5, 1.0, 2.0, 5.0)):
        e2 = [(p, dict(t, E0=np.float64(E0))) for p, t in [l0.prep(*env.sample_episode(np.random.default_rng(s), v=1)) for s in range(1200 + 32 * i, 1232 + 32 * i)]]
        n = int(max(int(np.sum(e[1]['nph'])) for e in e2)); st, cs, m, _ = l0.batch(e2, jax.random.PRNGKey(12), n, cfg_(gate_sign=gs), 'nominal', dict(v_d=0.15, g_f=0.3))
        vals.append(np.asarray(m['minT']))
    budget[gn] = np.stack(vals)
gen = {}
eps7 = [l0.prep(*env.sample_episode(np.random.default_rng(s), v=1)) for s in range(2400, 2464)]
for feats, name in ((0.0, 'class'), (1.0, 'obj')):
    st, cs, m, _ = l0.batch(eps7, jax.random.PRNGKey(18), 2500, cfg_(feats=feats), 'calib', dict(hold=1.5, K=cfg_()['Kn']))
    st = {k: np.asarray(v) for k, v in st.items()}; m = {k: np.asarray(v) for k, v in m.items()}
    prior = np.array([np.linalg.norm(np.asarray(C.prior_body(jnp.asarray(t['qs']), l0.LNOM, l0.MNOM)) - (np.asarray(Mo.gravity(jnp.asarray(t['qs']), dict({k: jnp.asarray(v) for k, v in p.items()}, rx=jnp.asarray(st['rx'][i]), ry=jnp.asarray(st['ry'][i])), 1.0)) - p['d'])) for i, (p, t) in enumerate(eps7)])
    gen[name] = m['m_B_first'] / prior
np.savez(str(paths.RECORDS / 'l0_fig_data.npz'), **{f'calib_{k}': v for k, v in D.items()}, budget_sign=budget['sign'], budget_level=budget['level'], gen_class=gen['class'], gen_obj=gen['obj'])
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 9.5,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def style(a):
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(length=3, width=0.6)
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
t = np.arange(1450) * P.TS
for k, (Kn_, lab) in enumerate((('klo', 'K = k_lo'), ('Kn', 'K = K_n'), ('khi', 'K = k_hi'))):
    for a, ln, title in ((ax[0], 'spec', '(a) update law of revision 11'), (ax[1], 'N', '(b) normalized update law')):
        med = np.median(D[f'{ln}_{Kn_}'], axis=0); p90 = np.percentile(D[f'{ln}_{Kn_}'], 90, axis=0)
        a.plot(t, med, color=CAT[k], lw=2, label=lab); a.plot(t, p90, color=CAT[k], lw=1, ls='--')
        a.set_title(title); a.set_yscale('log'); a.set_ylim(3e-3, 3); a.set_xlabel('time after the grasp [s]'); a.set_ylabel('mismatch ratio')
for a in ax[:2]:
    a.axhline(0.1, color=MUTED, lw=0.8, ls=':'); style(a)
ax[0].legend(frameon=False, fontsize=7.5, loc='upper right', title='solid: median, dashed: 90th pct.', title_fontsize=7)
a = ax[2]; xs = np.arange(4); rng = np.random.default_rng(0)
for j, (gn, col, lab) in enumerate((('sign', CAT[1], 'gate with sign exception'), ('level', CAT[0], 'gate on tank level only'))):
    for i in range(4):
        y = budget[gn][i]; a.scatter(i + (j - 0.5) * 0.3 + rng.uniform(-0.08, 0.08, y.size), y, s=10, color=col, alpha=0.8, label=lab if i == 0 else None)
a.axhline(1.0, color=MUTED, lw=0.8, ls='--'); a.annotate('T_min', (3.45, 1.0), xytext=(0, 3), textcoords='offset points', color=INK2, ha='right')
a.axhline(0.0, color=INK, lw=0.8); a.set_xticks(xs); a.set_xticklabels(['0.5', '1', '2', '5']); a.set_xlabel('energy budget E0 [J]'); a.set_ylabel('minimum tank level [J]')
a.set_title('(c) tank under small budgets'); a.legend(frameon=False, fontsize=7.5, loc='lower right'); style(a)
a = ax[3]
for j, (name, col, lab) in enumerate((('class', CAT[0], 'class basis (7)'), ('obj', CAT[2], 'oracle object basis (3)'))):
    y = gen[name]; a.scatter(j + rng.uniform(-0.15, 0.15, y.size), y, s=12, color=col, alpha=0.8)
    a.plot([j - 0.25, j + 0.25], [np.median(y)] * 2, color=INK, lw=1.5)
a.set_xticks([0, 1]); a.set_xticklabels(['class basis (7)', 'oracle object\nbasis (3)']); a.set_ylabel('mismatch at the next posture / prior only')
a.set_title('(d) one calibration, next posture'); style(a)
fig.tight_layout(); fig.savefig(str(paths.FIGURES / 'l0_results.png'), dpi=200)
print(json.dumps({k: float(np.median(v[:, int(1.0 / P.TS)])) for k, v in D.items()}), {k: float(v.min()) for k, v in budget.items()}, {k: float(np.median(v)) for k, v in gen.items()})
