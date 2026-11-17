"""Animation of four evaluation episodes (deterministic policy) of an L2 checkpoint: arm, commanded goal posture, gripper, object, block, targets, top-face force.
Usage: python3 anim_rollout.py RUN_DIR IT OUT_GIF [cost=3] [every=2] [start=2|3]   (start: the episodes begin from the restart state of env.restart)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, pickle, numpy as np, jax.numpy as jnp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle, FancyArrow
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P

RUN, IT, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
OPT = dict(a.split('=', 1) for a in sys.argv[4:] if '=' in a); env.COST_V = int(OPT.get('cost', 3)); EVERY = int(OPT.get('every', 2))
IDX = [0, 2, 1, 3]                                                         # evaluation episodes: light, heavy, light, heavy (seed 6000)
INK, INK2, MUTED, GRID, SURF = '#2b2a28', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
LIGHT, HEAVY, GREEN, AMBER, BLOCK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#d8d6cc'
PHASE = ['1: reach and grasp', '2: carry to p*', '3: press', '4: release and retreat']

ck = pickle.load(open(f'{RUN}/params_it{IT:04d}.pkl', 'rb')); bp = jax.tree_util.tree_map(jnp.asarray, ck['bp']); hp = jax.tree_util.tree_map(jnp.asarray, ck['hp']); l2.set_mode(ck)
if 'log_std' in bp: bp = l2.upgrade_v1(bp)
cfg = C.default_cfg(); ev_all = l1.with_E0(l1.episodes(6000, 64, lim=0.7), P.E_NOM); ev = [ev_all[i] for i in IDX]
START = int(OPT.get('start', 0))
if START: ev = [env.restart(p, t, START) for p, t in ev]
pb, tb = env.stack(ev)
st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, jax.random.split(jax.random.PRNGKey(555), len(ev)), bp, hp, 1.0, 0.0)
V = {k: np.asarray(a) for k, a in dec['viz'].items()}; U = np.asarray(dec['u']); PSI = np.asarray(dec['psi']); J = np.asarray(dec['j']); VAL = np.asarray(dec['valid']) > 0.5
ZETA = np.asarray(rec['zeta']).mean(-1)                                   # share of the decision interval with the memory updating (rest gate)
MHAT = np.asarray(jax.vmap(jax.vmap(lambda q, u: l1.tip_force(q, u)[1] / 9.81))(jnp.asarray(V['q']), jnp.asarray(V['uload'])))   # ledger load as a tip mass (L1 rule)
QD = np.asarray(jax.vmap(jax.vmap(l2.goal_of))(dec['u'], dec['x'][..., 0:3] * l2.QH + l2.QC))                 # commanded goal postures at the measured postures
def chain(q, L):
    th = np.cumsum(q); pts = np.zeros((4, 2))
    pts[1:] = np.cumsum(np.stack([L * np.cos(th), L * np.sin(th)], -1), 0); return pts
NE, ND = len(ev), V['q'].shape[1]; nend = VAL.sum(1)
pts_all = np.array([[chain(V['q'][e, k], ev[e][0]['L']) for k in range(ND)] for e in range(NE)])
lo = np.minimum(pts_all.reshape(-1, 2).min(0), [min(p['xe'] - p['we'] for p, t in ev), min(p['ye'] for p, t in ev) - 0.1]) - 0.06
hi = np.maximum(pts_all.reshape(-1, 2).max(0), [max(p['xe'] + p['we'] for p, t in ev), max(t['ps'][1] for p, t in ev)]) + 0.06
span = (hi - lo).max(); cen = (hi + lo) / 2; lo, hi = cen - span / 2, cen + span / 2

plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#c3c2b7', 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fig, axs = plt.subplots(2, 2, figsize=(9.6, 10.2)); axs = axs.ravel()
fig.suptitle(f'{RUN.rstrip("/").split("/")[-1]}, iteration {IT}: deterministic policy on four evaluation episodes' + (f', start of phase {START} with the object held' if START else ''), fontsize=10, color=INK)
art = []
for e, ax in enumerate(axs):
    par, task = ev[e]; heavy = task['heavy'] > 0.5; col = HEAVY if heavy else LIGHT
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.grid(True, color=GRID, lw=0.5); ax.tick_params(length=2, labelsize=7)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.add_patch(Rectangle((par['xe'] - par['we'], lo[1] - 1), 2 * par['we'], par['ye'] - lo[1] + 1, fc=BLOCK, ec='#a9a79c', lw=0.8, zorder=1))
    ax.plot(*task['ps'], marker='x', color=GREEN, ms=8, mew=2, zorder=3, label='p* (carry target)')
    ax.plot(task['ps'][0], par['ye'], marker='v', color=GREEN, ms=6, zorder=3, label='p_c (press point)')
    ax.plot(0, 0, marker='s', color=INK, ms=6, zorder=4)
    ax.set_title(f"m_obj {par['mo']:.2f} kg ({'heavy' if heavy else 'light'}): f* = {task['fstar']:.1f} N "
                 f"(light {task['f_light']:.1f}, heavy {task['f_heavy']:.1f})", fontsize=8.5, color=INK2)
    goal, = ax.plot([], [], ls='--', color=MUTED, lw=1.2, zorder=2, label='commanded goal posture')
    trace, = ax.plot([], [], color=INK2, lw=0.6, alpha=0.4, zorder=2)
    arm, = ax.plot([], [], color=INK, lw=2.4, marker='o', ms=4, zorder=5, label='arm')
    grip = Circle((0, 0), 0.012, fc=INK, ec=INK, lw=1.2, zorder=7)
    obar, = ax.plot([], [], color=col, lw=2.0, zorder=6)
    odisk = Circle((0, 0), 0.016 + 0.012 * par['mo'] / 1.5, fc=col, ec=INK, lw=0.6, zorder=6)
    ax.add_patch(grip); ax.add_patch(odisk)
    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=7.5, color=INK, family='monospace',
                  bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec=GRID, alpha=0.85), zorder=9)
    if e == 0: fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7.5, frameon=False)
    art.append(dict(goal=goal, trace=trace, arm=arm, grip=grip, obar=obar, odisk=odisk, txt=txt, ax=ax, arrow=None, col=col))
frames = list(range(0, int(nend.max()), EVERY)) + [int(nend.max()) - 1] * 6

def draw(kd):
    out = []
    for e, a in enumerate(art):
        par, task = ev[e]; k = min(kd, int(nend[e]) - 1); L = par['L']
        p = chain(V['q'][e, k], L); g = chain(QD[e, k], L)
        a['arm'].set_data(p[:, 0], p[:, 1]); a['goal'].set_data(g[:, 0], g[:, 1])
        tr = pts_all[e, max(0, k - 60):k + 1, 3]; a['trace'].set_data(tr[:, 0], tr[:, 1])
        closed = PSI[e, k] > 0.5; a['grip'].center = p[3]; a['grip'].set_facecolor(INK if closed else SURF)
        com = V['com'][e, k]; gp = com + par['Lobj'] * np.array([np.cos(V['opsi'][e, k]), np.sin(V['opsi'][e, k])])
        a['obar'].set_data([gp[0], com[0]], [gp[1], com[1]]); a['odisk'].center = com
        if a['arrow'] is not None: a['arrow'].remove(); a['arrow'] = None
        fe = V['fe'][e, k]; fn = V['fn'][e, k]
        if np.linalg.norm(fe) > 0.05: a['arrow'] = a['ax'].add_patch(FancyArrow(p[3, 0], p[3, 1], 0.006 * fe[0], 0.006 * fe[1], width=0.006, head_width=0.02, head_length=0.015, color=AMBER, zorder=8))
        held = V['xi'][e, k] > 0.5; side = (abs(fe[0]) > 0.05) and (fn <= 0)
        a['txt'].set_text(f"t {k * P.TH:5.2f} s  phase {PHASE[int(J[e, k])]}\n"
                          f"object {'held' if held else 'not held'}, gripper {'closed' if closed else 'open'}\n"
                          f"hand memory {'updating (rest)' if ZETA[e, k] > 0.5 else 'frozen (moving)'}\n"
                          f"ledger load {MHAT[e, k]:5.2f} kg (true {par['mo'] if held else 0.0:.2f})\n"
                          f"top-face force {fn:5.1f} N (target {task['fstar']:.1f}){'  SIDE CONTACT' if side else ''}\n"
                          f"stiffness {np.round(np.asarray(dec['Kp'])[e, k], 0).astype(int).tolist()}")
        out += [a['arm'], a['goal'], a['trace'], a['obar'], a['txt']]
    return out
anim = FuncAnimation(fig, draw, frames=frames, interval=1000 * P.TH * EVERY, blit=False)
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); anim.save(OUT, writer=PillowWriter(fps=round(1 / (P.TH * EVERY))), dpi=72)
print('saved', OUT, len(frames), 'frames')
