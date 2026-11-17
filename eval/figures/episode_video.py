"""Rollout videos (evaluation only; deterministic policy, the decision-evaluation episodes and keys, the success definitions of phase_success.py).
One frame layout: left the scene (arm, ghost arm at the realized goal q_d, grasp point p_o, target p* with its 2 cm circle, block and top face, contact
force arrow to scale, object, tip trail colored by phase) with a HUD; right four synchronized time series with a cursor (tip-to-target distance, top-face
force with the class targets and the +-10% band, tank level and gate, ledger payload estimate against the true payload); bottom the phase bar with the
verdict of each phase at its end. Playback at real time (20 frames per second, one frame per 50 ms decision), 0.5x for 1 s around grasp, release and first
top-face contact, a 0.5 s freeze at each phase end, a title card. MP4 (H.264, preset veryfast) and a GIF preview (at most about 5 MB).
Usage:
  python eval/figures/episode_video.py OUTDIR RUN_DIR:IT [starts=0,2,3,4] [only=median_cost,...]  representative episodes per start (success, median cost, failure types)
  python eval/figures/episode_video.py OUTDIR RUN_DIR:IT compare=RUN_DIR:IT [start=0]       side by side, previous (compare) vs current, same episodes
  python eval/figures/episode_video.py OUTDIR RUN_DIR:IT grid=1 [start=0]                  3 x 3 grid of random evaluation episodes with their phase bars
Run on the CPU (JAX_PLATFORMS=cpu), best through script/parallel_run.py."""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import imageio_ffmpeg
from PIL import Image
from pairl import paths
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo
sys.path.insert(0, str(paths.ROOT / 'eval' / 'diagnostics')); import phase_success as PS

INK, INK2, MUTED, GRID, SURF = '#2b2a28', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
PCOL = ['#2a78d6', '#1baf7a', '#eb6834', '#8a55a8']; AMBER, BLOCK, OKC, BADC = '#eda100', '#d8d6cc', '#1b8a5a', '#c0392b'
PHN = ['reach and grasp', 'carry to p*', 'press', 'release and retreat']
FPS, N = 20, PS.N
cfg = C.default_cfg(); tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
loadf = jax.jit(jax.vmap(jax.vmap(lambda q, u: l1.tip_force(q, u)[1] / 9.81)))
chain = lambda q, L: np.vstack([[0, 0], np.cumsum(np.stack([L * np.cos(np.cumsum(q)), L * np.sin(np.cumsum(q))], -1), 0)])


def data(spec, start):
    """Deterministic rollouts of the 256 evaluation episodes from a start; per decision everything the frame needs, and the phase verdicts."""
    run, it = spec.rsplit(':', 1); ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck)
    bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
    eps = PS.ev if start == 0 else [env.restart(p, t, start) for p, t in PS.ev]; pb, tb = env.stack(eps)
    st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, PS.keys, bp, hp, 1.0, 0.0); s = l2.summarize(st, m); V = {k: np.asarray(a) for k, a in dec['viz'].items()}
    q = V['q']; D = dict(q=q, qd=np.asarray(jax.vmap(jax.vmap(l2.goal_of))(dec['u'], dec['x'][..., 0:3] * l2.QH + l2.QC)), xi=V['xi'] > 0.5, fn=V['fn'], fe=V['fe'],
                          com=V['com'], opsi=V['opsi'], T=V['T'], payload=np.asarray(loadf(jnp.asarray(q), jnp.asarray(V['uload']))), psi=np.asarray(dec['psi']) > 0.5,
                          Kp=np.asarray(dec['Kp']), j=np.asarray(dec['j']), v=np.asarray(dec['valid']) > 0.5, p=np.asarray(tipf(jnp.asarray(q), pb['L'])),
                          alpha=np.asarray(rec['alpha']).reshape(N, q.shape[1], -1).min(-1), cost=np.asarray(s['cost']))
    tr = dict(v=D['v'], j=D['j'], xi=D['xi'], p=D['p'], fn=D['fn'], ps=np.asarray(tb['ps']), fstar=np.asarray(tb['fstar']), fl=np.asarray(tb['f_light']), fh=np.asarray(tb['f_heavy']))
    D['ok'] = PS.metrics(tr); D['eps'] = eps; D['spec'] = f"{run.rstrip('/').split('/')[-1]}:{it}"; D['start'] = start
    lo = np.asarray(pb['Lobj'])[:, None, None]; D['gp'] = D['com'] + lo * np.stack([np.cos(D['opsi']), np.sin(D['opsi'])], -1)
    return D


def verdicts(D, e):
    """Phase verdicts (None for phases before the start) and the end decision of each phase."""
    ok = D['ok']; keys = ['p1', 'p2', 'p3', 'p4']; v = D['v'][e]; j = D['j'][e]
    ends = [int(np.where(v & (j == k))[0][-1]) if (v & (j == k)).any() else None for k in range(4)]
    return [(bool(ok[keys[k]][e]) if ends[k] is not None and (k > 0 or D['start'] == 0) else None) for k in range(4)], ends


def failure_types(D, e):
    v, j, xi, d = D['v'][e], D['j'][e], D['xi'][e], np.linalg.norm(D['p'][e] - D['gp'][e], axis=-1); ok = D['ok']; out = []
    free1 = v & (j == 0) & ~xi
    if D['start'] == 0 and not ok['p1'][e]: out.append('never_within_2cm' if not (free1 & (d <= 0.02)).any() else 'within_2cm_too_fast')
    if ok['contact'][e] and not ok['p3c'][e]: out.append('wrong_class')
    w4 = v & (j == 3)
    if w4.any() and xi[np.where(w4)[0][-1]]: out.append('did_not_release')
    dps = np.linalg.norm(D['p'][e] - D['eps'][e][1]['ps'], axis=-1)
    if (w4.any() and not xi[np.where(w4)[0][-1]] and dps[np.where(w4)[0][-1]] > 0.02) or ((v & (j == 1)).any() and not ok['p2'][e] and xi[np.where(v & (j == 1))[0][-1]]): out.append('missed_p_star')
    return out


def select(D):
    """Representative episodes: one success (most phases passed, full success preferred), the median-cost one, the first of each failure type."""
    n_ok = np.array([sum(x for x in verdicts(D, e)[0] if x) for e in range(N)]); e0 = int(np.argmax(n_ok))
    full = all(x for x in verdicts(D, e0)[0] if x is not None); sel = {('success' if full else 'best_most_phases'): e0}       # 'success' only if every phase passed
    sel['median_cost'] = int(np.argsort(D['cost'])[N // 2])
    for e in range(N):
        for f in failure_types(D, e): sel.setdefault(f, e)
    return sel


def schedule(D, e):
    """Frame list: (decision index, kind) with 0.5x around events, freezes at phase ends."""
    v = D['v'][e]; n = int(v.sum()); xi, fn = D['xi'][e], D['fn'][e]
    ev = set(np.where(np.diff(xi[:n].astype(int)) != 0)[0] + 1) | ({int(np.argmax(fn[:n] > 0))} if (fn[:n] > 0).any() else set())
    slow = np.zeros(n, bool)
    for k in ev: slow[max(0, k - 10):k + 10] = True
    ends = set(x for x in verdicts(D, e)[1] if x is not None); fr = []
    for k in range(n):
        fr += [k] * (2 if slow[k] else 1)
        if k in ends: fr += [k] * (FPS // 2)
    return fr


class Episode:
    """Artists of one episode in a scene axes (and optionally the time-series axes and the phase bar)."""
    def __init__(self, D, e, ax, series=None, bar=None, hud=True, small=False):
        self.D, self.e, self.ax = D, e, ax; par, task = D['eps'][e]; self.par, self.task = par, task; L = par['L']; v = D['v'][e]; self.n = int(v.sum())
        pts = np.array([chain(D['q'][e, k], L)[-1] for k in range(self.n)]); allp = np.vstack([pts, [[par['xe'] - par['we'], par['ye'] - 0.08], [par['xe'] + par['we'], 0], [0, 0]], D['gp'][e, :self.n]])
        lo, hi = allp.min(0) - 0.12, allp.max(0) + 0.12; span = (hi - lo).max(); c = (hi + lo) / 2
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2); ax.set_ylim(c[1] - span / 2, c[1] + span / 2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color(GRID)
        ax.add_patch(Rectangle((par['xe'] - par['we'], c[1] - span), 2 * par['we'], par['ye'] - c[1] + span, fc=BLOCK, ec='#a9a79c', lw=0.8, zorder=1))
        ax.plot([par['xe'] - par['we'], par['xe'] + par['we']], [par['ye']] * 2, color=INK2, lw=2.0, zorder=2)
        ax.add_patch(Circle(task['ps'], 0.02, fc='none', ec=OKC, lw=1.0, ls='--', zorder=3)); ax.plot(*task['ps'], marker='x', color=OKC, ms=7, mew=2, zorder=3)
        self.po, = ax.plot([], [], marker='+', color=PCOL[0], ms=10, mew=2, zorder=4); ax.plot(0, 0, marker='s', color=INK, ms=6, zorder=4)
        self.trail = [ax.plot([], [], color=PCOL[k], lw=1.0, alpha=0.7, zorder=2)[0] for k in range(4)]
        self.ghost, = ax.plot([], [], color=MUTED, lw=3.0, alpha=0.35, solid_capstyle='round', zorder=4)
        self.arm, = ax.plot([], [], color=INK, lw=2.6, marker='o', ms=4, zorder=6)
        heavy = task['heavy'] > 0.5; col = '#eb6834' if heavy else '#2a78d6'
        self.obar, = ax.plot([], [], color=col, lw=2.2, zorder=7); self.odisk = Circle((0, 0), 0.016 + 0.012 * par['mo'] / 1.5, fc=col, ec=INK, lw=0.6, zorder=7); ax.add_patch(self.odisk)
        self.grip = Circle((0, 0), 0.011, fc=INK, ec=INK, zorder=8); ax.add_patch(self.grip); self.arrow = None
        self.hud = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=6.5 if small else 8, family='monospace', color=INK,
                           bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec=GRID, alpha=0.9), zorder=9) if hud else None
        self.series, self.bar = series, bar
        t = np.arange(self.n) * P.TH; self.t = t
        if series is not None:
            a0, a1, a2, a3 = series; j = D['j'][e, :self.n]
            tgt = np.where(j[:, None] == 0, D['gp'][e, :self.n], task['ps']); dist = np.linalg.norm(D['p'][e, :self.n] - tgt, axis=-1) * 100
            a0.plot(t, dist, color=INK, lw=1); a0.axhline(2, color=OKC, ls='--', lw=0.8); a0.set_ylabel('tip to target [cm]', fontsize=7); a0.set_yscale('symlog', linthresh=2)
            fs, fl, fh = task['fstar'], task['f_light'], task['f_heavy']
            a1.plot(t, D['fn'][e, :self.n], color=INK, lw=1); a1.axhspan(0.9 * fs, 1.1 * fs, color=OKC, alpha=0.15); a1.axhline(fl, color='#2a78d6', ls=':', lw=0.9); a1.axhline(fh, color='#eb6834', ls=':', lw=0.9)
            a1.set_ylabel('top force [N]', fontsize=7)
            a2.plot(t, D['T'][e, :self.n], color=INK, lw=1); a2.set_ylabel('tank T [J]', fontsize=7); a2b = a2.twinx(); a2b.plot(t, D['alpha'][e, :self.n], color=AMBER, lw=0.9); a2b.set_ylim(-0.05, 1.05)
            a2b.set_ylabel('gate alpha', fontsize=7, color=AMBER); a2b.tick_params(labelsize=6)
            true = np.where(D['xi'][e, :self.n], par['mo'], 0.0)
            a3.plot(t, D['payload'][e, :self.n], color=INK, lw=1, label='ledger'); a3.plot(t, true, color=OKC, lw=1, ls='--', label='true'); a3.set_ylabel('payload [kg]', fontsize=7)
            a3.legend(fontsize=6, frameon=False, loc='upper left'); a3.set_xlabel('t [s]', fontsize=7)
            self.cur = [a.axvline(0, color=PCOL[2], lw=0.8) for a in series]
            for a in series: a.tick_params(labelsize=6); a.grid(color=GRID, lw=0.5); a.set_xlim(0, t[-1])
        if bar is not None:
            vd, ends = verdicts(D, e); self.vd, self.ends = vd, ends; jj = D['j'][e, :self.n]; bar.set_xlim(0, t[-1] + P.TH); bar.set_ylim(0, 1); bar.set_yticks([]); bar.tick_params(labelsize=6)
            for k in range(4):
                idx = np.where(jj == k)[0]
                if idx.size: bar.add_patch(Rectangle((t[idx[0]], 0), (idx[-1] - idx[0] + 1) * P.TH, 1, fc=PCOL[k], alpha=0.18, ec='none'))
            self.prog = bar.add_patch(Rectangle((0, 0), 0, 1, fc=INK, alpha=0.10, ec='none')); self.marks = []

    def draw(self, k):
        D, e, par, task = self.D, self.e, self.par, self.task; L = par['L']; p = chain(D['q'][e, k], L); g = chain(D['qd'][e, k], L)
        self.arm.set_data(p[:, 0], p[:, 1]); self.ghost.set_data(g[:, 0], g[:, 1]); self.grip.center = p[-1]; self.grip.set_facecolor(INK if D['psi'][e, k] else SURF)
        jj = D['j'][e, :k + 1]; tips = D['p'][e, :k + 1]
        for ph in range(4): sel = jj == ph; self.trail[ph].set_data(tips[sel, 0], tips[sel, 1])
        com, gp = D['com'][e, k], D['gp'][e, k]; self.obar.set_data([gp[0], com[0]], [gp[1], com[1]]); self.odisk.center = com
        self.po.set_data([gp[0]], [gp[1]]) if not D['xi'][e, k] else self.po.set_data([], [])
        if self.arrow is not None: self.arrow.remove(); self.arrow = None
        fe = D['fe'][e, k]
        if np.linalg.norm(fe) > 0.05: self.arrow = self.ax.add_patch(FancyArrow(p[-1, 0], p[-1, 1], 0.005 * fe[0], 0.005 * fe[1], width=0.005, head_width=0.016, head_length=0.012, color=AMBER, zorder=8))
        if self.hud is not None:
            heavy = task['heavy'] > 0.5
            self.hud.set_text(f"phase {int(D['j'][e, k]) + 1}/4 {PHN[int(D['j'][e, k])]}\nt {k * P.TH:5.2f} s   {'HELD' if D['xi'][e, k] else 'FREE'}   gripper {'close' if D['psi'][e, k] else 'open'}\n"
                              f"stiffness {np.round(D['Kp'][e, k]).astype(int).tolist()}\nforce {D['fn'][e, k]:5.1f} N  (f* {task['fstar']:.1f})\n"
                              f"[eval only] object {par['mo']:.2f} kg, {'heavy' if heavy else 'light'}")
        if self.series is not None:
            for c in self.cur: c.set_xdata([k * P.TH] * 2)
        if self.bar is not None:
            self.prog.set_width((k + 1) * P.TH)
            for m in self.marks: m.remove()
            self.marks = []
            for ph in range(4):
                if self.vd[ph] is not None and self.ends[ph] is not None and k >= self.ends[ph]:
                    self.marks.append(self.bar.text((self.ends[ph] + 0.5) * P.TH, 0.5, f"P{ph + 1} {'OK' if self.vd[ph] else 'FAIL'}", ha='right', va='center', fontsize=7,
                                                    color='white', bbox=dict(boxstyle='round,pad=0.2', fc=OKC if self.vd[ph] else BADC, ec='none')))


def rgb(fig):
    fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write(frames, path_mp4, title_png=None):
    """MP4 (H.264, veryfast) and a GIF preview of at most about 5 MB."""
    h, w = frames[0].shape[:2]; h2, w2 = h - h % 2, w - w % 2
    wr = imageio_ffmpeg.write_frames(path_mp4, (w2, h2), fps=FPS, codec='libx264', pix_fmt_out='yuv420p', output_params=['-preset', 'veryfast', '-crf', '23'], macro_block_size=1)
    wr.send(None)
    for f in frames: wr.send(np.ascontiguousarray(f[:h2, :w2]))
    wr.close()
    gif = path_mp4[:-4] + '.gif'
    for stride, scale in ((3, 0.5), (4, 0.4), (6, 0.35), (8, 0.3)):
        ims = [Image.fromarray(f).resize((int(w * scale), int(h * scale))).convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames[::stride]]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 * stride / FPS), loop=0, optimize=True)
        if os.path.getsize(gif) <= 5_000_000: break
    return os.path.getsize(path_mp4), os.path.getsize(gif)


def title_frames(lines, size):
    fig = plt.figure(figsize=(size[1] / 100, size[0] / 100), dpi=100, facecolor=SURF)
    fig.text(0.5, 0.5, '\n'.join(lines), ha='center', va='center', fontsize=13, color=INK, family='monospace'); f = rgb(fig); plt.close(fig); return [f] * int(1.5 * FPS)


def episode_video(D, e, path, label):
    plt.rcParams.update({'font.size': 7.5, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); gs = fig.add_gridspec(5, 2, width_ratios=[1.15, 1], height_ratios=[1, 1, 1, 1, 0.28], hspace=0.35, wspace=0.12, left=0.02, right=0.94, top=0.95, bottom=0.05)
    sc = fig.add_subplot(gs[0:4, 0]); ser = [fig.add_subplot(gs[r, 1]) for r in range(4)]; bar = fig.add_subplot(gs[4, :])
    for a in ser[1:]: a.sharex(ser[0])
    par, task = D['eps'][e]; info = f"{D['spec']}  start {D['start']}  episode {e}  ({label})"
    fig.suptitle(info, fontsize=9, color=INK2, x=0.02, ha='left'); E = Episode(D, e, sc, ser, bar)
    frames = []
    for k in schedule(D, e): E.draw(k); frames.append(rgb(fig))
    plt.close(fig)
    tf = title_frames([D['spec'], f"start {D['start']}, episode {e}: {label}", f"object {par['mo']:.2f} kg ({'heavy' if task['heavy'] > 0.5 else 'light'}), f* {task['fstar']:.1f} N",
                       f"block stiffness {par['ke']:.0f} N/m", os.path.basename(path)], frames[0].shape[:2])
    return write(tf + frames, path)


def compare_video(Dp, Dc, e, path):
    fig = plt.figure(figsize=(12.8, 6.0), dpi=100); gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.09], hspace=0.12, wspace=0.04, left=0.01, right=0.99, top=0.93, bottom=0.04)
    A = [Episode(D, e, fig.add_subplot(gs[0, c]), None, fig.add_subplot(gs[1, c]), small=True) for c, D in enumerate((Dp, Dc))]
    for c, D in enumerate((Dp, Dc)): fig.text(0.25 + 0.5 * c, 0.965, ('previous: ' if c == 0 else 'current: ') + D['spec'], ha='center', fontsize=9, color=INK)
    n = max(a.n for a in A); frames = []
    ends = set(x for a in A for x in verdicts(a.D, e)[1] if x is not None)
    for k in range(n):
        for a in A: a.draw(min(k, a.n - 1))
        f = rgb(fig); frames += [f] * (1 + (FPS // 2 if k in ends else 0))
    plt.close(fig); return write(frames, path)


def grid_video(D, idx, path):
    fig = plt.figure(figsize=(12.0, 12.0), dpi=80); gs = fig.add_gridspec(6, 3, height_ratios=[1, 0.12] * 3, hspace=0.15, wspace=0.04, left=0.01, right=0.99, top=0.96, bottom=0.02)
    A = [Episode(D, e, fig.add_subplot(gs[2 * (i // 3), i % 3]), None, fig.add_subplot(gs[2 * (i // 3) + 1, i % 3]), small=True) for i, e in enumerate(idx)]
    fig.suptitle(f"{D['spec']}  start {D['start']}: 9 random evaluation episodes " + str([int(i) for i in idx]), fontsize=10, color=INK)
    n = max(a.n for a in A); frames = []
    for k in range(n):
        for a in A: a.draw(min(k, a.n - 1))
        frames.append(rgb(fig))
    plt.close(fig); return write(frames + [frames[-1]] * FPS, path)


if __name__ == '__main__':
    OUT, SPEC = sys.argv[1], sys.argv[2]; OPT = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a); os.makedirs(OUT, exist_ok=True)
    tag = SPEC.rsplit(':', 1)[0].rstrip('/').split('/')[-1] + '_it' + f"{int(SPEC.rsplit(':', 1)[1]):04d}"; log = []
    if 'compare' in OPT:
        st = int(OPT.get('start', 0)); Dc, Dp = data(SPEC, st), data(OPT['compare'], st); sel = select(Dc)
        for name in [k for k in sel if k in ('success', 'best_most_phases', 'median_cost')]:
            pth = os.path.join(OUT, f"compare_{Dp['spec'].replace(':', '_it')}_vs_{tag}_start{st}_{name}_ep{sel[name]:03d}.mp4"); log.append((pth, compare_video(Dp, Dc, sel[name], pth)))
    elif OPT.get('grid') == '1':
        st = int(OPT.get('start', 0)); D = data(SPEC, st); idx = np.sort(np.random.default_rng(0).choice(N, 9, replace=False))
        pth = os.path.join(OUT, f"grid_{tag}_start{st}.mp4"); log.append((pth, grid_video(D, idx, pth)))
    else:
        for st in [int(x) for x in OPT.get('starts', '0,2,3,4').split(',')]:
            D = data(SPEC, st); sel = select(D); seen = set()
            for name, e in sel.items():
                if e in seen or ('only' in OPT and name not in OPT['only'].split(',')): continue
                seen.add(e); pth = os.path.join(OUT, f"{tag}_start{st}_{name}_ep{e:03d}.mp4"); log.append((pth, episode_video(D, e, pth, name.replace('_', ' '))))
    for pth, (a, b) in log: print('VIDEO', os.path.basename(pth), f'mp4 {a / 1e6:.1f} MB gif {b / 1e6:.1f} MB', flush=True)
