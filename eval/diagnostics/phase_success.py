"""Phase success metrics (evaluation definitions fixed 2026-11-16; they change no training, cost or rule). Deterministic policy, the 256 evaluation episodes
per start and the PRNG keys of the decision evaluation (eval/diagnostics/l2_pilotO_eval.py), evaluated at the decision instants (every 50 ms):
  P1  grasp: attached at the end of phase 1 (first start).
  P2  carry: attached and tip within 2 cm of p* at every decision of the last 0.5 s of phase 2.
  P3  press: attached and top-face force > 0 at every decision of the last 1 s of phase 3, mean normal force within +-10% of f*.
  P3' class: the mean force over the last 1 s of phase 3 is closer to the correct class target than to the other; episodes with top-face contact in that
      window only (count reported).
  P4  release: not attached and tip within 2 cm of p* at every decision of the last 0.5 s of phase 4.
Alone: P1 from start 0, P2 from start 2, P3 and P3' from start 3, P4 from start 4 (env.restart kind 4: at rest at the posture of p*, holding the object,
at the start of phase 4; evaluation only). Chained from start 0: P1, P1&P2, P1&P2&P3, P1&P2&P3&P4; late grasp (attached at any decision) for reference.
Bootstrap 95% CIs (10000 resamples of episodes, numpy default_rng(0), percentile).
Usage: python eval/diagnostics/phase_success.py OUT_JSON SPEC [...]   SPEC = RUN_DIR:IT or ref (the L1 reference brain with the v3 hand)"""
import jax; jax.config.update('jax_enable_x64', True)
import sys, os, json, pickle, numpy as np, jax.numpy as jnp
from pairl.learn import l1, l2
from pairl.stageb import env, controller as C, params as P, model as Mo

N, KEY, NB = 256, 555, 10000
W = {2: 10, 3: 20, 4: 10}                                                 # window lengths in decisions (0.5 s, 1 s, 0.5 s)
env.COST_V = 4; cfg = C.default_cfg()
ev = l1.with_E0(l1.episodes(6000, N, lim=0.7), P.E_NOM); keys = jax.random.split(jax.random.PRNGKey(KEY), N)
tipf = jax.jit(jax.vmap(jax.vmap(Mo.tip, in_axes=(0, None)), in_axes=(0, 0)))
REF = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, noqd=1.0, lead_t=0.15, k_c=1.0, f_mode=1.0)   # L1 reference brain (l2_train.py)
cfgj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}; optj = {k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in REF.items()}
ref_fn = jax.jit(jax.vmap(lambda p, t, k: l1.rollout(p, t, k, l2.ND, cfgj, 'ref', optj, log_every=1)))


def traces(spec, start, mode=1.0):
    """Per decision: valid, phase, attached, tip, top-face force; and the task targets."""
    eps = ev if start == 0 else [env.restart(p, t, start) for p, t in ev]; pb, tb = env.stack(eps)
    if spec == 'ref':
        st, cs, m, lg = ref_fn(pb, tb, keys); sl = lambda a: np.asarray(a)[:, :, 0]                                # (N, ND, NB, ...): state at each decision instant
        k0 = np.arange(l2.ND) * P.NB; edges = np.cumsum(np.asarray(tb['nph']), 1)
        v = k0[None] < edges[:, 3:4]; tr = dict(v=v, j=sl(lg['j']), xi=sl(lg['xi']) > 0.5, p=sl(lg['p']), fn=sl(lg['fn']))
    else:
        run, it = spec.rsplit(':', 1); ck = pickle.load(open(os.path.join(run, f'params_it{int(it):04d}.pkl'), 'rb')); l2.set_mode(ck)
        bp, hp = (jax.tree_util.tree_map(jnp.asarray, ck[k]) for k in ('bp', 'hp'))
        st, m, dec, rec = l2.make_batch_fn(cfg)(pb, tb, keys, bp, hp, mode, 0.0); q = np.asarray(dec['viz']['q'])
        tr = dict(v=np.asarray(dec['valid']) > 0.5, j=np.asarray(dec['j']), xi=np.asarray(dec['viz']['xi']) > 0.5, p=np.asarray(tipf(jnp.asarray(q), pb['L'])), fn=np.asarray(dec['viz']['fn']))
    tr.update(ps=np.asarray(tb['ps']), fstar=np.asarray(tb['fstar']), fl=np.asarray(tb['f_light']), fh=np.asarray(tb['f_heavy']))
    return tr


def window(tr, phase):
    """Boolean (N, ND): the last W[phase] valid decisions of the phase (0-based phase index phase - 1)."""
    m = tr['v'] & (tr['j'] == phase - 1); out = np.zeros_like(m)
    for e in range(m.shape[0]):
        idx = np.where(m[e])[0][-W[phase]:]; out[e, idx] = True
    return out


def metrics(tr):
    dps = np.linalg.norm(tr['p'] - tr['ps'][:, None], axis=-1); every = lambda cond, w: np.all(np.where(w, cond, True), 1) & w.any(1)
    w2, w3, w4 = window(tr, 2), window(tr, 3), window(tr, 4)
    p1e = tr['v'] & (tr['j'] == 0); last1 = np.array([np.where(p1e[e])[0][-1] if p1e[e].any() else -1 for e in range(N)])
    p1 = np.array([tr['xi'][e, i] if i >= 0 else False for e, i in enumerate(last1)])
    p2 = every(tr['xi'] & (dps <= 0.02), w2)
    fmean = np.where(w3, tr['fn'], 0).sum(1) / np.maximum(w3.sum(1), 1)
    p3 = every(tr['xi'] & (tr['fn'] > 0), w3) & (np.abs(fmean - tr['fstar']) <= 0.1 * tr['fstar'])
    contact = np.any(w3 & (tr['fn'] > 0), 1); other = np.where(np.isclose(tr['fstar'], tr['fl']), tr['fh'], tr['fl'])
    p3c = np.abs(fmean - tr['fstar']) < np.abs(fmean - other)
    p4 = every(~tr['xi'] & (dps <= 0.02), w4)
    late = np.any(tr['xi'] & tr['v'], 1)
    return dict(p1=p1, p2=p2, p3=p3, p3c=p3c, contact=contact, p4=p4, late=late)


def rate(x):
    x = np.asarray(x, float)
    if x.size == 0: return dict(n=0, rate=None, ci95=None)
    bs = np.random.default_rng(0).choice(x, (NB, x.size)).mean(1)
    return dict(n=int(x.size), rate=float(x.mean()), ci95=[float(v) for v in np.percentile(bs, [2.5, 97.5])], per_episode=[int(v) for v in x])


def summary(M):
    m0 = M[0]
    o = dict(alone=dict(P1=rate(M[0]['p1']), P2=rate(M[2]['p2']), P3=rate(M[3]['p3']), P3c=rate(M[3]['p3c'][M[3]['contact']]), P4=rate(M[4]['p4'])),
             chain=dict(P1=rate(m0['p1']), P12=rate(m0['p1'] & m0['p2']), P123=rate(m0['p1'] & m0['p2'] & m0['p3']), P1234=rate(m0['p1'] & m0['p2'] & m0['p3'] & m0['p4']),
                        P3c_chain=rate(m0['p3c'][m0['contact'] & m0['p1']])),
             late_grasp=rate(m0['late']), P3c_contact_episodes=int(M[3]['contact'].sum()), P3c_contact_ep=[int(v) for v in M[3]['contact']])
    return o


if __name__ == '__main__':
    OUT = sys.argv[1]
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R['_note'] = ('Phase success metrics (eval/diagnostics/phase_success.py). Always computed on this PC\'s CPU (one platform; JAX_PLATFORMS=cpu, parallel launcher), '
                  'so values here are compared only with each other. Modes: deterministic (mean motion, gripper logit > 0) and gripper_sampled (mean motion, gripper '
                  'Bernoulli with the fixed keys; l2.rollout mode 2). The L1 reference brain has a scripted gripper: its gripper_sampled entry equals the deterministic one.')
    DET_ONLY = 'modes=deterministic' in sys.argv                     # success curves (records/phase_success_curve.json): deterministic only
    for spec in [a for a in sys.argv[2:] if '=' not in a]:
        o = dict(deterministic=summary({s_: metrics(traces(spec, s_, 1.0)) for s_ in (0, 2, 3, 4)}))
        o['gripper_sampled'] = o['deterministic'] if spec == 'ref' else (None if DET_ONLY else summary({s_: metrics(traces(spec, s_, 2.0)) for s_ in (0, 2, 3, 4)}))
        o['device'] = str(jax.devices()[0])
        key = 'L1_reference' if spec == 'ref' else f"{spec.rsplit(':', 1)[0].rstrip('/').split('/')[-1]}:{spec.rsplit(':', 1)[1]}"; R[key] = o
        for md in (('deterministic',) if DET_ONLY else ('deterministic', 'gripper_sampled')):
            print(key, md, json.dumps({g: {k: v['rate'] for k, v in o[md][g].items()} for g in ('alone', 'chain')}), 'late', o[md]['late_grasp']['rate'], flush=True)
        import fcntl                                                     # concurrent writers: lock, re-read, merge this key, write
        with open(str(OUT) + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX); D_ = json.load(open(OUT)) if os.path.exists(OUT) else {}; D_.update({key: o, '_note': R['_note']})
            json.dump(D_, open(str(OUT) + '.tmp', 'w')); os.replace(str(OUT) + '.tmp', OUT)
