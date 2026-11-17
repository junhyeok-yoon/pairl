"""Contact integration accuracy by regime, stiffness and substep count (used to choose the substep count)."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, json, sys
from pairl import paths
from pairl.stageb import model as Mo, env, params as P, harness as H
def make_tau(lo, hi, dy):
    def tau(st, par, task):
        q, dq = st['q'], st['dq']; t = st['k'] * P.TS; Jm = Mo.jac(q, par['L']); p = Mo.tip(q, par['L']); v = Jm @ dq
        F = 0.5 * (lo + hi) + 0.5 * (hi - lo) * jnp.sin(2 * jnp.pi * 0.7 * t)
        return env.u_star(st, par) + Jm.T @ jnp.stack([300.0 * (par['xe'] + 0.05 * jnp.sin(jnp.pi * t) - p[0]) - 40.0 * v[0], -F - dy * v[1]]) - jnp.asarray([3.0, 2.0, 1.0]) * dq
    return tau
REG = dict(operating=make_tau(1.0, 15.0, 60.0), harsh=make_tau(-4.0, 20.0, 20.0))
def run(eps, keys, regime, ke, nsub):
    pb, tb = env.stack([(dict(p, ke=np.float64(ke)), t) for p, t in eps])
    st, lg = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, 1500, nsub, REG[regime], events=False)))(pb, tb, keys)
    res = np.abs(np.asarray(lg['e'] - lg['e'][:, :1] - (lg['w_act'] - lg['w_fric'] + lg['w_env']))); sc = np.maximum(1.0, np.max(np.abs(np.asarray(lg['w_act'])), 1, keepdims=True))
    return dict(residual_rel=float(np.max(res / sc)), passivity_excess=float(np.max(np.asarray(lg['w_env'] + lg['senv']))), fn_max=float(np.max(np.asarray(lg['fn']))),
                contact_frac=float(np.mean(np.asarray(lg['fn']) > 0))), np.asarray(st['q']), np.asarray(lg['fn'])
def table(eps, keys, subs=(2, 4, 8, 16)):
    T = {}
    for regime in REG:
        for ke in (2e3, 2e4, 2e5):
            out = {n: run(eps, keys, regime, ke, n) for n in subs}
            for n in subs:
                d = out[n][0]
                if n != subs[-1]:
                    d['q_end_vs_ref'] = float(np.max(np.abs(out[n][1] - out[subs[-1]][1]))); d['fn_rms_vs_ref'] = float(np.sqrt(np.mean((out[n][2] - out[subs[-1]][2]) ** 2)))
                T[f'{regime}_{int(ke)}_{n}'] = d
    return T
if __name__ == '__main__':
    eps = []
    for s in range(16):
        par, task = H.add_postures(*env.sample_episode(np.random.default_rng(200 + s), hard=False, v=1)); task['q0'] = task['qs']; eps.append((par, task))
    T = table(eps, jax.random.split(jax.random.PRNGKey(0), 64)[:16])
    for k, v in T.items(): print(k, {a: float('%.3g' % b) for a, b in v.items()})
    json.dump(T, open(str(paths.RECORDS / 'stageb_contact_runs.json'), 'w'), indent=1)
