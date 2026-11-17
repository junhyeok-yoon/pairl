"""Test harness for the Stage B environment: rollout with a privileged test controller (not the v3 structure)."""
import jax, jax.numpy as jnp, numpy as np
from pairl.stageb import model as Mo, env, params as P
from pairl.stageb.kin_np import best_margin, ik_all, gauge

KP = np.array([150.0, 100.0, 40.0]); KD = np.array([20.0, 12.0, 5.0])

def alt_posture(p, L, q_ref, dmin=0.4, mmin=0.08):
    """Another posture with the same tip position whose last-link angle differs from that of q_ref by at least dmin (largest margin)."""
    qs, ok = ik_all(p, L); mg = np.where(ok, 1.0 - gauge(qs, P.QC, P.QH), -np.inf)
    d3 = np.abs((np.sum(qs, -1) - np.sum(q_ref) + np.pi) % (2 * np.pi) - np.pi)
    mg = np.where(d3 >= dmin, mg, -np.inf); i = int(np.argmax(mg))
    return (qs[i], float(d3[i])) if mg[i] >= mmin else (np.asarray(q_ref), 0.0)

def add_postures(par, task):
    task = dict(task); task['qo'] = best_margin(task['po'], par['L'], P.QC, P.QH)[1]; task['qs'] = best_margin(task['ps'], par['L'], P.QC, P.QH)[1]
    task['qs2'], task['dth3'] = alt_posture(task['ps'], par['L'], task['qs'])
    return par, task

def psi_test(st, par, task):
    return jnp.where(env.phase(st['k'], task) <= 2, 1.0, 0.0)

def tau_test(st, par, task):
    q, dq = st['q'], st['dq']; j = env.phase(st['k'], task); us = env.u_star(st, par)
    kp, kd = jnp.asarray(KP, q.dtype), jnp.asarray(KD, q.dtype)
    pd = lambda qt: us + kp * (qt - q) - kd * dq
    J = Mo.jac(q, par['L']); p = Mo.tip(q, par['L']); v = J @ dq
    press = us + J.T @ jnp.stack([300.0 * (task['ps'][0] - p[0]) - 40.0 * v[0], -task['fstar'] - 60.0 * v[1]]) - jnp.asarray([3.0, 2.0, 1.0], q.dtype) * dq
    tau = jnp.stack([pd(task['qo']), pd(task['qs']), press, pd(task['qs'])])[j]
    ub = jnp.asarray(P.UB, q.dtype)
    return jnp.clip(tau, -ub, ub)

def frac_in_phase(st, task, j):
    edges = jnp.concatenate([jnp.zeros(1, task['nph'].dtype), jnp.cumsum(task['nph'])])
    return (st['k'] - edges[j]) / task['nph'][j]

def psi_regrasp(st, par, task):
    """Grasp in phase 0; in phase 1 release at 55% and command the grasp again from 80%; hold in phase 2; release in phase 3."""
    j = env.phase(st['k'], task); f = frac_in_phase(st, task, 1)
    return jnp.where(j == 0, 1.0, jnp.where(j == 1, jnp.where((f >= 0.55) & (f < 0.80), 0.0, 1.0), jnp.where(j == 2, 1.0, 0.0)))

def tau_regrasp(st, par, task):
    """tau_test, except that in phase 1 from 60% on the arm moves to another posture with the same tip position."""
    q, dq = st['q'], st['dq']; j = env.phase(st['k'], task); f = frac_in_phase(st, task, 1)
    us = env.u_star(st, par); kp, kd = jnp.asarray(KP, q.dtype), jnp.asarray(KD, q.dtype)
    alt = us + kp * (task['qs2'] - q) - kd * dq
    ub = jnp.asarray(P.UB, q.dtype)
    return jnp.where((j == 1) & (f >= 0.60), jnp.clip(alt, -ub, ub), tau_test(st, par, task))

def rollout(par, task, key, nsteps, nsub=4, tau_fn=tau_test, psi_fn=psi_test, events=True):
    def step(carry, k):
        st, key = carry
        key, ko = jax.random.split(key)
        if events:
            st_ev = env.gripper_event(st, psi_fn(st, par, task), par, task)
            st = jax.tree_util.tree_map(lambda a, b: jnp.where(k % P.NB == 0, a, b), st_ev, st)
        obs = env.observe(st, par, task, ko)
        tau = tau_fn(st, par, task)
        fn, p, v = env.top_force(st, par)
        _, senv = Mo.contact(p, v, par)
        log = dict(q=st['q'], dq=st['dq'], p=p, fn=fn, xi=st['xi'], ph=env.phase(st['k'], task), e=env.e_rob(st, par), sobj=env.s_obj(st, par), w_act=st['w_act'], w_fric=st['w_fric'],
                   w_env=st['w_env'], e_jump=st['e_jump'], senv=senv, cost=env.cost_rate(st, par, task), tau=tau, tau_ext=obs['tau_ext'], tauc=Mo.jac(st['q'], par['L']).T @ Mo.contact(p, v, par)[0])
        return (env.control_step(st, tau, par, nsub), key), log
    (st, _), logs = jax.lax.scan(step, (env.init_state(par, task), key), jnp.arange(nsteps))
    return st, logs
