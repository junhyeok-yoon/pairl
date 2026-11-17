"""V1-V4: kinematics, mass matrix, forward dynamics and gravity against SymPy and MuJoCo (float64)."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np, json, mujoco, sys
from pairl import paths
from pairl.stageb import model as Mo, refs, params
rng = np.random.default_rng(0)
sym = refs.sympy_model()
R = dict(n_param_sets=0, n_states=0, kin_p=0.0, kin_J=0.0, M_sym=0.0, M_mj=0.0, M_asym=0.0, acc_sym=0.0, acc_mj=0.0, acc_mj_force=0.0, grav_sym=0.0, grav_mj=0.0,
         hold=0.0, energy_sym=0.0, M_eig_min=1e9, M_eig_max=0.0, acc_scale=0.0)
acc = jax.jit(Mo.accel); massf = jax.jit(Mo.mass); gravf = jax.jit(Mo.gravity); enf = jax.jit(Mo.energy)
rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / (1.0 + np.max(np.abs(np.asarray(b)))))
for ip in range(240):
    par = params.make_par(rng, ood=(ip % 2 == 1)); xi = float(ip % 4 >= 2)
    pj = {k: jnp.asarray(v) for k, v in par.items()}
    mdl, dat = refs.mujoco_model(par, xi); sid = mdl.site('tip').id
    R['n_param_sets'] += 1
    for _ in range(25):
        q = rng.uniform(-np.pi, np.pi, 3); dq = rng.uniform(-6, 6, 3); tau = rng.uniform(-80, 80, 3); f = rng.uniform(-50, 50, 2)
        a = [*q, *dq, *par['L'], *par['m'], *par['lc'], *par['Icm'], par['Ir'], xi * par['mo'], par['rx'], par['ry'], Mo.G0]
        Ms, bias, ps, Js, gs, KEs, PEs = [np.asarray(x, float) for x in sym(*a)]
        # mine
        Mm = np.asarray(massf(q, pj, xi)); gm = np.asarray(gravf(q, pj, xi)); pm = np.asarray(Mo.tip(jnp.asarray(q), pj['L'])); Jm = np.asarray(Mo.jac(jnp.asarray(q), pj['L']))
        am, _ = acc(q, dq, tau, xi, pj); am = np.asarray(am)
        af, _ = acc(q, dq, tau, xi, pj, jnp.asarray(f)); af = np.asarray(af)
        ke, pe = enf(q, dq, pj, xi)
        # sympy
        a_s = np.linalg.solve(Ms, tau + par['d'] - par['Fv'] * dq - bias.ravel())
        # mujoco
        dat.qpos[:] = q; dat.qvel[:] = dq; dat.qfrc_applied[:] = tau + par['d']; mujoco.mj_forward(mdl, dat)
        a_mj = dat.qacc.copy(); p_mj = dat.site_xpos[sid][:2].copy()
        Mfull = np.zeros((3, 3)); mujoco.mj_fullM(mdl, dat, Mfull)
        jp = np.zeros((3, 3)); mujoco.mj_jacSite(mdl, dat, jp, None, sid)
        dat.qfrc_applied[:] = tau + par['d'] + jp[:2].T @ f; mujoco.mj_forward(mdl, dat); a_mjf = dat.qacc.copy()
        dat.qvel[:] = 0; dat.qfrc_applied[:] = 0; mujoco.mj_forward(mdl, dat); g_mj = dat.qfrc_bias.copy()
        hold, _ = acc(q, np.zeros(3), gm - par['d'], xi, pj)
        for k, v in [('kin_p', max(rel(pm, ps.ravel()), rel(pm, p_mj))), ('kin_J', max(rel(Jm, Js), rel(Jm, jp[:2]))), ('M_sym', rel(Mm, Ms)), ('M_mj', rel(Mm, Mfull)),
                     ('M_asym', float(np.max(np.abs(Mm - Mm.T)))), ('acc_sym', rel(am, a_s)), ('acc_mj', rel(am, a_mj)), ('acc_mj_force', rel(af, a_mjf)),
                     ('grav_sym', rel(gm, gs.ravel())), ('grav_mj', rel(gm, g_mj)), ('hold', float(np.max(np.abs(np.asarray(hold))))),
                     ('energy_sym', max(rel(ke, KEs), rel(pe, PEs))), ('acc_scale', float(np.max(np.abs(am))))]:
            R[k] = max(R[k], v)
        ev = np.linalg.eigvalsh(Mm); R['M_eig_min'] = min(R['M_eig_min'], float(ev[0])); R['M_eig_max'] = max(R['M_eig_max'], float(ev[-1]))
        R['n_states'] += 1
# attachment: momentum conservation and kinetic energy
att = dict(mom=0.0, ke_increase=-1e9, ke_ratio_min=1.0)
for _ in range(2000):
    par = params.make_par(rng, ood=bool(rng.integers(2))); pj = {k: jnp.asarray(v) for k, v in par.items()}
    q = rng.uniform(-np.pi, np.pi, 3); dq = rng.uniform(-3, 3, 3)
    dqp = np.asarray(Mo.attach_velocity(jnp.asarray(q), jnp.asarray(dq), pj))
    Mp, Mm_ = np.asarray(massf(q, pj, 1.0)), np.asarray(massf(q, pj, 0.0))
    # independent check: impulse form with the object-point Jacobian (finite differences of the object position)
    def pobj(qq):
        th = np.cumsum(qq); tp = np.array([np.sum(par['L'] * np.cos(th)), np.sum(par['L'] * np.sin(th))]); c3, s3 = np.cos(th[2]), np.sin(th[2])
        return tp + np.array([c3 * par['rx'] - s3 * par['ry'], s3 * par['rx'] + c3 * par['ry']])
    eps = 1e-6
    Jo = np.stack([(pobj(q + eps * e) - pobj(q - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    dq_ref = np.linalg.solve(Mm_ + par['mo'] * Jo.T @ Jo, Mm_ @ dq)
    att['mom'] = max(att['mom'], rel(dqp, dq_ref))
    kem, kep = 0.5 * dq @ Mm_ @ dq, 0.5 * dqp @ Mp @ dqp
    att['ke_increase'] = max(att['ke_increase'], float(kep - kem)); att['ke_ratio_min'] = min(att['ke_ratio_min'], float(kep / kem))
R['attach'] = att
crit = dict(kin_p=1e-12, kin_J=1e-9, M_sym=1e-12, M_mj=1e-10, M_asym=1e-12, acc_sym=1e-10, acc_mj=1e-9, acc_mj_force=1e-9, grav_sym=1e-12, grav_mj=1e-10, hold=1e-9, energy_sym=1e-12)
R['pass'] = {k: bool(R[k] <= c) for k, c in crit.items()}; R['pass']['attach_momentum'] = bool(att['mom'] <= 1e-7); R['pass']['attach_ke_not_increasing'] = bool(att['ke_increase'] <= 1e-12)
R['criteria'] = crit
print(json.dumps(R, indent=1))
json.dump(R, open(str(paths.RECORDS / 'stageb_dyn_check.json'), 'w'), indent=1)
