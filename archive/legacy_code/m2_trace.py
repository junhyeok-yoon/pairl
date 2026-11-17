import numpy as np, json, jax, jax.numpy as jnp
src = open('m2_diag.py').read()
R = [("if __name__ == '__main__':", "if False:"),
     ("    def rollout(bv, hv, task, noise, ledger_scale, rest_gate=0.0):", "    def rollout(bv, hv, task, noise, ledger_scale, rest_gate=0.0, tfix=-1.0):"),
     ("                obs = jnp.array([t / TEND, q, dq, qd, K / 100, ts / 3, led * T / T0, led * mhat / 3, led * kap, led * alp, led * acon])",
      "                tin = jnp.where(tfix >= 0, tfix, t)\n                obs = jnp.array([tin / TEND, q, dq, qd, K / 100, ts / 3, led * T / T0, led * mhat / 3, led * kap, led * alp, led * acon])"),
     ("            return (x0, x1, z, qd, K, T, tsn, tau, alp, qdp, Kp, kap, acc), None",
      "            ys = jnp.array([t, q, dq, qd, K, ts, T, mhat, kap, alp, acon, uad, gk * jnp.cos(q) - d, (0.0 if A['torque'] else sv), -tc, qdp, Kp])\n            return (x0, x1, z, qd, K, T, tsn, tau, alp, qdp, Kp, kap, acc), ys"),
     ("        c, _ = lax.scan(step, c0, (jnp.arange(NSTEP), noise))", "        c, ys = lax.scan(step, c0, (jnp.arange(NSTEP), noise))"),
     ("        return jnp.array([R, acc[4] / NSTEP,", "        return ys, jnp.array([R, acc[4] / NSTEP,"),
     ("    return bvec, hvec, ev\n", "    return bvec, hvec, ev, rollout, uh\n")]
for a, b in R:
    assert src.count(a) == 1, a[:50]
    src = src.replace(a, b)
ns = {}; exec(compile(src, 'm2_trace', 'exec'), ns)
out = {}
for arm in ('full', 'torque_policy'):
    bv, hv, _, _, _ = ns['build'](arm, jax.random.PRNGKey(0)); out[f'params_{arm}'] = [int(bv.size), int(hv.size)]
print('parameter counts [brain, hand]:', out)
for seed in (1, 2):
    P = np.load(f'../records/m2diag_full_{seed}_params.npz')
    _, _, _, roll, uh = ns['build']('full', jax.random.PRNGKey(seed))
    f = jax.jit(jax.vmap(roll, in_axes=(None, None, 0, 0, None, None, None)))
    b, h = P['b'].astype(np.float32), P['h'].astype(np.float32)
    tasks, noise = ns['sample_tasks'](np.random.default_rng(999), 64)
    ys, met = [np.array(x) for x in f(b, h, tasks, noise, 1.0, 0.0, -1.0)]
    rec = np.array(json.load(open(f'../records/m2_full_{seed}.json'))['eval'])
    _, met_t = f(b, h, tasks, noise, 1.0, 0.0, 1.5); met_t = np.array(met_t)
    t, q, dq, qd, K, ts, T, mh, kap, alp, acon, uad, ust, spr, tc, qdp, Kp = [ys[:, :, i] for i in range(17)]
    con = tc > 1e-6; mov = np.abs(dq) > 0.1
    first = np.array([t[i, np.argmax(con[i])] if con[i].any() else np.nan for i in range(64)])
    vimp = np.array([dq[i, np.argmax(con[i])] if con[i].any() else np.nan for i in range(64)])
    hp = uh(h); G = jax.vmap(lambda x: ns['feats'](hp, x))
    qw, pay, dd = tasks[:, 5], tasks[:, 0], tasks[:, 4]
    gk = 9.81 * (1.5 * 0.2 + pay * 0.4)
    m_prior = np.array(G(jnp.array(qw)) @ hp['abar']) - (gk * np.cos(qw) - dd)
    ss = slice(1000, 1500)
    m_end = (uad - ust)[:, ss].mean(1); p_end = -spr[:, ss].mean(1)
    cc = [np.corrcoef(kap[i, con[i]], -ts[i, con[i]])[0, 1] for i in range(64) if con[i].sum() > 50 and kap[i, con[i]].std() > 1e-3]
    rng_o = np.random.default_rng(777); to, no = ns['sample_tasks'](rng_o, 64); to[:, 0] = rng_o.uniform(2.0, 4.0, 64)
    _, mo = f(b, h, to, no, 1.0, 0.0, -1.0); mo = np.array(mo)
    led = dict(T_over_T0=T / 20, mhat_over_3=mh / 3, kappa=kap, alpha=alp, a_con=acon)
    S = dict(reproduces_record=bool(np.allclose(met, rec, atol=1e-4)),
             ledger_std={k: float(np.std(v)) for k, v in led.items()}, ledger_range={k: [float(v.min()), float(v.max())] for k, v in led.items()},
             T_min=float(T.min()), frac_alpha_lt1=float(np.mean(alp < 1)), frac_acon_pos=float(np.mean(acon > 0)), max_rho_proxy=float(np.max(np.abs((dq + 2 * q - 0.6) / 1.4))),
             first_contact_median=float(np.nanmedian(first)), first_contact_range=[float(np.nanmin(first)), float(np.nanmax(first))], impact_speed_median=float(np.nanmedian(vimp)),
             kappa_free_moving=float(kap[~con & mov].mean()), kappa_free_rest=float(kap[~con & ~mov].mean()) if (~con & ~mov).any() else None, kappa_contact=float(kap[con].mean()),
             corr_kappa_force_in_contact_median=float(np.median(cc)) if cc else None, n_corr=len(cc),
             mismatch_t0=float(np.median(np.abs(uad - ust)[:, 0])), mismatch_t1=float(np.median(np.abs(uad - ust)[:, 500])), mismatch_end=float(np.median(np.abs(m_end))),
             signed_mismatch_end=[float(np.min(m_end)), float(np.median(m_end)), float(np.max(m_end))], spring_push_end=[float(np.min(p_end)), float(np.median(p_end)), float(np.max(p_end))],
             frac_spring_at_cap_in_contact=float(np.mean(np.abs(spr[con]) >= 5.99)),
             prior_only_mismatch=[float(m_prior.min()), float(np.median(m_prior)), float(m_prior.max())], frac_prior_needs_adaptation=float(np.mean((m_prior < -3) | (m_prior > 9))),
             err=float(np.median(np.abs(met[:, 2] - 3))), err_time_frozen=float(np.median(np.abs(met_t[:, 2] - 3))),
             ood_err=float(np.median(np.abs(mo[:, 2] - 3))), ood_mismatch=float(np.median(mo[:, 3])), ood_reached=float(mo[:, 9].mean()), ood_viol=int(mo[:, 8].sum()))
    out[f'full_{seed}'] = S
    print(f'seed {seed}:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items()}))
json.dump(out, open('../records/m2_trace_audit.json', 'w'), indent=1)
