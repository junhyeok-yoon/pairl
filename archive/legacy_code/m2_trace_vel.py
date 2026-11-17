import numpy as np, json, jax, jax.numpy as jnp
src = open('m2_diag.py').read()
R = [("if __name__ == '__main__':", "if False:"),
     ("    def rollout(bv, hv, task, noise, ledger_scale, rest_gate=0.0):", "    def rollout(bv, hv, task, noise, ledger_scale, rest_gate=0.0, vel=1.0):"),
     ("z = jnp.clip(z - feats(hp, 0.5 * (q + x0)) * dqk - ", "z = jnp.clip(z - vel * feats(hp, 0.5 * (q + x0)) * dqk - "),
     ("            return (x0, x1, z, qd, K, T, tsn, tau, alp, qdp, Kp, kap, acc), None",
      "            ys = jnp.array([uad - (gk * jnp.cos(q) - d), (0.0 if A['torque'] else sv)])\n            return (x0, x1, z, qd, K, T, tsn, tau, alp, qdp, Kp, kap, acc), ys"),
     ("        c, _ = lax.scan(step, c0, (jnp.arange(NSTEP), noise))", "        c, ys = lax.scan(step, c0, (jnp.arange(NSTEP), noise))"),
     ("        return jnp.array([R, acc[4] / NSTEP,", "        return ys, jnp.array([R, acc[4] / NSTEP,"),
     ("    return bvec, hvec, ev\n", "    return bvec, hvec, ev, rollout\n")]
for a, b in R:
    assert src.count(a) == 1, a[:50]
    src = src.replace(a, b)
ns = {}; exec(compile(src, 'm2_trace_vel', 'exec'), ns)
tasks, noise = ns['sample_tasks'](np.random.default_rng(999), 64)
out = {}
for seed in (1, 2):
    P = np.load(f'../records/m2diag_full_{seed}_params.npz')
    _, _, _, roll = ns['build']('full', jax.random.PRNGKey(seed))
    f = jax.jit(jax.vmap(roll, in_axes=(None, None, 0, 0, None, None, None)))
    for vel, rg in ((1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        ys, met = [np.array(x) for x in f(P['b'].astype(np.float32), P['h'].astype(np.float32), tasks, noise, 1.0, rg, vel)]
        m = ys[:, :, 0]; p = -ys[:, :, 1]
        r = dict(err=float(np.median(np.abs(met[:, 2] - 3))), mismatch_t1=float(np.median(np.abs(m[:, 500]))), signed_mismatch_end=float(np.median(m[:, 1000:].mean(1))),
                 spring_push_end=float(np.median(p[:, 1000:].mean(1))), reached=float(met[:, 9].mean()))
        out[f'seed{seed}_vel{int(vel)}_rest{int(rg)}'] = r
        print(seed, 'velocity term' if vel else 'no velocity term', '| calibration', 'rest only' if rg else 'as trained', json.dumps({k: round(v, 3) for k, v in r.items()}))
a = json.load(open('../records/m2_trace_audit.json')); a['velocity_term_check'] = out
json.dump(a, open('../records/m2_trace_audit.json', 'w'), indent=1)
