import jax, jax.numpy as jnp, numpy as np, json, sys
assert not jax.config.jax_enable_x64
from pairl.stageb import env, harness as H
from pairl import paths
eps = [H.add_postures(*env.sample_episode(np.random.default_rng(400 + s), v=1)) for s in range(8)]
pb, tb = env.stack(eps); pb = {k: v.astype(jnp.float32) for k, v in pb.items()}; tb = {k: (v if k == 'nph' else v.astype(jnp.float32)) for k, v in tb.items()}
ref = np.load(paths.TMP / 'stageb_x64_ref.npz'); nst = ref['q'].shape[1]
st, lg = jax.jit(jax.vmap(lambda p, t, k: H.rollout(p, t, k, nst, 4)))(pb, tb, jax.random.split(jax.random.PRNGKey(1), 64)[:8])
q, fn, xi = np.asarray(lg['q']), np.asarray(lg['fn']), np.asarray(lg['xi'])
print(json.dumps(dict(dtype=str(q.dtype), q_diff_max=float(np.max(np.abs(q - ref['q']))), q_diff_median_final=float(np.median(np.abs(q[:, -1] - ref['q'][:, -1]))),
                      fn_diff_rms=float(np.sqrt(np.mean((fn - ref['fn']) ** 2))), xi_mismatch_steps=int(np.sum(xi != ref['xi'])))))
