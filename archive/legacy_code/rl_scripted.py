import json, sys, numpy as np, jax, jax.numpy as jnp
from jax import jit, vmap
import rl_compare as R

# Decisive check: what the lower layer of the proposed architecture achieves on the test tasks under a scripted
# command (proportional steering of the reference to the target), as a function of the speed cap and the feedback gain.
test = R.sample_tasks(np.random.default_rng(777), R.N_TEST)
out = []
for vbar, kc, ga, psip in [(4.0, 15.0, 10.0, 0.05), (4.0, 15.0, 10.0, 0.4), (4.0, 60.0, 10.0, 0.05), (4.0, 60.0, 40.0, 0.05),
                           (6.0, 15.0, 10.0, 0.05), (6.0, 60.0, 40.0, 0.05), (6.0, 60.0, 40.0, 0.2), (6.0, 60.0, 0.0, 0.05)]:
    R.VBAR, R.KC, R.GA = vbar, kc, ga
    def scripted(x, tq, psip=psip):
        vp = 8.0 * (R.K1 * tq - x[4:6]); vp = vp * jnp.minimum(1.0, R.VBAR / jnp.maximum(jnp.linalg.norm(vp), 1e-12))
        return jnp.concatenate([vp, jnp.array([psip])])
    episode, init, ND = R.setup('A', scripted=scripted)
    ev = {k: np.asarray(v) for k, v in jit(vmap(episode, in_axes=(None, 0, 0, 0)))(jnp.asarray(init(0)), *map(jnp.asarray, test)).items()}
    rec = dict(vbar=vbar, kc=kc, ga=ga, psip=psip, ret=float(ev['ret'].mean()), err=float(ev['err'].mean()),
               viol_eps=int((ev['viol'] > 1e-9).sum()), wmax=float(ev['wmax'].max()), yield_mean=float(ev['y1'].mean()),
               yield_active=float(ev['flag'].mean()), sat=float(ev['sat'].mean()), eff=float(ev['eff'].mean()))
    out.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump(out, open('../records/rl_scripted.json', 'w'), indent=1)
