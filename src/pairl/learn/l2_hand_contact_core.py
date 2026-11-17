"""Role-loss variants of the offline hand experiments: 'all' equals l2.replay_loss with the role weighting; 'mask' drops contact steps (rec['cf'] = 1);
'reset' also restarts the rest time of the weighting when contact begins or ends."""
import jax, jax.numpy as jnp
from pairl.learn import l2
from pairl.stageb import params as P

def loss_variant(hp, rec, mode, cfgj):
    def one(r):
        def step(c, x):
            z, rr, xp, cp = c
            chg = jnp.abs(x['xi'] - xp) > 0.5
            if mode == 'reset': chg = chg | (jnp.abs(x['cf'] - cp) > 0.5)
            rr = jnp.where(chg, 0.0, rr)
            G = l2.hand_G(hp, x['q']); uad = G.T @ (l2.abar_of(hp) + z); w = 1.0 - jnp.exp(-cfgj['lam_s'] * rr)
            v = x['val'] * (1.0 - x['cf']) if mode == 'mask' else x['val']
            e = v * w * jnp.sum((uad - x['ustar']) ** 2)
            mhat = uad - x['taup'] - x['te']; GtG = G.T @ G; eps = 1e-3 * jnp.trace(GtG) / 3.0 + 1e-9
            dz = G @ jnp.linalg.solve(GtG + eps * jnp.eye(3), mhat)
            return (jnp.clip(z - P.TS * cfgj['lam_s'] * x['zeta'] * dz, -cfgj['zbar'], cfgj['zbar']), rr + P.TS * x['zeta'], x['xi'], x['cf']), e
        _, e = jax.lax.scan(jax.checkpoint(step), (jnp.zeros(l2.NA), jnp.asarray(1e9), jnp.zeros(()), jnp.zeros(())), r)
        return jnp.sum(e) / jnp.maximum(jnp.sum(r['val'] * (1.0 - r['cf']) if mode == 'mask' else r['val']), 1.0)
    flat = jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0], -1) + a.shape[3:]), rec)
    return jnp.mean(jax.vmap(one)(flat))
