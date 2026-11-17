"""Largest contact force at which the sampled authority condition (B3) holds (addendum E0-E3 of records/contact_class_plan.md).
Usage: python contact_authority_limit.py"""
import jax; jax.config.update('jax_enable_x64', True)
import json, numpy as np, jax.numpy as jnp
from pathlib import Path
try:
    from pairl.stageb import model as Mo, params as P; from pairl.tools import design_study as DS
    from pairl import paths; RECORDS = paths.RECORDS
except ImportError:
    import model as Mo, params as P, design_study as DS
    RECORDS = Path(__file__).resolve().parents[1] / 'records'
from contact_bound_constants import REGION

K1, N, SEED, NU = 2.0, 40000, 1, 0.8
_TIP = jax.jit(jax.vmap(lambda q, L: Mo.tip(q, L)))

def limits(ub, ood):
    """Contact-free margin, slope per newton and region mask per sample; the sampling repeats design_study.b3_samples."""
    v0, v1, S = DS.b3_samples(P.QC, P.QH, ub, K1, N, SEED, ood, fbar=1.0, nu=NU)
    rng = np.random.default_rng(SEED); L = np.stack([P.make_par(rng, ood=ood)['L'] for _ in range(N)])
    p = np.asarray(_TIP(jnp.asarray(S['q']), jnp.asarray(L)))
    reg = (p[:, 0] >= REGION[0][0]) & (p[:, 0] <= REGION[0][1]) & (p[:, 1] >= REGION[1][0]) & (p[:, 1] <= REGION[1][1])
    return v0, v1 - v0, reg

def fstar(v0, s, mask):
    m = mask & (s > 0); return float(np.min(-v0[m] / s[m]))

if __name__ == '__main__':
    REC = {}
    for ub in ((120.0, 60.0, 25.0), (150.0, 75.0, 40.0), (180.0, 90.0, 50.0), (240.0, 120.0, 60.0)):
        for ood in (False, True):
            v0, s, reg = limits(ub, ood); key = f"{'-'.join(str(int(u)) for u in ub)}_{'ood' if ood else 'in'}"
            REC[key] = dict(margin_free=float(-v0.max()), margin_60=float(-(v0 + 60.0 * s).max()), fstar_all=fstar(v0, s, np.ones(N, bool)), fstar_region=fstar(v0, s, reg),
                            share_region=float(reg.mean()))
            print(key, {k: round(v, 6) for k, v in REC[key].items()}, flush=True)
    json.dump(REC, open(RECORDS / 'contact_authority_limit.json', 'w'), indent=1)
