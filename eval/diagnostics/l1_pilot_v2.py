"""Pilot on the revised task (seeds 3000-3199, task version 2; not used for any claim): proportional correction k_c, class decision, budgets."""
import jax; jax.config.update('jax_enable_x64', True)
import sys
import numpy as np, time, json
from pairl.learn import l1
from pairl.stageb import controller as C
cfg = C.default_cfg()
base = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, noqd=1.0, lead_t=0.15)
ID = l1.episodes(3000, 64, lim=0.7); OB = l1.episodes(3100, 64, lim=0.7, ood_body=True)
def run(name, eps, o, E0=None, c=cfg):
    t0 = time.time(); r, _ = l1.batch(l1.with_E0(eps, 60.0 if E0 is None else E0), jax.random.PRNGKey(0), c, 'ref', o)
    md = lambda k: float(np.median(r[k])); p9 = lambda k: float(np.percentile(r[k], 90))
    print(f"{name:22s} {time.time()-t0:5.1f}s cost {np.mean(r['cost']):.3f} grasp {np.mean(r['grasp']):.2f} class {np.mean(r['class_ok'][r['grasp'] > 0.5]):.3f} "
          f"pos2 {md('pos2'):.4f}/{p9('pos2'):.4f} frel {md('frel'):.3f}/{p9('frel'):.3f} gate {np.mean(r['gate']):.3f} tank_bound min {np.min(r['tank_bound']):.4f} "
          f"sat_below {np.sum(r['sat_below'])} viol {np.sum(r['viol'])} Etask {md('E_task'):.2f}", flush=True)
    return r
R = {}
for kc in [0.0, 1.0, 2.0]:
    R[f'id_kc{kc}'] = run(f'ID k_c={kc}', ID, dict(base, k_c=kc))
    R[f'ob_kc{kc}'] = run(f'OOD body k_c={kc}', OB, dict(base, k_c=kc))
run('ID no ledger', ID, dict(base, f_mode=0.0))
Et = R['id_kc0.0']['E_task']
run('BUD (all, rho sampled)', ID, dict(base), E0=[1.0 + t['rho'] * e for (p, t), e in zip(ID, Et)])
