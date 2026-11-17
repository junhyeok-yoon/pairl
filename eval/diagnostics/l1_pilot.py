"""L1 engineering pilot (not pre-registered, not used for any claim; seeds 3000-3999). Writes records/l1_pilot.json.
A: in-distribution episodes, unrestricted budget: reference brains with and without the realized goal q_d, without memory, stiff, integral action through q_d.
B: heavy objects (1.5-2.5 kg). C: out-of-distribution bodies. D: budgets. E: payload estimate from the learned load."""
import jax; jax.config.update('jax_enable_x64', True)
import json, time, numpy as np
from pairl import paths
from pairl.learn import l1
from pairl.stageb import controller as C, env, params as P, model as Mo

cfg = C.default_cfg(); cfg0 = dict(cfg, lam_s=0.0)
base = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5)
O = dict(ref_qd=(cfg, 'ref', dict(base, noqd=0.0)),
         ref=(cfg, 'ref', dict(base, noqd=1.0, lead_t=0.15)),
         ref_nomem=(cfg0, 'ref', dict(base, noqd=1.0, lead_t=0.15)),
         stiff_nomem=(cfg0, 'ref', dict(base, noqd=1.0, lead_t=0.15, K=np.asarray(cfg['khi']))),
         int_nomem=(cfg0, 'int', dict(base, noqd=0.0, g_p=0.25, g_f=0.6)),
         intr_nomem=(cfg0, 'intr', dict(base, noqd=0.0, g_p=0.3, c_max=0.1, g_f=0.6)),
         slow=(cfg, 'eco', dict(base, noqd=1.0, lead_t=0.15, d_eco=0.1)),
         low=(cfg, 'low', dict(base, noqd=1.0, lead_t=0.15)),
         ref_kc2=(cfg, 'ref', dict(base, noqd=1.0, lead_t=0.15, k_c=2.0)),
         intr_mem=(cfg, 'intr', dict(base, noqd=0.0, g_p=0.3, c_max=0.1, g_f=0.6)),
         ledJ=(cfg, 'ledJ', dict(base, noqd=1.0, lead_t=0.15, d_eco=0.1, theta=1.0)))
R = dict(config={k: dict(brain=b, memory=bool(c['lam_s'] > 0), opts={kk: (np.asarray(vv).tolist()) for kk, vv in o.items()}) for k, (c, b, o) in O.items()})

def run(tag, name, eps, E0=None):
    c, b, o = O[name]
    e2 = l1.with_E0(eps, 1000.0 if E0 is None else E0)
    t0 = time.time(); r, _ = l1.batch(e2, jax.random.PRNGKey(0), c, b, o, obs_qd=True)
    R[tag] = {k: np.asarray(v).tolist() for k, v in r.items()}
    print(f"{tag:22s} {time.time() - t0:5.1f}s cost {np.mean(r['cost']):.3f} grasp {np.mean(r['grasp']):.2f} pos2 p50 {np.median(r['pos2']):.4f} p90 {np.percentile(r['pos2'], 90):.4f} "
          f"frel p50 {np.median(r['frel']):.3f} p90 {np.percentile(r['frel'], 90):.3f} gate {np.mean(r['gate']):.3f} Etask p50 {np.median(r['E_task']):.2f}", flush=True)
    return r

ID = l1.episodes(3000, 64, lim=0.7, v=1)
R['E_pe_save'] = dict(target=[float(l1.pe_arm_np(t['qs_b']) - l1.pe_arm_np(t['qs_l'])) for p, t in ID],
                      grasp=[float(l1.pe_arm_np(t['qo_b']) - l1.pe_arm_np(t['qo_l'])) for p, t in ID])      # nominal arm, best-margin minus minimum-energy posture
for n in ['ref_qd', 'ref', 'ref_nomem', 'stiff_nomem', 'int_nomem', 'intr_nomem', 'slow', 'low']:
    run(f'A_{n}', n, ID)
rng = np.random.default_rng(3600)
HV = [(dict(p, mo=np.float64(rng.uniform(1.5, 2.5))), t) for p, t in [l1.prep(*env.sample_episode(np.random.default_rng(3500 + i), v=1), lim=0.7) for i in range(64)]]
for n in ['ref_qd', 'ref', 'int_nomem', 'intr_nomem']:
    run(f'B_{n}', n, HV)
OB = l1.episodes(3100, 64, lim=0.7, ood_body=True, v=1)
for n in ['ref', 'ref_kc2', 'intr_mem']:
    run(f'C_{n}', n, OB)
Et = np.asarray(R['A_ref']['E_task'])
for rho in [0.4, 0.6, 0.8, 1.0]:
    for n in ['ref', 'slow', 'ledJ']:
        run(f'D_{n}_off{rho}', n, ID, cfg['dT'] + rho * Et)
for rho in [1.2, 2.0]:
    run(f'D_ref_prop{rho}', 'ref', ID, rho * Et)
R['E_mo'] = [float(p['mo']) for p, t in ID]
R['E_hard'] = [float(t['hard']) for p, t in ID]
import sys
json.dump(R, open(sys.argv[1] if len(sys.argv) > 1 else str(paths.RECORDS / 'l1_pilot.json'), 'w'))
