"""L1 confirmatory runs, pre-registered in records/preregistration_l1.md (task version 2, seeds 4000-4999).
Writes records/l1_results.json (per-episode metrics) and records/l1_summary.json (pre-registered statistics and verdicts)."""
import jax; jax.config.update('jax_enable_x64', True)
import json, time, numpy as np
from pairl import paths
from pairl.learn import l1
from pairl.stageb import controller as C, params as P, model as Mo

KC = 1.0                                                                 # proportional task-space correction of the reference brain (pilot choice)
cfg = C.default_cfg(); cfg_nomem = dict(cfg, lam_s=0.0)
REF = dict(l1.default_opts(cfg), v_d=0.15, v_fast=0.3, y_fast=0.045, k_x=0.5, noqd=1.0, lead_t=0.15, k_c=KC, f_mode=1.0)
ARMS = dict(ref=(cfg, REF), noled=(cfg, dict(REF, f_mode=0.0)), nomem=(cfg_nomem, REF), nomem_khi=(cfg_nomem, dict(REF, K=np.asarray(cfg['khi']))))

def two_sided(rng, n): return np.where(rng.random(n) < 0.5, rng.uniform(0.7, 0.9, n), rng.uniform(1.1, 1.3, n))
def set_payload(par, task, mo):
    par = dict(par, mo=np.float64(mo)); task = dict(task, heavy=np.float64(mo > P.M_TH))
    task['fstar'] = task['f_heavy'] if mo > P.M_TH else task['f_light']
    return par, task
def set_masses(par, m):
    return dict(par, m=np.asarray(m, float), Icm=np.asarray(m, float) * par['L'] ** 2 / 12)

ID = l1.episodes(4000, 256, lim=0.7)
r_h = np.random.default_rng(4301)
HEAVY = [set_payload(p, t, r_h.uniform(1.5, 2.5)) for p, t in l1.episodes(4300, 128, lim=0.7)]
BODY = l1.episodes(4500, 128, lim=0.7, ood_body=True)
r_m = np.random.default_rng(4701)
BODY_MASS = [set_payload(set_masses(p, P.M_NOM * two_sided(r_m, 3)), t, r_m.uniform(1.5, 2.5)) for p, t in l1.episodes(4700, 128, lim=0.7)]
BODY_LEN = [(set_masses(p, P.M_NOM), t) for p, t in BODY]
STIFF = l1.episodes(4900, 128, lim=0.7, ood_stiff=True)

R, T0 = {}, time.time()
def run(tag, eps, arm, E0, seed):
    c, o = ARMS[arm]; t0 = time.time()
    r, _ = l1.batch(l1.with_E0(eps, E0), jax.random.PRNGKey(seed), c, 'ref', o)
    R[tag] = {k: np.asarray(v).tolist() for k, v in r.items()}
    print(f"{tag:16s} {time.time() - t0:6.1f}s cost {np.mean(r['cost']):.3f} grasp {np.mean(r['grasp']):.3f} frel {np.median(r['frel']):.3f} viol {np.sum(r['viol']):.0f}", flush=True)
    return r

for arm in ARMS: run(f'ID_{arm}', ID, arm, P.E_NOM, 1)
Et = np.asarray(R['ID_ref']['E_task'])
E0b = [cfg['dT'] + float(t['rho']) * float(e) for (p, t), e in zip(ID, Et)]
run('BUD_ref', ID, 'ref', E0b, 1)
for tag, eps, seed in (('HEAVY', HEAVY, 2), ('BODY', BODY, 3), ('BODY_MASS', BODY_MASS, 4), ('BODY_LEN', BODY_LEN, 3), ('STIFF', STIFF, 5)):
    run(f'{tag}_ref', eps, 'ref', P.E_NOM, seed)
R['ID_hard'] = [float(t['hard']) for p, t in ID]; R['ID_heavy'] = [float(t['heavy']) for p, t in ID]; R['ID_mo'] = [float(p['mo']) for p, t in ID]
R['ID_rho'] = [float(t['rho']) for p, t in ID]; R['BUD_E0'] = E0b
R['config'] = dict(KC=KC, ref_opts={k: np.asarray(v).tolist() for k, v in REF.items()}, seconds=time.time() - T0)
json.dump(R, open(str(paths.RECORDS / 'l1_results.json'), 'w'))

# pre-registered statistics and verdicts
A = lambda k: {m: np.asarray(v) for m, v in R[k].items()}
ref, nol, nom, nomk, bud = A('ID_ref'), A('ID_noled'), A('ID_nomem'), A('ID_nomem_khi'), A('BUD_ref')
hard = np.asarray(R['ID_hard']) > 0.5
g = ref['grasp'] > 0.5; g2 = g & (nol['grasp'] > 0.5)
S = dict(
    a1_grasp=float(np.mean(ref['grasp'])), a2_class=float(np.mean(ref['class_ok'][g])),
    a3_frel_med=float(np.median(ref['frel'])), a3_frel_p90=float(np.percentile(ref['frel'], 90)),
    a4_pos2_med=float(np.median(ref['pos2'])), a4_pos2_p90=float(np.percentile(ref['pos2'], 90)),
    a5_viol=float(sum(np.sum(A(k)['viol']) for k in R if isinstance(R[k], dict) and 'viol' in R[k])), a5_nan=float(sum(np.sum(A(k)['nan']) for k in R if isinstance(R[k], dict) and 'nan' in R[k])),
    b1_cost_ref=float(np.mean(ref['cost'])), b1_cost_nomem=float(np.mean(nom['cost'])), b1_cost_nomem_khi=float(np.mean(nomk['cost'])),
    b2_frel_nomem=float(np.median(nom['frel'])), b2_frel_nomem_khi=float(np.median(nomk['frel'])),
    c1_frel_noled=float(np.median(nol['frel'])), c1_ratio=float(np.median(nol['frel']) / np.median(ref['frel'])),
    c2_press_better=float(np.mean(ref['cost_ph'][g2, 2] < nol['cost_ph'][g2, 2])), c3_class_noled=float(np.mean(nol['class_ok'][nol['grasp'] > 0.5])),
    d1_cons_hard=float(np.mean(ref['cons'][hard])), d1_cons_normal=float(np.mean(ref['cons'][~hard])),
    d2_gate_bud=float(np.mean(bud['gate'])), d2_gate_any=float(np.mean(bud['gate'] > 0)),
    d4_tank_bound_min=float(np.min(bud['tank_bound'][bud['sat_below'] < 0.5])) if np.any(bud['sat_below'] < 0.5) else None, d4_n_sat_below=float(np.sum(bud['sat_below'] > 0.5)),
    d5_guard=float(np.mean(ref['guard'])), d6_sat=float(np.mean(ref['sat'])))
V = dict(a1=S['a1_grasp'] >= 0.95, a2=S['a2_class'] >= 0.95, a3=S['a3_frel_med'] <= 0.05 and S['a3_frel_p90'] <= 0.20, a4=S['a4_pos2_med'] <= 0.005 and S['a4_pos2_p90'] <= 0.02,
         a5=S['a5_viol'] == 0 and S['a5_nan'] == 0,
         b1=S['b1_cost_nomem'] >= 2 * S['b1_cost_ref'] and S['b1_cost_nomem_khi'] >= 2 * S['b1_cost_ref'], b2=S['b2_frel_nomem'] >= 0.25 and S['b2_frel_nomem_khi'] >= 0.25,
         c1=S['c1_ratio'] >= 3.0, c2=S['c2_press_better'] >= 0.80, c3=S['c3_class_noled'] <= 0.65,
         d1=S['d1_cons_hard'] >= 0.30, d2=S['d2_gate_bud'] >= 0.05 and S['d2_gate_any'] >= 0.5, d4=(S['d4_tank_bound_min'] is None) or S['d4_tank_bound_min'] >= -0.01)
json.dump(dict(stats=S, verdict={k: bool(v) for k, v in V.items()}, all_pass=bool(all(V.values()))), open(str(paths.RECORDS / 'l1_summary.json'), 'w'), indent=1)
print(json.dumps(dict(stats=S, verdict={k: bool(v) for k, v in V.items()}), indent=1))
