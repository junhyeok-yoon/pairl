"""Writes records/stageb_design.json: workspace margins, sampled (B3) margins for the chosen and the rejected torque limits, gravity load ratios."""
import sys
from pairl import paths
import numpy as np, json
from pairl.tools import design_study as DS
from pairl.stageb import params as P
ws, mg = DS.workspace_stats(P.QC, P.QH, band=P.MARG_HARD, normal=P.MARG_NORMAL)
R = dict(box=dict(c=P.QC.tolist(), h=P.QH.tolist()), ub=P.UB.tolist(), k1=2.0, p=P.PG, workspace=ws, b3={}, fbar=60.0, nu_max=0.8, n_samples=40000,
         object_offset_max=P.LOBJ_MAX + P.GRASP_DIST)
for name, ub in (('chosen', tuple(P.UB)), ('rejected_80_40_15', (80.0, 40.0, 15.0))):
    for ood in (False, True):
        v0, v1, _ = DS.b3_samples(P.QC, P.QH, ub, 2.0, 40000, 1, ood)
        R['b3'][f"{name}_{'ood' if ood else 'in'}"] = dict(margin_no_contact=float(-v0.max()), margin_with_contact=float(-v1.max()), frac_negative_with_contact=float(np.mean(v1 >= 0)))
R['gravity_ratio_in'] = [float(x) for x in DS.gravity_ratio(P.QC, P.QH, P.UB, 20000, 2, False)]; R['gravity_ratio_ood'] = [float(x) for x in DS.gravity_ratio(P.QC, P.QH, P.UB, 20000, 2, True)]
json.dump(R, open(str(paths.RECORDS / 'stageb_design.json'), 'w'), indent=1)
print(json.dumps(R['b3']), R['gravity_ratio_in'], R['gravity_ratio_ood'])
