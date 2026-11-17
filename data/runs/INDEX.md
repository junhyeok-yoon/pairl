# data/runs index

One line per group: data/runs/<timestamp>/<run>/; manifest.json in each group. Groups up to 20261115134229 were run on the cloud
(bundle v3) and moved here unchanged on 2026-11-16; originals are in 20261115_v3_brain_hand_full_bundle.zip (records/<run>/).

Checkpoints are not in git; the ones needed for the tracked results are listed with size and sha256 in [records/checkpoints.sha256](../../records/checkpoints.sha256).
Only this index and the PNG summaries at the top of each group folder are tracked.

| timestamp | purpose | plan | status | key result | P1 / full chain (deterministic, CPU) |
|---|---|---|---|---|---|
| [20261114221259](20261114221259/summary.md) | L2 attempt 1 (revision 13); the two aborted runs are the observations behind amendments 1 and 2 | records/l2_attempt1_plan.md (configuration, amendments 1-3 and the diagnosis at 100) | complete (stopped at 100 by the diagnosis of the plan) | stopped at 100: cost defect (release incentive) and role-loss attribution defect | - |
| [20261114234037](20261114234037/summary.md) | cost and gripper pilots after attempt 1 (iterations 101-130) | records/l2_attempt1_plan.md (pilots P1-P3) | complete | led to cost revision 14 and the role-loss weighting | - |
| [20261115010124](20261115010124/summary.md) | L2 attempt 2 (revision 14) | records/l2_attempt2_plan.md (configuration and amendments 1-5) | complete (300) | pressing 0%, (H1) not met | - |
| [20261115021024](20261115021024/summary.md) | cause of the slow reaching in attempt 2 (E1 relzero, E2 cost13, E3 init1_rev14, E4 init1_rev13, E5 init1_rev14_vnorm) | records/l2_attempt2_plan.md (decisive experiments E1-E5) | complete (40 iterations each) | critic return scale identified (E5) | - |
| [20261115102001](20261115102001/summary.md) | pilot R: start-state distribution (revision 15) | records/l2_pilotR_plan.md | complete | rule (P) failed; restart share not adopted there | - |
| [20261115113808](20261115113808/summary.md) | pilot T: action mode (revision 16) | records/l2_pilotT_plan.md | complete | task-space action adopted | - |
| [20261115134229](20261115134229/summary.md) | L2 attempt 3 (continues l2_pilotT_T from 100) | records/l2_attempt3_plan.md | complete (300) | (G) met, (H1) and (H2) not met; evaluation cost 17.37 at 300 (cloud) | 0.137 / 0.000 (l2_attempt3:300) |
| [20261116012508](20261116012508/summary.md) | pilot O: optimization of the coupled learning | records/l2_pilotO_plan.md | complete | adopted N and H; amendment 1: NH adopted (NH minus N -1.735 [-2.028, -1.474]); K, D not adopted | 0.152 / 0.000 (l2_pilotO_NH:200) |
| [20261116140005](20261116140005/summary.md) | pilot L: transfer term L_tr in training | records/l2_pilotL_plan.md | superseded | arm C only, without the deterministic XLA flag and before the pilot-G code; not used for decisions (pilot L amendment 1); kept as a descriptive comparison | - |
| [20261116145109](20261116145109/summary.md) | pilot L: transfer term L_tr in training (hand role) | records/l2_pilotL_plan.md | complete | no L_tr arm adopted (role loss ratio 1.243, 1.205 > 1.10); residual 0.58-0.69 vs C 6.08; BASE = l2_pilotLd_C at 260 | 0.359 / 0.000 (l2_pilotLd_L1:260) |
| [20261116165912](20261116165912/summary.md) | pilot G: grasp gate (brain, phase 1) and ledger channel (hand to brain) | records/l2_pilotG_plan.md | complete | nothing adopted: S, S2 lower grasp; B passes class decision, class gap, P3prime, cost but not P3 (+0.016 [-0.020, +0.051]) | 0.461 / 0.008 (l2_pilotG_B:320) |
| [20261116200435](20261116200435/summary.md) | pilot L amendment 2: L_tr weight 0.25 (hand role) | records/l2_pilotL_plan.md | complete | L025 not adopted: role loss ratio 1.177 > 1.10; residual 0.863 vs C 6.081; chain stops before attempt 4 | 0.293 / 0.000 (l2_pilotLd_L025:260) |
| 20261117004739 | attempt 4: integrated configuration I against control N | records/l2_attempt4_plan.md | running |  | - |
