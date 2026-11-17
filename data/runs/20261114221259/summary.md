# 20261114221259: L2 attempt 1 (revision 13); the two aborted runs are the observations behind amendments 1 and 2

- Plan: `records/l2_attempt1_plan.md` (configuration, amendments 1-3 and the diagnosis at 100), sha256 prefix f0bbb550
- Status: complete (stopped at 100 by the diagnosis of the plan); start 2026-11-14T22:12:59, end 2026-11-14T23:37:37
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: stopped at 100: cost defect (release incentive) and role-loss attribution defect

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_attempt1_relgoal_aborted | 1-12 | 37.28 | 46.92 | 6.161 | 0 / 0 / 3.11e-01 |
| l2_attempt1_absgoal_aborted | 1-14 | 33.50 | 32.58 | 5.944 | 0 / 0 / 5.17e-06 |
| l2_attempt1 | 1-74 | 16.83 | 17.43 | 1.276 | 0 / 0 / 2.46e-06 |
| l2_attempt1b | 71-100 | 15.42 | 15.16 | 1.417 | 0 / 0 / 1.90e-03 |

## Images

- [summary.png](summary.png)
- [l2_attempt1_start0_it0070.gif](l2_attempt1_start0_it0070.gif)
- [l2_attempt1_start2_it0070.gif](l2_attempt1_start2_it0070.gif)
- [l2_attempt1_start3_it0070.gif](l2_attempt1_start3_it0070.gif)
- [l2_attempt1b_start0_it0100.gif](l2_attempt1b_start0_it0100.gif)
- [l2_attempt1b_start2_it0100.gif](l2_attempt1b_start2_it0100.gif)
- [l2_attempt1b_start3_it0100.gif](l2_attempt1b_start3_it0100.gif)

- `l2_attempt1_start0_it0070.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 3016 kB
- `l2_attempt1_start2_it0070.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1980 kB
- `l2_attempt1_start3_it0070.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1135 kB
- `l2_attempt1b_start0_it0100.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2412 kB
- `l2_attempt1b_start2_it0100.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2019 kB
- `l2_attempt1b_start3_it0100.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 995 kB
