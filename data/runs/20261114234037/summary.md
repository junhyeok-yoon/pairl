# 20261114234037: cost and gripper pilots after attempt 1 (iterations 101-130)

- Plan: `records/l2_attempt1_plan.md` (pilots P1-P3), sha256 prefix f0bbb550
- Status: complete; start 2026-11-14T23:40:37, end 2026-11-15T00:41:10
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: led to cost revision 14 and the role-loss weighting

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilot_cost2 | 101-130 | 33.12 | 28.71 | 1.628 | 0 / 0 / 5.01e-04 |
| l2_pilot_cost2_grip | 101-130 | 30.90 | 29.17 | 4.413 | 0 / 0 / 4.55e-04 |
| l2_pilot_v3 | 101-130 | 34.43 | 32.26 | 1.163 | 0 / 0 / 1.88e-04 |

## Images

- [summary.png](summary.png)
- [l2_pilot_cost2_grip_start0_it0130.gif](l2_pilot_cost2_grip_start0_it0130.gif)
- [l2_pilot_cost2_grip_start2_it0130.gif](l2_pilot_cost2_grip_start2_it0130.gif)
- [l2_pilot_cost2_grip_start3_it0130.gif](l2_pilot_cost2_grip_start3_it0130.gif)
- [l2_pilot_cost2_start0_it0130.gif](l2_pilot_cost2_start0_it0130.gif)
- [l2_pilot_cost2_start2_it0130.gif](l2_pilot_cost2_start2_it0130.gif)
- [l2_pilot_cost2_start3_it0130.gif](l2_pilot_cost2_start3_it0130.gif)
- [l2_pilot_v3_start0_it0130.gif](l2_pilot_v3_start0_it0130.gif)
- [l2_pilot_v3_start2_it0130.gif](l2_pilot_v3_start2_it0130.gif)
- [l2_pilot_v3_start3_it0130.gif](l2_pilot_v3_start3_it0130.gif)

- `l2_pilot_cost2_start0_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2105 kB
- `l2_pilot_cost2_start2_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2057 kB
- `l2_pilot_cost2_start3_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1313 kB
- `l2_pilot_cost2_grip_start0_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2710 kB
- `l2_pilot_cost2_grip_start2_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1994 kB
- `l2_pilot_cost2_grip_start3_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1164 kB
- `l2_pilot_v3_start0_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2886 kB
- `l2_pilot_v3_start2_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1968 kB
- `l2_pilot_v3_start3_it0130.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1332 kB
