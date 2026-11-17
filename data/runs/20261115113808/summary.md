# 20261115113808: pilot T: action mode (revision 16)

- Plan: `records/l2_pilotT_plan.md`, sha256 prefix 7d7e86dd
- Status: complete; start 2026-11-15T11:38:08, end 2026-11-15T12:42:30
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: task-space action adopted

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotT_J | 1-100 | 44.48 | 43.24 | 1.185 | 0 / 0 / 2.73e-09 |
| l2_pilotT_T | 1-100 | 31.28 | 31.55 | 1.366 | 0 / 0 / 2.57e-04 |

## Images

- [summary.png](summary.png)
- [l2_pilotT_J_start0_it0100.gif](l2_pilotT_J_start0_it0100.gif)
- [l2_pilotT_J_start2_it0100.gif](l2_pilotT_J_start2_it0100.gif)
- [l2_pilotT_J_start3_it0100.gif](l2_pilotT_J_start3_it0100.gif)
- [l2_pilotT_T_start0_it0100.gif](l2_pilotT_T_start0_it0100.gif)
- [l2_pilotT_T_start2_it0100.gif](l2_pilotT_T_start2_it0100.gif)
- [l2_pilotT_T_start3_it0100.gif](l2_pilotT_T_start3_it0100.gif)

## GIFs

- `l2_pilotT_J_start3_it0100.gif`: copied from docs/figures/rollouts/l2_pilotT_J_start3_it0100.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2230 kB
- `l2_pilotT_T_start0_it0100.gif`: copied from docs/figures/rollouts/l2_pilotT_T_start0_it0100.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3906 kB
- `l2_pilotT_T_start2_it0100.gif`: copied from docs/figures/rollouts/l2_pilotT_T_start2_it0100.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3529 kB
- `l2_pilotT_T_start3_it0100.gif`: copied from docs/figures/rollouts/l2_pilotT_T_start3_it0100.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2124 kB
- `l2_pilotT_J_start0_it0100.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 3182 kB
- `l2_pilotT_J_start2_it0100.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2115 kB
