<!-- success -->
# Success table: 20261116012508

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (l2_attempt3:300) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 | l2_pilotO_C:200 | l2_pilotO_N:200 | l2_pilotO_H:200 | l2_pilotO_NH:200 |
|---|---|---|---|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] | 0.023 [0.01, 0.04] * (-0.113) | 0.074 [0.04, 0.11] * (-0.062) | 0.055 [0.03, 0.09] * (-0.082) | 0.152 [0.11, 0.20] |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] | 0.090 [0.05, 0.12] * (-0.246) | 0.141 [0.10, 0.18] * (-0.195) | 0.199 [0.15, 0.25] * (-0.137) | 0.332 [0.27, 0.39] |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] | 0.059 [0.03, 0.09] * (-0.078) | 0.059 [0.03, 0.09] * (-0.078) | 0.098 [0.06, 0.14] | 0.133 [0.09, 0.18] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] | 0.594 [0.50, 0.69] * (+0.123) | 0.626 [0.53, 0.72] * (+0.150) | 0.597 [0.52, 0.67] * (+0.113) | 0.621 [0.55, 0.69] * (+0.126) |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.004 [0.00, 0.01] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.105 [0.07, 0.14] * (+0.098) |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] | 0.000 [0.00, 0.00] * (-0.051) | 0.027 [0.01, 0.05] | 0.016 [0.00, 0.03] * (-0.035) | 0.047 [0.02, 0.07] |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 | 0.438 | 0.664 | 0.512 | 0.746 |
| P3' contact episodes | 256 | 250 | 106 | 107 | 159 | 182 |

- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)
- videos (MP4, GIF preview next to each): [compare_l2_attempt3_it300_vs_l2_pilotO_NH_it0200_start0_best_most_phases_ep008.mp4](compare_l2_attempt3_it300_vs_l2_pilotO_NH_it0200_start0_best_most_phases_ep008.mp4), [compare_l2_attempt3_it300_vs_l2_pilotO_NH_it0200_start0_median_cost_ep001.mp4](compare_l2_attempt3_it300_vs_l2_pilotO_NH_it0200_start0_median_cost_ep001.mp4), [grid_l2_pilotO_NH_it0200_start0.mp4](grid_l2_pilotO_NH_it0200_start0.mp4), [l2_pilotO_NH_it0200_start0_best_most_phases_ep008.mp4](l2_pilotO_NH_it0200_start0_best_most_phases_ep008.mp4), [l2_pilotO_NH_it0200_start0_median_cost_ep001.mp4](l2_pilotO_NH_it0200_start0_median_cost_ep001.mp4), [l2_pilotO_NH_it0200_start0_never_within_2cm_ep005.mp4](l2_pilotO_NH_it0200_start0_never_within_2cm_ep005.mp4), [l2_pilotO_NH_it0200_start0_within_2cm_too_fast_ep000.mp4](l2_pilotO_NH_it0200_start0_within_2cm_too_fast_ep000.mp4), [l2_pilotO_NH_it0200_start0_wrong_class_ep006.mp4](l2_pilotO_NH_it0200_start0_wrong_class_ep006.mp4), [l2_pilotO_NH_it0200_start2_best_most_phases_ep014.mp4](l2_pilotO_NH_it0200_start2_best_most_phases_ep014.mp4), [l2_pilotO_NH_it0200_start2_did_not_release_ep000.mp4](l2_pilotO_NH_it0200_start2_did_not_release_ep000.mp4), [l2_pilotO_NH_it0200_start2_median_cost_ep163.mp4](l2_pilotO_NH_it0200_start2_median_cost_ep163.mp4), [l2_pilotO_NH_it0200_start2_wrong_class_ep005.mp4](l2_pilotO_NH_it0200_start2_wrong_class_ep005.mp4), [l2_pilotO_NH_it0200_start3_best_most_phases_ep002.mp4](l2_pilotO_NH_it0200_start3_best_most_phases_ep002.mp4), [l2_pilotO_NH_it0200_start3_did_not_release_ep000.mp4](l2_pilotO_NH_it0200_start3_did_not_release_ep000.mp4), [l2_pilotO_NH_it0200_start3_median_cost_ep026.mp4](l2_pilotO_NH_it0200_start3_median_cost_ep026.mp4), [l2_pilotO_NH_it0200_start3_missed_p_star_ep001.mp4](l2_pilotO_NH_it0200_start3_missed_p_star_ep001.mp4), [l2_pilotO_NH_it0200_start3_wrong_class_ep005.mp4](l2_pilotO_NH_it0200_start3_wrong_class_ep005.mp4), [l2_pilotO_NH_it0200_start4_did_not_release_ep000.mp4](l2_pilotO_NH_it0200_start4_did_not_release_ep000.mp4), [l2_pilotO_NH_it0200_start4_median_cost_ep130.mp4](l2_pilotO_NH_it0200_start4_median_cost_ep130.mp4), [l2_pilotO_NH_it0200_start4_missed_p_star_ep002.mp4](l2_pilotO_NH_it0200_start4_missed_p_star_ep002.mp4), [l2_pilotO_NH_it0200_start4_success_ep201.mp4](l2_pilotO_NH_it0200_start4_success_ep201.mp4)
<!-- /success -->

# 20261116012508: pilot O: optimization of the coupled learning

- Plan: `records/l2_pilotO_plan.md`, sha256 prefix 316d6d87
- Status: complete; start 2026-11-16T01:25:08, end 2026-11-16T12:59:00
- Platform: GPU, Intel(R) Core(TM) Ultra 9 285K
- Code: commit 81f2173d028a8e3b4028849b07c8b145954e4f21, dirty, diff 018ab29bbf20
- Key result: adopted N and H; amendment 1: NH adopted (NH minus N -1.735 [-2.028, -1.474]); K, D not adopted

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotO_C | 141-200 | 27.21 | 26.10 | 2.030 | 0 / 0 / 5.36e-04 |
| l2_pilotO_N | 141-200 | 20.40 | 20.27 | 1.909 | 0 / 0 / 8.79e-04 |
| l2_pilotO_K | 141-200 | 26.82 | 25.97 | 2.870 | 0 / 0 / 8.74e-04 |
| l2_pilotO_H | 141-200 | 26.34 | 24.75 | 0.801 | 0 / 0 / 3.56e-04 |
| l2_pilotO_D | 141-200 | 27.66 | 26.61 | 2.194 | 0 / 0 / 5.41e-04 |
| l2_pilotO_NH | 141-200 | 17.43 | 15.21 | 2.908 | 0 / 0 / 7.74e-04 |

## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)

| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |
|---|---|---|---|---|---|---|---|
| start checkpoint 140 | 27.599 / 27.360 | 8.600 / 8.704 | 6.506 / 6.646 | 0.035 | 0.117 | 0.0523 | 1.005 |
| l2_pilotO_C:200 | 25.422 / 24.790 | 7.494 / 7.632 | 5.722 / 5.838 | 0.023 | 0.207 | 0.0366 | 1.794 |
| l2_pilotO_N:200 | 19.509 / 19.482 | 7.354 / 7.506 | 5.677 / 5.773 | 0.070 | 0.266 | 0.0320 | 1.645 |
| l2_pilotO_K:200 | 25.003 / 24.433 | 8.223 / 8.298 | 6.311 / 6.387 | 0.039 | 0.172 | 0.0415 | 2.637 |
| l2_pilotO_H:200 | 23.742 / 23.769 | 7.199 / 7.708 | 5.415 / 5.808 | 0.059 | 0.391 | 0.0314 | 0.722 |
| l2_pilotO_D:200 | 26.204 / 25.224 | 7.306 / 7.475 | 5.618 / 5.729 | 0.039 | 0.387 | 0.0332 | 1.961 |
| l2_pilotO_NH:200 | 16.335 / 15.273 | 6.279 / 5.724 | 4.719 / 4.128 | 0.168 | 0.469 | 0.0212 | 2.444 |

## Decision (records/l2_pilotO_plan.md)

| arm | mean diff vs reference | 95% CI | accurate-press diff | KL median | residual arm / C | role loss arm / C | adopted |
|---|---|---|---|---|---|---|---|
| N | -2.033 | [-2.452, -1.617] | +0.059 | 0.0361 | 1.645 / 1.794 | 4.605 / 5.694 | yes |
| K | +0.300 | [+0.059, +0.536] | -0.035 | 0.0174 | 2.637 / 1.794 | 7.987 / 5.694 | no |
| H | -0.761 | [-1.022, -0.511] | +0.184 | 0.0391 | 0.722 / 1.794 | 9.020 / 5.694 | yes |
| D | +0.163 | [-0.066, +0.405] | +0.180 | 0.0575 | 1.961 / 1.794 | 5.214 / 5.694 | no |
| NH vs N | -1.735 | [-2.028, -1.474] | +0.203 | 0.0267 | 2.444 / 1.645 | 8.882 / 4.605 | yes |

## Images

- [summary.png](summary.png)
- [eval_l2_pilotO_C_it0200.png](eval_l2_pilotO_C_it0200.png)
- [eval_l2_pilotO_N_it0200.png](eval_l2_pilotO_N_it0200.png)
- [eval_l2_pilotO_K_it0200.png](eval_l2_pilotO_K_it0200.png)
- [eval_l2_pilotO_H_it0200.png](eval_l2_pilotO_H_it0200.png)
- [eval_l2_pilotO_D_it0200.png](eval_l2_pilotO_D_it0200.png)
- [eval_l2_pilotO_NH_it0200.png](eval_l2_pilotO_NH_it0200.png)
- [decision.png](decision.png)
- [l2_pilotO_C_start0_it0200.gif](l2_pilotO_C_start0_it0200.gif)
- [l2_pilotO_C_start2_it0200.gif](l2_pilotO_C_start2_it0200.gif)
- [l2_pilotO_C_start3_it0200.gif](l2_pilotO_C_start3_it0200.gif)
- [l2_pilotO_D_start0_it0200.gif](l2_pilotO_D_start0_it0200.gif)
- [l2_pilotO_D_start2_it0200.gif](l2_pilotO_D_start2_it0200.gif)
- [l2_pilotO_D_start3_it0200.gif](l2_pilotO_D_start3_it0200.gif)
- [l2_pilotO_H_start0_it0200.gif](l2_pilotO_H_start0_it0200.gif)
- [l2_pilotO_H_start2_it0200.gif](l2_pilotO_H_start2_it0200.gif)
- [l2_pilotO_H_start3_it0200.gif](l2_pilotO_H_start3_it0200.gif)
- [l2_pilotO_K_start0_it0200.gif](l2_pilotO_K_start0_it0200.gif)
- [l2_pilotO_K_start2_it0200.gif](l2_pilotO_K_start2_it0200.gif)
- [l2_pilotO_K_start3_it0200.gif](l2_pilotO_K_start3_it0200.gif)
- [l2_pilotO_NH_start0_it0200.gif](l2_pilotO_NH_start0_it0200.gif)
- [l2_pilotO_NH_start2_it0200.gif](l2_pilotO_NH_start2_it0200.gif)
- [l2_pilotO_NH_start3_it0200.gif](l2_pilotO_NH_start3_it0200.gif)
- [l2_pilotO_N_start0_it0200.gif](l2_pilotO_N_start0_it0200.gif)
- [l2_pilotO_N_start2_it0200.gif](l2_pilotO_N_start2_it0200.gif)
- [l2_pilotO_N_start3_it0200.gif](l2_pilotO_N_start3_it0200.gif)

- `l2_pilotO_C_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2842 kB
- `l2_pilotO_C_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1705 kB
- `l2_pilotO_C_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1277 kB
- `l2_pilotO_N_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2468 kB
- `l2_pilotO_N_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1869 kB
- `l2_pilotO_N_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1340 kB
- `l2_pilotO_K_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2351 kB
- `l2_pilotO_K_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2095 kB
- `l2_pilotO_K_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1084 kB
- `l2_pilotO_H_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2523 kB
- `l2_pilotO_H_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1736 kB
- `l2_pilotO_H_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1258 kB
- `l2_pilotO_D_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2646 kB
- `l2_pilotO_D_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1799 kB
- `l2_pilotO_D_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1278 kB
- `l2_pilotO_NH_start0_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2796 kB
- `l2_pilotO_NH_start2_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1960 kB
- `l2_pilotO_NH_start3_it0200.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1082 kB
