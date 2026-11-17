<!-- success -->
# Success table: 20261116145109

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (l2_attempt3:300) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 | l2_pilotLd_C:260 | l2_pilotLd_L1:260 | l2_pilotLd_L05:260 |
|---|---|---|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] | 0.309 [0.25, 0.37] * (+0.172) | 0.359 [0.30, 0.42] * (+0.223) | 0.348 [0.29, 0.41] * (+0.211) |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] | 0.719 [0.66, 0.77] * (+0.383) | 0.621 [0.56, 0.68] * (+0.285) | 0.688 [0.63, 0.74] * (+0.352) |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] | 0.207 [0.16, 0.26] * (+0.070) | 0.184 [0.14, 0.23] | 0.176 [0.13, 0.22] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] | 0.657 [0.59, 0.72] * (+0.148) | 0.635 [0.57, 0.69] * (+0.142) | 0.654 [0.60, 0.71] * (+0.152) |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] | 0.266 [0.21, 0.32] * (+0.262) | 0.176 [0.13, 0.23] * (+0.172) | 0.363 [0.30, 0.43] * (+0.359) |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] | 0.176 [0.13, 0.22] * (+0.168) | 0.121 [0.08, 0.16] * (+0.113) | 0.297 [0.24, 0.35] * (+0.289) |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] | 0.242 [0.19, 0.30] * (+0.191) | 0.266 [0.21, 0.32] * (+0.215) | 0.270 [0.21, 0.32] * (+0.219) |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] | 0.035 [0.02, 0.06] * (+0.031) | 0.047 [0.02, 0.07] * (+0.043) | 0.027 [0.01, 0.05] * (+0.023) |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 | 0.820 | 0.824 | 0.828 |
| P3' contact episodes | 256 | 250 | 236 | 249 | 243 |

- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)
- videos (MP4, GIF preview next to each): [compare_l2_attempt3_it300_vs_l2_pilotLd_L05_it0260_start0_best_most_phases_ep039.mp4](compare_l2_attempt3_it300_vs_l2_pilotLd_L05_it0260_start0_best_most_phases_ep039.mp4), [compare_l2_attempt3_it300_vs_l2_pilotLd_L05_it0260_start0_median_cost_ep116.mp4](compare_l2_attempt3_it300_vs_l2_pilotLd_L05_it0260_start0_median_cost_ep116.mp4), [grid_l2_pilotLd_L05_it0260_start0.mp4](grid_l2_pilotLd_L05_it0260_start0.mp4), [l2_pilotLd_L05_it0260_start0_best_most_phases_ep039.mp4](l2_pilotLd_L05_it0260_start0_best_most_phases_ep039.mp4), [l2_pilotLd_L05_it0260_start0_did_not_release_ep006.mp4](l2_pilotLd_L05_it0260_start0_did_not_release_ep006.mp4), [l2_pilotLd_L05_it0260_start0_median_cost_ep116.mp4](l2_pilotLd_L05_it0260_start0_median_cost_ep116.mp4), [l2_pilotLd_L05_it0260_start0_missed_p_star_ep000.mp4](l2_pilotLd_L05_it0260_start0_missed_p_star_ep000.mp4), [l2_pilotLd_L05_it0260_start0_never_within_2cm_ep005.mp4](l2_pilotLd_L05_it0260_start0_never_within_2cm_ep005.mp4), [l2_pilotLd_L05_it0260_start0_within_2cm_too_fast_ep002.mp4](l2_pilotLd_L05_it0260_start0_within_2cm_too_fast_ep002.mp4), [l2_pilotLd_L05_it0260_start0_wrong_class_ep007.mp4](l2_pilotLd_L05_it0260_start0_wrong_class_ep007.mp4), [l2_pilotLd_L05_it0260_start2_best_most_phases_ep013.mp4](l2_pilotLd_L05_it0260_start2_best_most_phases_ep013.mp4), [l2_pilotLd_L05_it0260_start2_did_not_release_ep006.mp4](l2_pilotLd_L05_it0260_start2_did_not_release_ep006.mp4), [l2_pilotLd_L05_it0260_start2_median_cost_ep098.mp4](l2_pilotLd_L05_it0260_start2_median_cost_ep098.mp4), [l2_pilotLd_L05_it0260_start2_missed_p_star_ep000.mp4](l2_pilotLd_L05_it0260_start2_missed_p_star_ep000.mp4), [l2_pilotLd_L05_it0260_start2_wrong_class_ep005.mp4](l2_pilotLd_L05_it0260_start2_wrong_class_ep005.mp4), [l2_pilotLd_L05_it0260_start3_did_not_release_ep027.mp4](l2_pilotLd_L05_it0260_start3_did_not_release_ep027.mp4), [l2_pilotLd_L05_it0260_start3_median_cost_ep028.mp4](l2_pilotLd_L05_it0260_start3_median_cost_ep028.mp4), [l2_pilotLd_L05_it0260_start3_missed_p_star_ep000.mp4](l2_pilotLd_L05_it0260_start3_missed_p_star_ep000.mp4), [l2_pilotLd_L05_it0260_start3_success_ep155.mp4](l2_pilotLd_L05_it0260_start3_success_ep155.mp4), [l2_pilotLd_L05_it0260_start3_wrong_class_ep005.mp4](l2_pilotLd_L05_it0260_start3_wrong_class_ep005.mp4), [l2_pilotLd_L05_it0260_start4_did_not_release_ep006.mp4](l2_pilotLd_L05_it0260_start4_did_not_release_ep006.mp4), [l2_pilotLd_L05_it0260_start4_median_cost_ep018.mp4](l2_pilotLd_L05_it0260_start4_median_cost_ep018.mp4), [l2_pilotLd_L05_it0260_start4_missed_p_star_ep000.mp4](l2_pilotLd_L05_it0260_start4_missed_p_star_ep000.mp4), [l2_pilotLd_L05_it0260_start4_success_ep002.mp4](l2_pilotLd_L05_it0260_start4_success_ep002.mp4)
<!-- /success -->

# 20261116145109: pilot L: transfer term L_tr in training (hand role)

- Plan: `records/l2_pilotL_plan.md`, sha256 prefix f42c54a3
- Status: complete; start 2026-11-16T14:51:09, end 2026-11-16T16:49:33
- Platform: GPU, Intel(R) Core(TM) Ultra 9 285K
- Code: commit 81f2173d028a8e3b4028849b07c8b145954e4f21, dirty, diff 5beafc9c067e

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotLd_C | 201-260 | 13.58 | 11.48 | 6.077 | 0 / 0 / 1.06e-02 |
| l2_pilotLd_L1 | 201-260 | 13.09 | 11.33 | 0.576 | 0 / 0 / 1.68e-02 |
| l2_pilotLd_L05 | 201-260 | 12.98 | 11.34 | 0.704 | 0 / 0 / 1.40e-02 |

## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)

| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |
|---|---|---|---|---|---|---|---|
| l2_pilotLd_C:260 | 11.983 / 12.544 | 4.205 / 4.532 | 2.953 / 3.312 | 0.324 | 0.695 | 0.0129 | 6.081 |
| l2_pilotLd_L1:260 | 11.710 / 11.626 | 4.203 / 4.449 | 2.999 / 3.254 | 0.379 | 0.730 | 0.0150 | 0.579 |
| l2_pilotLd_L05:260 | 11.626 / 11.683 | 4.190 / 4.442 | 2.960 / 3.266 | 0.363 | 0.719 | 0.0128 | 0.686 |

## Decision (records/l2_pilotO_plan.md)

| arm | mean diff vs reference | 95% CI | accurate-press diff | KL median | residual arm / C | role loss arm / C | adopted |
|---|---|---|---|---|---|---|---|
| L1 | -0.076 | [-0.165, -0.004] | +0.035 | 0.0231 | 0.579 / 6.081 | 11.344 / 9.125 | no |
| L05 | -0.121 | [-0.233, -0.035] | +0.023 | 0.0223 | 0.686 / 6.081 | 10.993 / 9.125 | no |

## Images

- [summary.png](summary.png)
- [eval_l2_pilotLd_C_it0260.png](eval_l2_pilotLd_C_it0260.png)
- [eval_l2_pilotLd_L1_it0260.png](eval_l2_pilotLd_L1_it0260.png)
- [eval_l2_pilotLd_L05_it0260.png](eval_l2_pilotLd_L05_it0260.png)
- [decision.png](decision.png)
- [l2_pilotLd_C_start0_it0260.gif](l2_pilotLd_C_start0_it0260.gif)
- [l2_pilotLd_C_start2_it0260.gif](l2_pilotLd_C_start2_it0260.gif)
- [l2_pilotLd_C_start3_it0260.gif](l2_pilotLd_C_start3_it0260.gif)
- [l2_pilotLd_L05_start0_it0260.gif](l2_pilotLd_L05_start0_it0260.gif)
- [l2_pilotLd_L05_start2_it0260.gif](l2_pilotLd_L05_start2_it0260.gif)
- [l2_pilotLd_L05_start3_it0260.gif](l2_pilotLd_L05_start3_it0260.gif)
- [l2_pilotLd_L1_start0_it0260.gif](l2_pilotLd_L1_start0_it0260.gif)
- [l2_pilotLd_L1_start2_it0260.gif](l2_pilotLd_L1_start2_it0260.gif)
- [l2_pilotLd_L1_start3_it0260.gif](l2_pilotLd_L1_start3_it0260.gif)

- `l2_pilotLd_C_start0_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2358 kB
- `l2_pilotLd_C_start2_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1764 kB
- `l2_pilotLd_C_start3_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1376 kB
- `l2_pilotLd_L1_start0_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2315 kB
- `l2_pilotLd_L1_start2_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2020 kB
- `l2_pilotLd_L1_start3_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1351 kB
- `l2_pilotLd_L05_start0_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2126 kB
- `l2_pilotLd_L05_start2_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1989 kB
- `l2_pilotLd_L05_start3_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1131 kB
