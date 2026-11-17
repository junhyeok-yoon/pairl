<!-- success -->
# Success table: 20261116200435

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (l2_pilotG_B:320) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 | previous best: l2_pilotG_B:320 | l2_pilotLd_L025:260 |
|---|---|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] | 0.461 [0.40, 0.52] | 0.293 [0.24, 0.35] * (-0.168) |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] | 0.684 [0.62, 0.74] | 0.766 [0.71, 0.82] * (+0.082) |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] | 0.230 [0.18, 0.28] | 0.191 [0.14, 0.24] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] | 0.606 [0.55, 0.67] | 0.642 [0.58, 0.70] |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] | 0.121 [0.08, 0.16] | 0.117 [0.08, 0.16] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] | 0.082 [0.05, 0.12] | 0.062 [0.04, 0.09] |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] | 0.359 [0.30, 0.42] | 0.242 [0.19, 0.30] * (-0.117) |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] | 0.082 [0.05, 0.12] | 0.043 [0.02, 0.07] * (-0.039) |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] | 0.008 [0.00, 0.02] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 | 0.859 | 0.805 |
| P3' contact episodes | 256 | 250 | 249 | 240 |

- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)
- videos (MP4, GIF preview next to each): [compare_l2_pilotG_B_it320_vs_l2_pilotLd_L025_it0260_start0_best_most_phases_ep000.mp4](compare_l2_pilotG_B_it320_vs_l2_pilotLd_L025_it0260_start0_best_most_phases_ep000.mp4), [compare_l2_pilotG_B_it320_vs_l2_pilotLd_L025_it0260_start0_median_cost_ep021.mp4](compare_l2_pilotG_B_it320_vs_l2_pilotLd_L025_it0260_start0_median_cost_ep021.mp4), [grid_l2_pilotLd_L025_it0260_start0.mp4](grid_l2_pilotLd_L025_it0260_start0.mp4), [l2_pilotLd_L025_it0260_start0_best_most_phases_ep000.mp4](l2_pilotLd_L025_it0260_start0_best_most_phases_ep000.mp4), [l2_pilotLd_L025_it0260_start0_did_not_release_ep027.mp4](l2_pilotLd_L025_it0260_start0_did_not_release_ep027.mp4), [l2_pilotLd_L025_it0260_start0_median_cost_ep021.mp4](l2_pilotLd_L025_it0260_start0_median_cost_ep021.mp4), [l2_pilotLd_L025_it0260_start0_never_within_2cm_ep003.mp4](l2_pilotLd_L025_it0260_start0_never_within_2cm_ep003.mp4), [l2_pilotLd_L025_it0260_start0_within_2cm_too_fast_ep001.mp4](l2_pilotLd_L025_it0260_start0_within_2cm_too_fast_ep001.mp4), [l2_pilotLd_L025_it0260_start0_wrong_class_ep002.mp4](l2_pilotLd_L025_it0260_start0_wrong_class_ep002.mp4), [l2_pilotLd_L025_it0260_start2_best_most_phases_ep000.mp4](l2_pilotLd_L025_it0260_start2_best_most_phases_ep000.mp4), [l2_pilotLd_L025_it0260_start2_did_not_release_ep027.mp4](l2_pilotLd_L025_it0260_start2_did_not_release_ep027.mp4), [l2_pilotLd_L025_it0260_start2_median_cost_ep239.mp4](l2_pilotLd_L025_it0260_start2_median_cost_ep239.mp4), [l2_pilotLd_L025_it0260_start2_wrong_class_ep002.mp4](l2_pilotLd_L025_it0260_start2_wrong_class_ep002.mp4), [l2_pilotLd_L025_it0260_start3_best_most_phases_ep000.mp4](l2_pilotLd_L025_it0260_start3_best_most_phases_ep000.mp4), [l2_pilotLd_L025_it0260_start3_did_not_release_ep090.mp4](l2_pilotLd_L025_it0260_start3_did_not_release_ep090.mp4), [l2_pilotLd_L025_it0260_start3_median_cost_ep061.mp4](l2_pilotLd_L025_it0260_start3_median_cost_ep061.mp4), [l2_pilotLd_L025_it0260_start3_wrong_class_ep002.mp4](l2_pilotLd_L025_it0260_start3_wrong_class_ep002.mp4), [l2_pilotLd_L025_it0260_start4_did_not_release_ep006.mp4](l2_pilotLd_L025_it0260_start4_did_not_release_ep006.mp4), [l2_pilotLd_L025_it0260_start4_median_cost_ep070.mp4](l2_pilotLd_L025_it0260_start4_median_cost_ep070.mp4), [l2_pilotLd_L025_it0260_start4_missed_p_star_ep000.mp4](l2_pilotLd_L025_it0260_start4_missed_p_star_ep000.mp4), [l2_pilotLd_L025_it0260_start4_success_ep005.mp4](l2_pilotLd_L025_it0260_start4_success_ep005.mp4)
<!-- /success -->

# 20261116200435: pilot L amendment 2: L_tr weight 0.25 (hand role)

- Plan: `records/l2_pilotL_plan.md`, sha256 prefix 05f942f7
- Status: complete; start 2026-11-16T20:04:35, end 2026-11-16T20:44:01
- Platform: GPU, Intel(R) Core(TM) Ultra 9 285K
- Code: commit 81f2173d028a8e3b4028849b07c8b145954e4f21, dirty, diff 9e93ab0996ec

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotLd_L025 | 201-260 | 13.93 | 11.97 | 0.879 | 0 / 0 / 9.79e-03 |

## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)

| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |
|---|---|---|---|---|---|---|---|
| l2_pilotLd_L025:260 | 12.533 / 12.596 | 4.236 / 4.478 | 2.951 / 3.248 | 0.309 | 0.672 | 0.0112 | 0.863 |

## Images

- [summary.png](summary.png)
- [eval_l2_pilotLd_L025_it0260.png](eval_l2_pilotLd_L025_it0260.png)
- [l2_pilotLd_L025_start0_it0260.gif](l2_pilotLd_L025_start0_it0260.gif)
- [l2_pilotLd_L025_start2_it0260.gif](l2_pilotLd_L025_start2_it0260.gif)
- [l2_pilotLd_L025_start3_it0260.gif](l2_pilotLd_L025_start3_it0260.gif)

- `l2_pilotLd_L025_start0_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2628 kB
- `l2_pilotLd_L025_start2_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1878 kB
- `l2_pilotLd_L025_start3_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1292 kB
