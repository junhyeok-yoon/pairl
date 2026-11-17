<!-- success -->
# Success table: 20261116165912

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (l2_pilotLd_L1:260) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 | previous best: l2_pilotLd_L1:260 | l2_pilotG_C:320 | l2_pilotG_S:320 | l2_pilotG_S2:320 | l2_pilotG_B:320 |
|---|---|---|---|---|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] | 0.359 [0.30, 0.42] | 0.504 [0.44, 0.57] * (+0.145) | 0.453 [0.39, 0.52] * (+0.094) | 0.465 [0.40, 0.53] * (+0.105) | 0.461 [0.40, 0.52] * (+0.102) |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] | 0.621 [0.56, 0.68] | 0.594 [0.54, 0.66] | 0.703 [0.64, 0.76] * (+0.082) | 0.582 [0.52, 0.64] | 0.684 [0.62, 0.74] * (+0.062) |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] | 0.184 [0.14, 0.23] | 0.215 [0.17, 0.27] | 0.168 [0.12, 0.21] | 0.176 [0.13, 0.22] | 0.230 [0.18, 0.28] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] | 0.635 [0.57, 0.69] | 0.572 [0.51, 0.63] * (-0.060) | 0.613 [0.55, 0.67] | 0.610 [0.55, 0.67] | 0.606 [0.55, 0.67] |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] | 0.176 [0.13, 0.23] | 0.152 [0.11, 0.20] | 0.125 [0.09, 0.17] | 0.188 [0.14, 0.24] | 0.121 [0.08, 0.16] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] | 0.121 [0.08, 0.16] | 0.160 [0.12, 0.21] | 0.082 [0.05, 0.12] | 0.148 [0.11, 0.20] | 0.082 [0.05, 0.12] |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] | 0.266 [0.21, 0.32] | 0.355 [0.30, 0.41] * (+0.090) | 0.348 [0.29, 0.41] * (+0.082) | 0.332 [0.27, 0.39] * (+0.066) | 0.359 [0.30, 0.42] * (+0.094) |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] | 0.047 [0.02, 0.07] | 0.070 [0.04, 0.10] | 0.039 [0.02, 0.06] | 0.059 [0.03, 0.09] | 0.082 [0.05, 0.12] * (+0.035) |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.004 [0.00, 0.01] | 0.004 [0.00, 0.01] | 0.004 [0.00, 0.01] | 0.008 [0.00, 0.02] |
| late grasp (any time) | 1.000 | 0.680 | 0.824 | 0.883 | 0.863 | 0.863 | 0.859 |
| P3' contact episodes | 256 | 250 | 249 | 250 | 243 | 246 | 249 |

- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)
- videos (MP4, GIF preview next to each): [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_B_it0320_start0_median_cost_ep169.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_B_it0320_start0_median_cost_ep169.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_B_it0320_start0_success_ep026.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_B_it0320_start0_success_ep026.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_median_cost_ep216.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_median_cost_ep216.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_success_ep056.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_success_ep056.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_median_cost_ep242.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_median_cost_ep242.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_success_ep056.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_success_ep056.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_median_cost_ep160.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_median_cost_ep160.mp4), [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_success_ep056.mp4](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_success_ep056.mp4), [grid_l2_pilotG_B_it0320_start0.mp4](grid_l2_pilotG_B_it0320_start0.mp4), [grid_l2_pilotG_C_it0320_start0.mp4](grid_l2_pilotG_C_it0320_start0.mp4), [grid_l2_pilotG_S2_it0320_start0.mp4](grid_l2_pilotG_S2_it0320_start0.mp4), [grid_l2_pilotG_S_it0320_start0.mp4](grid_l2_pilotG_S_it0320_start0.mp4), [l2_pilotG_B_it0320_start0_did_not_release_ep027.mp4](l2_pilotG_B_it0320_start0_did_not_release_ep027.mp4), [l2_pilotG_B_it0320_start0_median_cost_ep169.mp4](l2_pilotG_B_it0320_start0_median_cost_ep169.mp4), [l2_pilotG_B_it0320_start0_missed_p_star_ep000.mp4](l2_pilotG_B_it0320_start0_missed_p_star_ep000.mp4), [l2_pilotG_B_it0320_start0_never_within_2cm_ep005.mp4](l2_pilotG_B_it0320_start0_never_within_2cm_ep005.mp4), [l2_pilotG_B_it0320_start0_success_ep026.mp4](l2_pilotG_B_it0320_start0_success_ep026.mp4), [l2_pilotG_B_it0320_start0_within_2cm_too_fast_ep002.mp4](l2_pilotG_B_it0320_start0_within_2cm_too_fast_ep002.mp4), [l2_pilotG_B_it0320_start2_did_not_release_ep027.mp4](l2_pilotG_B_it0320_start2_did_not_release_ep027.mp4), [l2_pilotG_B_it0320_start2_median_cost_ep240.mp4](l2_pilotG_B_it0320_start2_median_cost_ep240.mp4), [l2_pilotG_B_it0320_start2_missed_p_star_ep000.mp4](l2_pilotG_B_it0320_start2_missed_p_star_ep000.mp4), [l2_pilotG_B_it0320_start2_success_ep055.mp4](l2_pilotG_B_it0320_start2_success_ep055.mp4), [l2_pilotG_B_it0320_start2_wrong_class_ep003.mp4](l2_pilotG_B_it0320_start2_wrong_class_ep003.mp4), [l2_pilotG_B_it0320_start3_did_not_release_ep193.mp4](l2_pilotG_B_it0320_start3_did_not_release_ep193.mp4), [l2_pilotG_B_it0320_start3_median_cost_ep207.mp4](l2_pilotG_B_it0320_start3_median_cost_ep207.mp4), [l2_pilotG_B_it0320_start3_missed_p_star_ep000.mp4](l2_pilotG_B_it0320_start3_missed_p_star_ep000.mp4), [l2_pilotG_B_it0320_start3_success_ep026.mp4](l2_pilotG_B_it0320_start3_success_ep026.mp4), [l2_pilotG_B_it0320_start3_wrong_class_ep002.mp4](l2_pilotG_B_it0320_start3_wrong_class_ep002.mp4), [l2_pilotG_B_it0320_start4_did_not_release_ep007.mp4](l2_pilotG_B_it0320_start4_did_not_release_ep007.mp4), [l2_pilotG_B_it0320_start4_median_cost_ep063.mp4](l2_pilotG_B_it0320_start4_median_cost_ep063.mp4), [l2_pilotG_B_it0320_start4_missed_p_star_ep000.mp4](l2_pilotG_B_it0320_start4_missed_p_star_ep000.mp4), [l2_pilotG_B_it0320_start4_success_ep022.mp4](l2_pilotG_B_it0320_start4_success_ep022.mp4), [l2_pilotG_C_it0320_start0_did_not_release_ep027.mp4](l2_pilotG_C_it0320_start0_did_not_release_ep027.mp4), [l2_pilotG_C_it0320_start0_median_cost_ep216.mp4](l2_pilotG_C_it0320_start0_median_cost_ep216.mp4), [l2_pilotG_C_it0320_start0_missed_p_star_ep000.mp4](l2_pilotG_C_it0320_start0_missed_p_star_ep000.mp4), [l2_pilotG_C_it0320_start0_never_within_2cm_ep005.mp4](l2_pilotG_C_it0320_start0_never_within_2cm_ep005.mp4), [l2_pilotG_C_it0320_start0_success_ep056.mp4](l2_pilotG_C_it0320_start0_success_ep056.mp4), [l2_pilotG_C_it0320_start0_within_2cm_too_fast_ep002.mp4](l2_pilotG_C_it0320_start0_within_2cm_too_fast_ep002.mp4), [l2_pilotG_C_it0320_start2_did_not_release_ep027.mp4](l2_pilotG_C_it0320_start2_did_not_release_ep027.mp4), [l2_pilotG_C_it0320_start2_median_cost_ep235.mp4](l2_pilotG_C_it0320_start2_median_cost_ep235.mp4), [l2_pilotG_C_it0320_start2_missed_p_star_ep000.mp4](l2_pilotG_C_it0320_start2_missed_p_star_ep000.mp4), [l2_pilotG_C_it0320_start2_success_ep056.mp4](l2_pilotG_C_it0320_start2_success_ep056.mp4), [l2_pilotG_C_it0320_start2_wrong_class_ep002.mp4](l2_pilotG_C_it0320_start2_wrong_class_ep002.mp4), [l2_pilotG_C_it0320_start3_did_not_release_ep027.mp4](l2_pilotG_C_it0320_start3_did_not_release_ep027.mp4), [l2_pilotG_C_it0320_start3_median_cost_ep231.mp4](l2_pilotG_C_it0320_start3_median_cost_ep231.mp4), [l2_pilotG_C_it0320_start3_missed_p_star_ep000.mp4](l2_pilotG_C_it0320_start3_missed_p_star_ep000.mp4), [l2_pilotG_C_it0320_start3_success_ep056.mp4](l2_pilotG_C_it0320_start3_success_ep056.mp4), [l2_pilotG_C_it0320_start3_wrong_class_ep002.mp4](l2_pilotG_C_it0320_start3_wrong_class_ep002.mp4), [l2_pilotG_C_it0320_start4_did_not_release_ep007.mp4](l2_pilotG_C_it0320_start4_did_not_release_ep007.mp4), [l2_pilotG_C_it0320_start4_median_cost_ep177.mp4](l2_pilotG_C_it0320_start4_median_cost_ep177.mp4), [l2_pilotG_C_it0320_start4_missed_p_star_ep000.mp4](l2_pilotG_C_it0320_start4_missed_p_star_ep000.mp4), [l2_pilotG_C_it0320_start4_success_ep029.mp4](l2_pilotG_C_it0320_start4_success_ep029.mp4), [l2_pilotG_S2_it0320_start0_did_not_release_ep027.mp4](l2_pilotG_S2_it0320_start0_did_not_release_ep027.mp4), [l2_pilotG_S2_it0320_start0_median_cost_ep242.mp4](l2_pilotG_S2_it0320_start0_median_cost_ep242.mp4), [l2_pilotG_S2_it0320_start0_missed_p_star_ep000.mp4](l2_pilotG_S2_it0320_start0_missed_p_star_ep000.mp4), [l2_pilotG_S2_it0320_start0_never_within_2cm_ep005.mp4](l2_pilotG_S2_it0320_start0_never_within_2cm_ep005.mp4), [l2_pilotG_S2_it0320_start0_success_ep056.mp4](l2_pilotG_S2_it0320_start0_success_ep056.mp4), [l2_pilotG_S2_it0320_start0_within_2cm_too_fast_ep002.mp4](l2_pilotG_S2_it0320_start0_within_2cm_too_fast_ep002.mp4), [l2_pilotG_S2_it0320_start2_did_not_release_ep027.mp4](l2_pilotG_S2_it0320_start2_did_not_release_ep027.mp4), [l2_pilotG_S2_it0320_start2_median_cost_ep099.mp4](l2_pilotG_S2_it0320_start2_median_cost_ep099.mp4), [l2_pilotG_S2_it0320_start2_missed_p_star_ep000.mp4](l2_pilotG_S2_it0320_start2_missed_p_star_ep000.mp4), [l2_pilotG_S2_it0320_start2_success_ep056.mp4](l2_pilotG_S2_it0320_start2_success_ep056.mp4), [l2_pilotG_S2_it0320_start2_wrong_class_ep002.mp4](l2_pilotG_S2_it0320_start2_wrong_class_ep002.mp4), [l2_pilotG_S2_it0320_start3_did_not_release_ep090.mp4](l2_pilotG_S2_it0320_start3_did_not_release_ep090.mp4), [l2_pilotG_S2_it0320_start3_median_cost_ep027.mp4](l2_pilotG_S2_it0320_start3_median_cost_ep027.mp4), [l2_pilotG_S2_it0320_start3_missed_p_star_ep000.mp4](l2_pilotG_S2_it0320_start3_missed_p_star_ep000.mp4), [l2_pilotG_S2_it0320_start3_success_ep056.mp4](l2_pilotG_S2_it0320_start3_success_ep056.mp4), [l2_pilotG_S2_it0320_start3_wrong_class_ep005.mp4](l2_pilotG_S2_it0320_start3_wrong_class_ep005.mp4), [l2_pilotG_S2_it0320_start4_did_not_release_ep007.mp4](l2_pilotG_S2_it0320_start4_did_not_release_ep007.mp4), [l2_pilotG_S2_it0320_start4_median_cost_ep129.mp4](l2_pilotG_S2_it0320_start4_median_cost_ep129.mp4), [l2_pilotG_S2_it0320_start4_missed_p_star_ep000.mp4](l2_pilotG_S2_it0320_start4_missed_p_star_ep000.mp4), [l2_pilotG_S2_it0320_start4_success_ep005.mp4](l2_pilotG_S2_it0320_start4_success_ep005.mp4), [l2_pilotG_S_it0320_start0_did_not_release_ep193.mp4](l2_pilotG_S_it0320_start0_did_not_release_ep193.mp4), [l2_pilotG_S_it0320_start0_median_cost_ep160.mp4](l2_pilotG_S_it0320_start0_median_cost_ep160.mp4), [l2_pilotG_S_it0320_start0_missed_p_star_ep000.mp4](l2_pilotG_S_it0320_start0_missed_p_star_ep000.mp4), [l2_pilotG_S_it0320_start0_never_within_2cm_ep005.mp4](l2_pilotG_S_it0320_start0_never_within_2cm_ep005.mp4), [l2_pilotG_S_it0320_start0_success_ep056.mp4](l2_pilotG_S_it0320_start0_success_ep056.mp4), [l2_pilotG_S_it0320_start0_within_2cm_too_fast_ep002.mp4](l2_pilotG_S_it0320_start0_within_2cm_too_fast_ep002.mp4), [l2_pilotG_S_it0320_start2_did_not_release_ep193.mp4](l2_pilotG_S_it0320_start2_did_not_release_ep193.mp4), [l2_pilotG_S_it0320_start2_median_cost_ep029.mp4](l2_pilotG_S_it0320_start2_median_cost_ep029.mp4), [l2_pilotG_S_it0320_start2_missed_p_star_ep000.mp4](l2_pilotG_S_it0320_start2_missed_p_star_ep000.mp4), [l2_pilotG_S_it0320_start2_success_ep052.mp4](l2_pilotG_S_it0320_start2_success_ep052.mp4), [l2_pilotG_S_it0320_start2_wrong_class_ep003.mp4](l2_pilotG_S_it0320_start2_wrong_class_ep003.mp4), [l2_pilotG_S_it0320_start3_median_cost_ep107.mp4](l2_pilotG_S_it0320_start3_median_cost_ep107.mp4), [l2_pilotG_S_it0320_start3_missed_p_star_ep000.mp4](l2_pilotG_S_it0320_start3_missed_p_star_ep000.mp4), [l2_pilotG_S_it0320_start3_success_ep056.mp4](l2_pilotG_S_it0320_start3_success_ep056.mp4), [l2_pilotG_S_it0320_start3_wrong_class_ep002.mp4](l2_pilotG_S_it0320_start3_wrong_class_ep002.mp4), [l2_pilotG_S_it0320_start4_did_not_release_ep009.mp4](l2_pilotG_S_it0320_start4_did_not_release_ep009.mp4), [l2_pilotG_S_it0320_start4_median_cost_ep229.mp4](l2_pilotG_S_it0320_start4_median_cost_ep229.mp4), [l2_pilotG_S_it0320_start4_missed_p_star_ep000.mp4](l2_pilotG_S_it0320_start4_missed_p_star_ep000.mp4), [l2_pilotG_S_it0320_start4_success_ep010.mp4](l2_pilotG_S_it0320_start4_success_ep010.mp4)
<!-- /success -->

# 20261116165912: pilot G: grasp gate (brain, phase 1) and ledger channel (hand to brain)

- Plan: `records/l2_pilotG_plan.md`, sha256 prefix a23c7d49
- Status: complete; start 2026-11-16T16:59:12, end 2026-11-16T19:59:37
- Platform: GPU, Intel(R) Core(TM) Ultra 9 285K
- Code: commit 81f2173d028a8e3b4028849b07c8b145954e4f21, dirty, diff 9e93ab0996ec

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotG_C | 261-320 | 10.25 | 8.90 | 5.756 | 0 / 0 / 1.00e-02 |
| l2_pilotG_S | 261-320 | 11.32 | 10.26 | 3.021 | 0 / 0 / 6.30e-03 |
| l2_pilotG_S2 | 261-320 | 11.63 | 10.30 | 2.874 | 0 / 0 / 1.84e-02 |
| l2_pilotG_B | 261-320 | 11.63 | 10.07 | 2.853 | 0 / 0 / 1.21e-02 |

## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)

| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |
|---|---|---|---|---|---|---|---|
| l2_pilotG_C:320 | 9.051 / 9.047 | 3.683 / 3.959 | 2.600 / 2.859 | 0.547 | 0.754 | 0.0157 | 4.610 |
| l2_pilotG_S:320 | 9.899 / 10.063 | 3.651 / 3.996 | 2.558 / 2.888 | 0.445 | 0.707 | 0.0126 | 2.692 |
| l2_pilotG_S2:320 | 9.895 / 10.018 | 3.782 / 4.063 | 2.661 / 2.930 | 0.480 | 0.742 | 0.0164 | 2.594 |
| l2_pilotG_B:320 | 9.953 / 10.144 | 3.505 / 3.813 | 2.486 / 2.795 | 0.480 | 0.742 | 0.0138 | 2.466 |

## Decision (records/l2_pilotO_plan.md)

| arm | mean diff vs reference | 95% CI | accurate-press diff | KL median | residual arm / C | role loss arm / C | adopted |
|---|---|---|---|---|---|---|---|
| S | +0.258 | [+0.131, +0.409] | -0.047 | 0.0198 | 2.692 / 4.610 | 9.370 / 8.576 | no |
| S2 | +0.335 | [+0.211, +0.480] | -0.012 | 0.0202 | 2.594 / 4.610 | 9.288 / 8.576 | no |
| B | +0.203 | [+0.052, +0.380] | -0.012 | 0.0192 | 2.466 / 4.610 | 8.759 / 8.576 | no |

## Images

- [summary.png](summary.png)
- [eval_l2_pilotG_C_it0320.png](eval_l2_pilotG_C_it0320.png)
- [eval_l2_pilotG_S_it0320.png](eval_l2_pilotG_S_it0320.png)
- [eval_l2_pilotG_S2_it0320.png](eval_l2_pilotG_S2_it0320.png)
- [eval_l2_pilotG_B_it0320.png](eval_l2_pilotG_B_it0320.png)
- [decision.png](decision.png)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_median_cost_ep216.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_median_cost_ep216.gif)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_success_ep056.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_C_it0320_start0_success_ep056.gif)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_median_cost_ep242.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_median_cost_ep242.gif)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_success_ep056.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S2_it0320_start0_success_ep056.gif)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_median_cost_ep160.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_median_cost_ep160.gif)
- [compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_success_ep056.gif](compare_l2_pilotLd_L1_it260_vs_l2_pilotG_S_it0320_start0_success_ep056.gif)
- [grid_l2_pilotG_C_it0320_start0.gif](grid_l2_pilotG_C_it0320_start0.gif)
- [grid_l2_pilotG_S2_it0320_start0.gif](grid_l2_pilotG_S2_it0320_start0.gif)
- [grid_l2_pilotG_S_it0320_start0.gif](grid_l2_pilotG_S_it0320_start0.gif)
- [l2_pilotG_B_start0_it0320.gif](l2_pilotG_B_start0_it0320.gif)
- [l2_pilotG_B_start2_it0320.gif](l2_pilotG_B_start2_it0320.gif)
- [l2_pilotG_B_start3_it0320.gif](l2_pilotG_B_start3_it0320.gif)
- [l2_pilotG_C_it0320_start0_did_not_release_ep027.gif](l2_pilotG_C_it0320_start0_did_not_release_ep027.gif)
- [l2_pilotG_C_it0320_start0_median_cost_ep216.gif](l2_pilotG_C_it0320_start0_median_cost_ep216.gif)
- [l2_pilotG_C_it0320_start0_missed_p_star_ep000.gif](l2_pilotG_C_it0320_start0_missed_p_star_ep000.gif)
- [l2_pilotG_C_it0320_start0_never_within_2cm_ep005.gif](l2_pilotG_C_it0320_start0_never_within_2cm_ep005.gif)
- [l2_pilotG_C_it0320_start0_success_ep056.gif](l2_pilotG_C_it0320_start0_success_ep056.gif)
- [l2_pilotG_C_it0320_start0_within_2cm_too_fast_ep002.gif](l2_pilotG_C_it0320_start0_within_2cm_too_fast_ep002.gif)
- [l2_pilotG_C_it0320_start2_did_not_release_ep027.gif](l2_pilotG_C_it0320_start2_did_not_release_ep027.gif)
- [l2_pilotG_C_it0320_start2_median_cost_ep235.gif](l2_pilotG_C_it0320_start2_median_cost_ep235.gif)
- [l2_pilotG_C_it0320_start2_missed_p_star_ep000.gif](l2_pilotG_C_it0320_start2_missed_p_star_ep000.gif)
- [l2_pilotG_C_it0320_start2_success_ep056.gif](l2_pilotG_C_it0320_start2_success_ep056.gif)
- [l2_pilotG_C_it0320_start2_wrong_class_ep002.gif](l2_pilotG_C_it0320_start2_wrong_class_ep002.gif)
- [l2_pilotG_C_it0320_start3_did_not_release_ep027.gif](l2_pilotG_C_it0320_start3_did_not_release_ep027.gif)
- [l2_pilotG_C_it0320_start3_median_cost_ep231.gif](l2_pilotG_C_it0320_start3_median_cost_ep231.gif)
- [l2_pilotG_C_it0320_start3_missed_p_star_ep000.gif](l2_pilotG_C_it0320_start3_missed_p_star_ep000.gif)
- [l2_pilotG_C_it0320_start3_success_ep056.gif](l2_pilotG_C_it0320_start3_success_ep056.gif)
- [l2_pilotG_C_it0320_start3_wrong_class_ep002.gif](l2_pilotG_C_it0320_start3_wrong_class_ep002.gif)
- [l2_pilotG_C_it0320_start4_did_not_release_ep007.gif](l2_pilotG_C_it0320_start4_did_not_release_ep007.gif)
- [l2_pilotG_C_it0320_start4_median_cost_ep177.gif](l2_pilotG_C_it0320_start4_median_cost_ep177.gif)
- [l2_pilotG_C_it0320_start4_missed_p_star_ep000.gif](l2_pilotG_C_it0320_start4_missed_p_star_ep000.gif)
- [l2_pilotG_C_it0320_start4_success_ep029.gif](l2_pilotG_C_it0320_start4_success_ep029.gif)
- [l2_pilotG_C_start0_it0320.gif](l2_pilotG_C_start0_it0320.gif)
- [l2_pilotG_C_start2_it0320.gif](l2_pilotG_C_start2_it0320.gif)
- [l2_pilotG_C_start3_it0320.gif](l2_pilotG_C_start3_it0320.gif)
- [l2_pilotG_S2_it0320_start0_did_not_release_ep027.gif](l2_pilotG_S2_it0320_start0_did_not_release_ep027.gif)
- [l2_pilotG_S2_it0320_start0_median_cost_ep242.gif](l2_pilotG_S2_it0320_start0_median_cost_ep242.gif)
- [l2_pilotG_S2_it0320_start0_missed_p_star_ep000.gif](l2_pilotG_S2_it0320_start0_missed_p_star_ep000.gif)
- [l2_pilotG_S2_it0320_start0_never_within_2cm_ep005.gif](l2_pilotG_S2_it0320_start0_never_within_2cm_ep005.gif)
- [l2_pilotG_S2_it0320_start0_success_ep056.gif](l2_pilotG_S2_it0320_start0_success_ep056.gif)
- [l2_pilotG_S2_it0320_start0_within_2cm_too_fast_ep002.gif](l2_pilotG_S2_it0320_start0_within_2cm_too_fast_ep002.gif)
- [l2_pilotG_S2_it0320_start2_did_not_release_ep027.gif](l2_pilotG_S2_it0320_start2_did_not_release_ep027.gif)
- [l2_pilotG_S2_it0320_start2_median_cost_ep099.gif](l2_pilotG_S2_it0320_start2_median_cost_ep099.gif)
- [l2_pilotG_S2_it0320_start2_missed_p_star_ep000.gif](l2_pilotG_S2_it0320_start2_missed_p_star_ep000.gif)
- [l2_pilotG_S2_it0320_start2_success_ep056.gif](l2_pilotG_S2_it0320_start2_success_ep056.gif)
- [l2_pilotG_S2_it0320_start2_wrong_class_ep002.gif](l2_pilotG_S2_it0320_start2_wrong_class_ep002.gif)
- [l2_pilotG_S2_it0320_start3_did_not_release_ep090.gif](l2_pilotG_S2_it0320_start3_did_not_release_ep090.gif)
- [l2_pilotG_S2_it0320_start3_median_cost_ep027.gif](l2_pilotG_S2_it0320_start3_median_cost_ep027.gif)
- [l2_pilotG_S2_it0320_start3_missed_p_star_ep000.gif](l2_pilotG_S2_it0320_start3_missed_p_star_ep000.gif)
- [l2_pilotG_S2_it0320_start3_success_ep056.gif](l2_pilotG_S2_it0320_start3_success_ep056.gif)
- [l2_pilotG_S2_it0320_start3_wrong_class_ep005.gif](l2_pilotG_S2_it0320_start3_wrong_class_ep005.gif)
- [l2_pilotG_S2_it0320_start4_did_not_release_ep007.gif](l2_pilotG_S2_it0320_start4_did_not_release_ep007.gif)
- [l2_pilotG_S2_it0320_start4_median_cost_ep129.gif](l2_pilotG_S2_it0320_start4_median_cost_ep129.gif)
- [l2_pilotG_S2_it0320_start4_missed_p_star_ep000.gif](l2_pilotG_S2_it0320_start4_missed_p_star_ep000.gif)
- [l2_pilotG_S2_it0320_start4_success_ep005.gif](l2_pilotG_S2_it0320_start4_success_ep005.gif)
- [l2_pilotG_S2_start0_it0320.gif](l2_pilotG_S2_start0_it0320.gif)
- [l2_pilotG_S2_start2_it0320.gif](l2_pilotG_S2_start2_it0320.gif)
- [l2_pilotG_S2_start3_it0320.gif](l2_pilotG_S2_start3_it0320.gif)
- [l2_pilotG_S_it0320_start0_did_not_release_ep193.gif](l2_pilotG_S_it0320_start0_did_not_release_ep193.gif)
- [l2_pilotG_S_it0320_start0_median_cost_ep160.gif](l2_pilotG_S_it0320_start0_median_cost_ep160.gif)
- [l2_pilotG_S_it0320_start0_missed_p_star_ep000.gif](l2_pilotG_S_it0320_start0_missed_p_star_ep000.gif)
- [l2_pilotG_S_it0320_start0_never_within_2cm_ep005.gif](l2_pilotG_S_it0320_start0_never_within_2cm_ep005.gif)
- [l2_pilotG_S_it0320_start0_success_ep056.gif](l2_pilotG_S_it0320_start0_success_ep056.gif)
- [l2_pilotG_S_it0320_start0_within_2cm_too_fast_ep002.gif](l2_pilotG_S_it0320_start0_within_2cm_too_fast_ep002.gif)
- [l2_pilotG_S_it0320_start2_did_not_release_ep193.gif](l2_pilotG_S_it0320_start2_did_not_release_ep193.gif)
- [l2_pilotG_S_it0320_start2_median_cost_ep029.gif](l2_pilotG_S_it0320_start2_median_cost_ep029.gif)
- [l2_pilotG_S_it0320_start2_missed_p_star_ep000.gif](l2_pilotG_S_it0320_start2_missed_p_star_ep000.gif)
- [l2_pilotG_S_it0320_start2_success_ep052.gif](l2_pilotG_S_it0320_start2_success_ep052.gif)
- [l2_pilotG_S_it0320_start2_wrong_class_ep003.gif](l2_pilotG_S_it0320_start2_wrong_class_ep003.gif)
- [l2_pilotG_S_it0320_start3_median_cost_ep107.gif](l2_pilotG_S_it0320_start3_median_cost_ep107.gif)
- [l2_pilotG_S_it0320_start3_missed_p_star_ep000.gif](l2_pilotG_S_it0320_start3_missed_p_star_ep000.gif)
- [l2_pilotG_S_it0320_start3_success_ep056.gif](l2_pilotG_S_it0320_start3_success_ep056.gif)
- [l2_pilotG_S_it0320_start3_wrong_class_ep002.gif](l2_pilotG_S_it0320_start3_wrong_class_ep002.gif)
- [l2_pilotG_S_it0320_start4_did_not_release_ep009.gif](l2_pilotG_S_it0320_start4_did_not_release_ep009.gif)
- [l2_pilotG_S_it0320_start4_median_cost_ep229.gif](l2_pilotG_S_it0320_start4_median_cost_ep229.gif)
- [l2_pilotG_S_it0320_start4_missed_p_star_ep000.gif](l2_pilotG_S_it0320_start4_missed_p_star_ep000.gif)
- [l2_pilotG_S_it0320_start4_success_ep010.gif](l2_pilotG_S_it0320_start4_success_ep010.gif)
- [l2_pilotG_S_start0_it0320.gif](l2_pilotG_S_start0_it0320.gif)
- [l2_pilotG_S_start2_it0320.gif](l2_pilotG_S_start2_it0320.gif)
- [l2_pilotG_S_start3_it0320.gif](l2_pilotG_S_start3_it0320.gif)

- `l2_pilotG_C_start0_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2665 kB
- `l2_pilotG_C_start2_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1896 kB
- `l2_pilotG_C_start3_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1150 kB
- `l2_pilotG_S_start0_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2400 kB
- `l2_pilotG_S_start2_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1882 kB
- `l2_pilotG_S_start3_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1344 kB
- `l2_pilotG_S2_start0_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2743 kB
- `l2_pilotG_S2_start2_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1963 kB
- `l2_pilotG_S2_start3_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1239 kB
- `l2_pilotG_B_start0_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2935 kB
- `l2_pilotG_B_start2_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2074 kB
- `l2_pilotG_B_start3_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1108 kB
- `l2_pilotG_B_start0_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2935 kB
- `l2_pilotG_B_start2_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2074 kB
- `l2_pilotG_B_start3_it0320.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1108 kB
