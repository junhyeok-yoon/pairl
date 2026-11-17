<!-- success -->
# Success table: 20261115134229

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (None) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 |
|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 |
| P3' contact episodes | 256 | 250 |

- [success_table.md](success_table.md), [funnel.png](funnel.png), [success_curve.png](success_curve.png)
- videos (MP4, GIF preview next to each): [grid_l2_attempt3_it0300_start0.mp4](grid_l2_attempt3_it0300_start0.mp4), [l2_attempt3_it0300_start0_best_most_phases_ep001.mp4](l2_attempt3_it0300_start0_best_most_phases_ep001.mp4), [l2_attempt3_it0300_start0_median_cost_ep143.mp4](l2_attempt3_it0300_start0_median_cost_ep143.mp4), [l2_attempt3_it0300_start0_never_within_2cm_ep005.mp4](l2_attempt3_it0300_start0_never_within_2cm_ep005.mp4), [l2_attempt3_it0300_start0_within_2cm_too_fast_ep000.mp4](l2_attempt3_it0300_start0_within_2cm_too_fast_ep000.mp4), [l2_attempt3_it0300_start0_wrong_class_ep002.mp4](l2_attempt3_it0300_start0_wrong_class_ep002.mp4), [l2_attempt3_it0300_start2_best_most_phases_ep039.mp4](l2_attempt3_it0300_start2_best_most_phases_ep039.mp4), [l2_attempt3_it0300_start2_did_not_release_ep000.mp4](l2_attempt3_it0300_start2_did_not_release_ep000.mp4), [l2_attempt3_it0300_start2_median_cost_ep221.mp4](l2_attempt3_it0300_start2_median_cost_ep221.mp4), [l2_attempt3_it0300_start2_wrong_class_ep002.mp4](l2_attempt3_it0300_start2_wrong_class_ep002.mp4), [l2_attempt3_it0300_start3_best_most_phases_ep000.mp4](l2_attempt3_it0300_start3_best_most_phases_ep000.mp4), [l2_attempt3_it0300_start3_median_cost_ep206.mp4](l2_attempt3_it0300_start3_median_cost_ep206.mp4), [l2_attempt3_it0300_start3_wrong_class_ep002.mp4](l2_attempt3_it0300_start3_wrong_class_ep002.mp4), [l2_attempt3_it0300_start4_did_not_release_ep000.mp4](l2_attempt3_it0300_start4_did_not_release_ep000.mp4), [l2_attempt3_it0300_start4_median_cost_ep008.mp4](l2_attempt3_it0300_start4_median_cost_ep008.mp4), [l2_attempt3_it0300_start4_missed_p_star_ep001.mp4](l2_attempt3_it0300_start4_missed_p_star_ep001.mp4), [l2_attempt3_it0300_start4_success_ep136.mp4](l2_attempt3_it0300_start4_success_ep136.mp4)
<!-- /success -->

# 20261115134229: L2 attempt 3 (continues l2_pilotT_T from 100)

- Plan: `records/l2_attempt3_plan.md`, sha256 prefix 321654bc
- Status: complete (300); start 2026-11-15T13:42:29, end 2026-11-15T15:38:58
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: (G) met, (H1) and (H2) not met; evaluation cost 17.37 at 300 (cloud)

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_attempt3 | 101-300 | 17.45 | 17.37 | 4.427 | 0 / 0 / 6.79e-04 |

## Images

- [summary.png](summary.png)
- [l2_attempt3_start0_it0150.gif](l2_attempt3_start0_it0150.gif)
- [l2_attempt3_start0_it0200.gif](l2_attempt3_start0_it0200.gif)
- [l2_attempt3_start0_it0250.gif](l2_attempt3_start0_it0250.gif)
- [l2_attempt3_start0_it0300.gif](l2_attempt3_start0_it0300.gif)
- [l2_attempt3_start2_it0150.gif](l2_attempt3_start2_it0150.gif)
- [l2_attempt3_start2_it0200.gif](l2_attempt3_start2_it0200.gif)
- [l2_attempt3_start2_it0250.gif](l2_attempt3_start2_it0250.gif)
- [l2_attempt3_start2_it0300.gif](l2_attempt3_start2_it0300.gif)
- [l2_attempt3_start3_it0150.gif](l2_attempt3_start3_it0150.gif)
- [l2_attempt3_start3_it0200.gif](l2_attempt3_start3_it0200.gif)
- [l2_attempt3_start3_it0250.gif](l2_attempt3_start3_it0250.gif)
- [l2_attempt3_start3_it0300.gif](l2_attempt3_start3_it0300.gif)

## GIFs

- `l2_attempt3_start0_it0150.gif`: copied from docs/figures/rollouts/l2_attempt3_start0_it0150.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 4575 kB
- `l2_attempt3_start0_it0200.gif`: copied from docs/figures/rollouts/l2_attempt3_start0_it0200.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 4499 kB
- `l2_attempt3_start0_it0250.gif`: copied from docs/figures/rollouts/l2_attempt3_start0_it0250.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 4431 kB
- `l2_attempt3_start0_it0300.gif`: copied from docs/figures/rollouts/l2_attempt3_start0_it0300.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 4536 kB
- `l2_attempt3_start2_it0150.gif`: copied from docs/figures/rollouts/l2_attempt3_start2_it0150.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3467 kB
- `l2_attempt3_start2_it0200.gif`: copied from docs/figures/rollouts/l2_attempt3_start2_it0200.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2581 kB
- `l2_attempt3_start2_it0250.gif`: copied from docs/figures/rollouts/l2_attempt3_start2_it0250.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3366 kB
- `l2_attempt3_start2_it0300.gif`: copied from docs/figures/rollouts/l2_attempt3_start2_it0300.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3193 kB
- `l2_attempt3_start3_it0150.gif`: copied from docs/figures/rollouts/l2_attempt3_start3_it0150.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 1964 kB
- `l2_attempt3_start3_it0200.gif`: copied from docs/figures/rollouts/l2_attempt3_start3_it0200.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2052 kB
- `l2_attempt3_start3_it0250.gif`: copied from docs/figures/rollouts/l2_attempt3_start3_it0250.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2147 kB
- `l2_attempt3_start3_it0300.gif`: copied from docs/figures/rollouts/l2_attempt3_start3_it0300.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2088 kB
