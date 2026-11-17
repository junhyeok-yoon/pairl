# 20261115102001: pilot R: start-state distribution (revision 15)

- Plan: `records/l2_pilotR_plan.md`, sha256 prefix 352cad36
- Status: complete; start 2026-11-15T10:20:01, end 2026-11-15T10:59:50
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: rule (P) failed; restart share not adopted there

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotR_A | 301-355 | 39.70 | 38.04 | 1.914 | 0 / 0 / 2.56e-03 |
| l2_pilotR_B | 301-355 | 41.51 | 38.42 | 2.060 | 0 / 0 / 2.07e-03 |

## Images

- [summary.png](summary.png)
- [l2_pilotR_A_start0_it0355.gif](l2_pilotR_A_start0_it0355.gif)
- [l2_pilotR_A_start2_it0355.gif](l2_pilotR_A_start2_it0355.gif)
- [l2_pilotR_A_start3_it0355.gif](l2_pilotR_A_start3_it0355.gif)
- [l2_pilotR_B_start0_it0355.gif](l2_pilotR_B_start0_it0355.gif)
- [l2_pilotR_B_start2_it0355.gif](l2_pilotR_B_start2_it0355.gif)
- [l2_pilotR_B_start3_it0355.gif](l2_pilotR_B_start3_it0355.gif)

## GIFs

- `l2_pilotR_A_start0_it0355.gif`: copied from docs/figures/rollouts/l2_pilotR_A_start0_it0355.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3496 kB
- `l2_pilotR_A_start3_it0355.gif`: copied from docs/figures/rollouts/l2_pilotR_A_start3_it0355.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 1673 kB
- `l2_pilotR_B_start0_it0355.gif`: copied from docs/figures/rollouts/l2_pilotR_B_start0_it0355.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 3768 kB
- `l2_pilotR_B_start2_it0355.gif`: copied from docs/figures/rollouts/l2_pilotR_B_start2_it0355.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 2607 kB
- `l2_pilotR_B_start3_it0355.gif`: copied from docs/figures/rollouts/l2_pilotR_B_start3_it0355.gif (made on the cloud CPU with anim_rollout.py, bundle v3), 1738 kB
- `l2_pilotR_A_start2_it0355.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1916 kB
