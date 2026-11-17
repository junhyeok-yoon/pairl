# 20261115021024: cause of the slow reaching in attempt 2 (E1 relzero, E2 cost13, E3 init1_rev14, E4 init1_rev13, E5 init1_rev14_vnorm)

- Plan: `records/l2_attempt2_plan.md` (decisive experiments E1-E5), sha256 prefix 967b28c9
- Status: complete (40 iterations each); start 2026-11-15T02:10:24, end 2026-11-15T03:42:12
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: critic return scale identified (E5)

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_diag_relzero | 1-40 | 48.98 | 46.54 | 1.267 | 0 / 0 / 2.64e-11 |
| l2_diag_cost13 | 1-40 | 24.87 | 23.35 | 2.265 | 0 / 0 / 9.24e-07 |
| l2_diag_init1_rev14 | 1-40 | 49.63 | 47.36 | 1.332 | 0 / 0 / 4.83e-10 |
| l2_diag_init1_rev13 | 1-40 | 22.53 | 21.87 | 1.100 | 0 / 0 / 2.27e-05 |
| l2_diag_init1_rev14_vnorm | 1-40 | 49.03 | 46.59 | 1.206 | 0 / 0 / 1.76e-07 |

## Images

- [summary.png](summary.png)
- [l2_diag_cost13_start0_it0040.gif](l2_diag_cost13_start0_it0040.gif)
- [l2_diag_cost13_start2_it0040.gif](l2_diag_cost13_start2_it0040.gif)
- [l2_diag_cost13_start3_it0040.gif](l2_diag_cost13_start3_it0040.gif)
- [l2_diag_init1_rev13_start0_it0040.gif](l2_diag_init1_rev13_start0_it0040.gif)
- [l2_diag_init1_rev13_start2_it0040.gif](l2_diag_init1_rev13_start2_it0040.gif)
- [l2_diag_init1_rev13_start3_it0040.gif](l2_diag_init1_rev13_start3_it0040.gif)
- [l2_diag_init1_rev14_start0_it0040.gif](l2_diag_init1_rev14_start0_it0040.gif)
- [l2_diag_init1_rev14_start2_it0040.gif](l2_diag_init1_rev14_start2_it0040.gif)
- [l2_diag_init1_rev14_start3_it0040.gif](l2_diag_init1_rev14_start3_it0040.gif)
- [l2_diag_init1_rev14_vnorm_start0_it0040.gif](l2_diag_init1_rev14_vnorm_start0_it0040.gif)
- [l2_diag_init1_rev14_vnorm_start2_it0040.gif](l2_diag_init1_rev14_vnorm_start2_it0040.gif)
- [l2_diag_init1_rev14_vnorm_start3_it0040.gif](l2_diag_init1_rev14_vnorm_start3_it0040.gif)
- [l2_diag_relzero_start0_it0040.gif](l2_diag_relzero_start0_it0040.gif)
- [l2_diag_relzero_start2_it0040.gif](l2_diag_relzero_start2_it0040.gif)
- [l2_diag_relzero_start3_it0040.gif](l2_diag_relzero_start3_it0040.gif)

- `l2_diag_relzero_start0_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2963 kB
- `l2_diag_relzero_start2_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1853 kB
- `l2_diag_relzero_start3_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1012 kB
- `l2_diag_cost13_start0_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2699 kB
- `l2_diag_cost13_start2_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1804 kB
- `l2_diag_cost13_start3_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1111 kB
- `l2_diag_init1_rev14_start0_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2954 kB
- `l2_diag_init1_rev14_start2_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1738 kB
- `l2_diag_init1_rev14_start3_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1155 kB
- `l2_diag_init1_rev13_start0_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2437 kB
- `l2_diag_init1_rev13_start2_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1535 kB
- `l2_diag_init1_rev13_start3_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1148 kB
- `l2_diag_init1_rev14_vnorm_start0_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2494 kB
- `l2_diag_init1_rev14_vnorm_start2_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1592 kB
- `l2_diag_init1_rev14_vnorm_start3_it0040.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 835 kB
