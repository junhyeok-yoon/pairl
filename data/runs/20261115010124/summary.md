# 20261115010124: L2 attempt 2 (revision 14)

- Plan: `records/l2_attempt2_plan.md` (configuration and amendments 1-5), sha256 prefix 967b28c9
- Status: complete (300); start 2026-11-15T01:01:24, end 2026-11-15T06:29:26
- Platform: CPU, cloud container, 2 cores (Xeon 2.1 GHz), 7 GB RAM; records/port_check_cloud.log
- Code: 20261115_v3_brain_hand_full_bundle.zip, code/stageb (bundle v3)
- Key result: pressing 0%, (H1) not met

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_attempt2a | 1-60 | 52.01 | 51.01 | 1.184 | 0 / 0 / 8.06e-05 |
| l2_attempt2b | 1-59 | 50.18 | 48.98 | 1.639 | 0 / 0 / 5.93e-09 |
| l2_attempt2 | 51-300 | 41.52 | 39.48 | 2.947 | 0 / 0 / 1.49e-07 |

## Images

- [summary.png](summary.png)
- [l2_attempt2_start0_it0100.gif](l2_attempt2_start0_it0100.gif)
- [l2_attempt2_start0_it0150.gif](l2_attempt2_start0_it0150.gif)
- [l2_attempt2_start0_it0200.gif](l2_attempt2_start0_it0200.gif)
- [l2_attempt2_start0_it0250.gif](l2_attempt2_start0_it0250.gif)
- [l2_attempt2_start0_it0300.gif](l2_attempt2_start0_it0300.gif)
- [l2_attempt2_start2_it0300.gif](l2_attempt2_start2_it0300.gif)
- [l2_attempt2_start3_it0300.gif](l2_attempt2_start3_it0300.gif)
- [l2_attempt2a_start0_it0050.gif](l2_attempt2a_start0_it0050.gif)
- [l2_attempt2a_start0_it0060.gif](l2_attempt2a_start0_it0060.gif)
- [l2_attempt2a_start2_it0060.gif](l2_attempt2a_start2_it0060.gif)
- [l2_attempt2a_start3_it0060.gif](l2_attempt2a_start3_it0060.gif)
- [l2_attempt2b_start0_it0050.gif](l2_attempt2b_start0_it0050.gif)
- [l2_attempt2b_start2_it0050.gif](l2_attempt2b_start2_it0050.gif)
- [l2_attempt2b_start3_it0050.gif](l2_attempt2b_start3_it0050.gif)

## GIFs

- `l2_attempt2_start0_it0100.gif`: copied from docs/figures/rollouts/l2_attempt2_rollout_it0100.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 4754 kB
- `l2_attempt2_start0_it0150.gif`: copied from docs/figures/rollouts/l2_attempt2_rollout_it0150.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 4694 kB
- `l2_attempt2_start0_it0200.gif`: copied from docs/figures/rollouts/l2_attempt2_rollout_it0200.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 4055 kB
- `l2_attempt2_start0_it0250.gif`: copied from docs/figures/rollouts/l2_attempt2_rollout_it0250.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 3648 kB
- `l2_attempt2_start0_it0300.gif`: copied from docs/figures/rollouts/l2_attempt2_rollout_it0300.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 4072 kB
- `l2_attempt2a_start0_it0050.gif`: copied from docs/figures/rollouts/l2_attempt2a_rollout_it0050.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 3631 kB
- `l2_attempt2b_start0_it0050.gif`: copied from docs/figures/rollouts/l2_attempt2b_rollout_it0050.gif (made on the cloud CPU with anim_rollout.py, bundle v3); original start, 4513 kB
- `l2_attempt2a_start0_it0060.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2242 kB
- `l2_attempt2a_start2_it0060.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1937 kB
- `l2_attempt2a_start3_it0060.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1348 kB
- `l2_attempt2b_start2_it0050.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1688 kB
- `l2_attempt2b_start3_it0050.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1415 kB
- `l2_attempt2_start2_it0300.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1755 kB
- `l2_attempt2_start3_it0300.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 944 kB
