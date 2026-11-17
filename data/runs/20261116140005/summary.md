# 20261116140005: pilot L: transfer term L_tr in training

- Plan: `records/l2_pilotL_plan.md`, sha256 prefix 771ec94d
- Status: complete; start 2026-11-16T14:00:05, end 2026-11-16T14:28:06
- Platform: GPU, Intel(R) Core(TM) Ultra 9 285K
- Code: commit 81f2173d028a8e3b4028849b07c8b145954e4f21, dirty, diff e1480ff5464a

## Runs

| run | iterations | final training cost (5-it mean) | final eval cost, original (64 ep.) | hand residual (final eval) | (G): viol / nan / min tank margin |
|---|---|---|---|---|---|
| l2_pilotL_C | 201-260 | 13.96 | 11.98 | 6.457 | 0 / 0 / 1.20e-02 |

## Decision evaluation (256 episodes per start, records/l2_pilotO_eval.json)

| run | original det. / sampled | phase-2 det. / sampled | phase-3 det. / sampled | grasp (orig., det.) | accurate press (det.) | hold error (det.) | hand residual |
|---|---|---|---|---|---|---|---|
| l2_pilotL_C:260 | 12.528 / 12.534 | 4.329 / 4.533 | 2.973 / 3.240 | 0.328 | 0.723 | 0.0140 | 5.912 |

## Images

- [summary.png](summary.png)
- [eval_l2_pilotL_C_it0260.png](eval_l2_pilotL_C_it0260.png)
- [l2_pilotL_C_start0_it0260.gif](l2_pilotL_C_start0_it0260.gif)
- [l2_pilotL_C_start2_it0260.gif](l2_pilotL_C_start2_it0260.gif)
- [l2_pilotL_C_start3_it0260.gif](l2_pilotL_C_start3_it0260.gif)

- `l2_pilotL_C_start0_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 2517 kB
- `l2_pilotL_C_start2_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1895 kB
- `l2_pilotL_C_start3_it0260.gif`: made 2026-11-16 on this PC's CPU (anim_rollout.py, deterministic, 4 evaluation episodes, frame stride 3), 1091 kB
