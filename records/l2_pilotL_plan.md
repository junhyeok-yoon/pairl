# L2 pilot L: the transfer term L_tr in training (pre-registration)

Written 2026-11-16 before running; approved by the user on 2026-11-16 (option (B), extended to two weights). Platform: this PC's GPU, script/safe_run.sh.

## Background and status of the offline result
The offline test (records/l2_hand_ltr_plan.md, outcome hash 22986ee8) did not meet its registered rule: it passed on brains 140 and C200 but failed (b) on
N200 (role loss ratio 1.125 over the last 80 steps, limit 1.10), so by that rule L_tr was not proposed. Descriptively the residual fell to 0.47-0.65 on all
three brains while the role loss rose 4.7-7.5% on average. This online pilot is a new registered test, not a reinterpretation of the offline one.

## Question
With the brain configuration adopted in pilot O (arm NH: N_EP 1024, PPO minibatch 32768, entropy coefficient 1e-2), does adding lambda * L_tr to the hand's
role loss during training keep the hand's transfer residual low without raising the role loss by more than 10% and without making the brain worse?

## Common configuration
- Start: data/runs/20261116012508/l2_pilotO_NH/params_it0200.pkl, iterations 201-260 (60 iterations), brain and hand optimizer states carried over
  (as in NH). One seed (SEED 0 of l2_train.py).
- NH configuration: N_EP 1024 (512 original, 256 phase-2, 256 phase-3 starts), mb_scale=1 (minibatch 32768), c_ent=1e-2; everything else as attempt 3
  (cost revision 15, task action mode, role weighting, restart share 0.5, hand 16 Adam steps on 32 episodes per iteration, lr 1e-3, clip 1.0).
- L_tr: l2.transfer_loss (records/l2_hand_ltr_plan.md: 64 pairs per episode, same attachment state, eps = 1e-3 trace(G^T G)/3 + 1e-9), added to the role loss
  on the same 32-episode hand minibatch, keys PRNGKey(9000 + hand step). Only the hand's loss changes; the brain's loss is unchanged.

## Arms (one after another, in this order, one group created with group=new by the first)
- C: NH continued (no L_tr).
- L1: NH + 1.0 L_tr (ltr=1).
- L05: NH + 0.5 L_tr (ltr=0.5).
Command: sh script/safe_run.sh mem=16G python script/l2_train.py group=<new|G> l2_pilotL_<arm> 60 1024 6144 data/runs/20261116012508/l2_pilotO_NH/params_it0200.pkl
restart=0.5 eval_restart=1 mb_scale=1 c_ent=1e-2 plan=records/l2_pilotL_plan.md "purpose=pilot L: transfer term L_tr in training" [ltr=1 | ltr=0.5]

## Evaluation (as pilot O)
eval/diagnostics/l2_pilotO_eval.py at 260 of each arm, on the same GPU: 256 evaluation episodes per start (original, phase-2, phase-3), the same episodes and
keys as pilot O, deterministic for the decision, fully sampled secondary; results in records/l2_pilotO_eval.json (keys l2_pilotL_<arm>:260). Hand: median
transfer residual of l2.hand_diagnostics on the 256 original-start episodes at 260.

## Decision rule (per L_tr arm, paired with C at 260)
An L_tr arm passes if all hold:
  (a) its transfer residual at 260 is below C's at 260 and below 1.768 N m;
  (b) its role loss (hand_loss_all, the unweighted replayed role loss, mean over iterations 251-260) is at most 1.10 times C's over the same iterations;
  (c) brain non-inferiority: cost of the arm minus cost of C on the 768 paired evaluation episodes (deterministic), upper bound of the bootstrap 95% CI
      (10000 resamples, numpy default_rng(0), percentile) below +0.5;
  (d) (G) on every iteration 201-260 (violations 0, non-finite 0, tank-bound margin at least -0.01 J on episodes without saturation below T_min).
If both L_tr arms pass, the smaller weight (0.5) is adopted. If none passes, attempt 4 runs without L_tr.
Descriptive: L_tr per iteration (hand_ltr), residual at the built-in evaluations (64 episodes, 210-260), the paired cost difference per start, KL, clip
share, sigma, accurate presses, grasp, hold error, fully sampled evaluation.

## Visual summaries
In the group folder (summary.png, eval_<run>_it0260.png, decision.png with the L arms against C, GIFs of the final checkpoints, summary.md), made by
script/finish_run.sh after each arm.

## Budget
GPU: NH ran at 23.0 s per iteration; L_tr adds the pair terms to the hand step (measured on the first arm with L_tr and reported). About 25 min per arm,
about 1.3 h for three arms with evaluations.

## Outcome
(to be appended)

## Amendment 1 (2026-11-16, before the L_tr arms; written after arm C finished)
Two findings during the pilot, and what follows.
1. Arm L1 was refused at launch because the training code had changed after arm C (code added for pilot G: env.near_start, the revision-17 speed term,
   the tip-load observation; all behind options not used here). Check: the pre-change code was reconstructed exactly (its code-state hash equals the
   group's, e1480ff5...), and on this pilot's control configuration (NH settings, no L_tr; 2 iterations from NH 200, CPU) the old and the new code give
   bit-identical parameters, optimizer states, hist.json and evals.json.
2. The L_tr path is not run-to-run deterministic with the default XLA settings: two runs of the same code and inputs (2 iterations from NH 200, ltr=1) differ
   from the second iteration on, on the CPU (also single-threaded) and on the GPU. Without L_tr the runs are bit-identical. With
   XLA_FLAGS=--xla_gpu_deterministic_ops=true two GPU runs with L_tr are bit-identical (role loss 12.227 both at 202); the flag also changes the GPU
   rollouts at the rounding level (port check means 17.330616, 6.421522, 4.681267 with the flag against 17.369957, 6.421250, 4.681267 without), so it is a
   platform setting of its own, and it slows the hand step (21.2 -> 33.2 s per iteration at N_EP 128).
What follows: the pilot is rerun as a whole, in a new group, with XLA_FLAGS=--xla_gpu_deterministic_ops=true for training and evaluation: arms C, L1, L05
as registered (same start, configuration, evaluation and rules). The decision evaluation of every arm runs with the same flag. The earlier arm C (group
20261116140005, no flag, pre-change code) is not used for the decision; it is kept as a descriptive comparison of the flag's effect on the control path.
From this pilot on, all runs and decision evaluations use this setting, and comparisons are made within it (references re-evaluated under it when needed).
Run names of the rerun: l2_pilotLd_C, l2_pilotLd_L1, l2_pilotLd_L05 (run names must be unique across groups; the superseded group holds l2_pilotL_C).

## Outcome (2026-11-16; group data/runs/20261116145109, GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true; records/l2_pilotL_decision.json)
Run as amended (arms l2_pilotLd_C, _L1, _L05, 201-260). (G) on every iteration of every arm: violations 0, non-finite 0, no saturation below T_min;
minimum tank-bound margin C 0.0106, L1 0.0168, L05 0.0140 J.
| arm | transfer residual at 260 (C: 6.081) | role loss 251-260 (C: 9.125), ratio | cost minus C, 95% CI | (a) | (b) | (c) | (d) | pass |
|---|---|---|---|---|---|---|---|---|
| L1 (lambda 1) | 0.579 | 11.344, 1.243 | -0.076 [-0.165, -0.004] | yes | no | yes | yes | no |
| L05 (lambda 0.5) | 0.686 | 10.993, 1.205 | -0.121 [-0.233, -0.035] | yes | no | yes | yes | no |
No L_tr arm passes (rule (b), role loss more than 1.10 times C's). By the registered rule BASE = NH alone (l2_pilotLd_C at 260) and pilot G starts from it.
Descriptive, phase success (records/phase_success.json; this PC's CPU, deterministic; paired with C on the same episodes, bootstrap 95% CI):
- L1 minus C: P1 +0.051 [+0.020, +0.086], P2 -0.098 [-0.152, -0.047], P3 -0.023 [-0.074, +0.027], P3' +0.000 [-0.030, +0.030], P4 -0.090
  [-0.133, -0.047], chain P1&P2 +0.023 [-0.016, +0.062], P1-P3 +0.012 [-0.016, +0.039], full chain 0 in both.
- L05 minus C: P1 +0.039 [+0.012, +0.070], P2 -0.031 [-0.074, +0.016], P3 -0.031 [-0.078, +0.012], P3' +0.008 [-0.013, +0.030], P4 +0.098
  [+0.051, +0.145], chain P1&P2 +0.027 [-0.008, +0.062], P1-P3 -0.008 [-0.031, +0.016], full chain 0 in both.
- Absolute (deterministic) at 260, C / L1 / L05: P1 0.309 / 0.359 / 0.348; P2 0.719 / 0.621 / 0.688; P3 0.207 / 0.184 / 0.176; P3' 0.657 / 0.635 /
  0.654; P4 0.266 / 0.176 / 0.363; full chain 0.000 in all three (L1 reference 0.922).
Reading (not registered): L_tr keeps the transfer residual low online (0.58-0.69 against 6.08 for C, whose residual drifted from 1.357 at 140), at a role-
loss cost of 20-24%, above the registered 10%; the brain is not worse (cost CIs below 0). The hand's two objectives conflict at this weight; the role
loss alone lets the transfer drift. This stays open for the spec revision.

## Amendment 2 (2026-11-16, before running; approved by the user): arm L025
The registered outcome stands (L_tr not adopted under the 1.10 rule). One more arm under the same registered rule and settings:
- L025: NH + 0.25 L_tr (ltr=0.25), start l2_pilotO_NH at 200 with the optimizer states carried over, iterations 201-260, deterministic XLA flag, run name
  l2_pilotLd_L025; decided against l2_pilotLd_C at 260 by rules (a)-(d) unchanged; the same evaluation and phase success report.
- Group: a new group (data/runs/<timestamp>/, group=new), because the code changed after this pilot's group started (l2.rollout mode 2 and the gate value
  alpha in the per-step record, added during arm L05 for the evaluation tools; inert for training, sixth gate). Before the arm, a check on the GPU with the
  flag: the current code, run for 2 iterations in C's configuration from NH 200 (temporary directory), must reproduce the hist.json entries of
  l2_pilotLd_C at 201 and 202 exactly (all fields apart from t_it and t_roll). If it does not, C is rerun in the new group with the current code before L025
  and L025 is paired with that rerun.
- Order: after all arms of pilot G.
- What follows: if L025 passes, BASE for attempt 4 = NH + L_tr(0.25), started from l2_pilotLd_L025 at 260, with the changes adopted in pilot G added (as
  instructed by the user). If it fails, the chain stops before attempt 4 and reports to the user (the hand's transfer residual is then the main open
  role; the user decides between going on without L_tr and a newly registered criterion based on task-relevant hand error).
Note on pilot G: pilot G starts from l2_pilotLd_C at 260 (BASE = NH, the registered pilot-L outcome). If L025 is adopted, the adopted pilot-G changes are
added on L025's checkpoint for attempt 4, as instructed.

## Outcome of amendment 2 (arm L025; 2026-11-16; new group, GPU with the deterministic flag)
Check before the arm: the current code reproduced the hist.json entries of l2_pilotLd_C at 201 and 202 exactly (GPU, deterministic flag), so L025 is
paired with l2_pilotLd_C at 260. (G) on 201-260: violations 0, non-finite 0, no saturation below T_min, minimum tank-bound margin 0.0098 J.
L025 (lambda 0.25): (a) transfer residual 0.863 against C's 6.081 and 1.768 holds; (b) role loss 10.742 against C's 9.125, ratio 1.177 > 1.10 fails;
(c) cost minus C +0.193 [+0.078, +0.336] holds (non-inferiority bound +0.5); (d) holds. L025 is not adopted.
Across the weights (role-loss ratio against C over 251-260): lambda 1.0 1.243, 0.5 1.205, 0.25 1.177; transfer residual at 260: 0.579, 0.686, 0.863
(C 6.081). Phase success, L025 minus C (deterministic, CPU): P1 -0.016 [-0.043, +0.012], P2 +0.047 [+0.016, +0.082], P3 -0.016 [-0.062, +0.031], P3'
-0.013 [-0.043, +0.017], P4 -0.148 [-0.195, -0.105], chains with CIs containing 0 (full chain 0 in both).
What follows (amendment 2): the chain stops before attempt 4 and reports to the user.
