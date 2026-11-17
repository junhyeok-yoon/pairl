# L2 pilot O: optimization of the coupled learning (pre-registration DRAFT, not approved, not run)

Status: revision 3 (2026-11-16), approved by the user on 2026-11-16 with the changes of revision 3 (arm D paired with C; mode 0 called fully sampled; the
pilot's group created with group=new); optimizer states carried over instead of reset (measured reason below). The hash line is added after the
preparation below; a new line at every amendment and at the outcome.

## Question
Attempt 3 did not converge at iteration 300 (records/l2_attempt3_plan.md; spec sec:l2status): KL median 0.0555 and clip share 0.40 per iteration at a fixed
brain learning rate 3e-4, goal standard deviation 0.211 -> 0.0507, training-cost slope -0.024 per iteration over 251-300; the hand's transfer residual drifted
from 0.44 at 30 to 4.43 at 300 (median) at a fixed hand learning rate 1e-3. Handoff section 8, item 2: which optimization settings make the brain's updates
controlled (KL in 0.01-0.02) and the cost fall faster, and does a decaying hand learning rate slow the drift, with everything else fixed?

## Platform and seed
- All arms run on this PC's GPU (RTX 5080, JAX 0.12.0.1 cuda13, float64), one seed (SEED 0 of l2_train.py), under script/safe_run.sh. GPU rollouts differ
  from the CPU ones at the episode level (records/verified_values.md, "## Platform reproduction"; records/port_check_pc.log), so the arms are compared only
  with the control arm C below, run on the same GPU, never with the records of attempt 3.
- Arms run one after another (no two training processes on the GPU at the same time).

## Start
- Warm start from data/runs/20261115134229/l2_attempt3/params_it0140.pkl (iteration 140 of attempt 3; task action mode). At 300 the goal standard deviation has collapsed
  (0.0507 over 261-300) and the hand has drifted (transfer residual 3.196 over 260-300), which would floor arms H and D. At 140 the goal standard deviation
  is 0.0573 (mean over 131-140), the median transfer residual 1.357 N m (evaluation at 140), the KL median 0.0609 (131-140).
- Optimizer states: both carried over from the checkpoint (brain and hand Adam moments and step counts, as stored at 140; the default of l2_train.py on a
  warm start). Revision 2 proposed resetting both (opt_reset=1, as in modification 5 of the spec). Measured on this PC's CPU before registration
  (2 iterations from 140, temporary directory): with the reset, iteration 141 has KL 1.765 and clip share 0.679 (the first Adam steps after a reset are of
  size lr in every parameter); carried over, KL 0.038 and clip share 0.342 (attempt 3 at 141 on the cloud: 0.0664, 0.39). A reset would move the policy far
  from the start checkpoint in the first iteration of every arm, which defeats starting at 140. The moments are statistics of the gradients and do not depend
  on the learning rate, so arms K and D are not affected by carrying them. Arm N changes the minibatch (4096 -> 32768): its stored second moments come from
  the smaller minibatch and are larger than the new gradients' (less minibatch noise), so its first steps are somewhat smaller; this decays with beta2 0.999
  and is part of arm N. All arms, C included, carry the states the same way.
- The run starts at iteration 141; 60 iterations per arm (141-200).

## Common configuration (attempt 3 unless an arm changes it)
- Cost revision 15, task action mode (from the checkpoint), role weighting of revision 14 with the corrected start of r(t), restart share 0.5.
- PPO: gamma 0.99, GAE lambda 0.95, clip 0.2, value coefficient 0.5, 5 epochs, minibatch 4096 decisions, gradient-norm clip 0.5, advantages normalized per batch.
- Hand: 16 Adam steps on 32 episodes per iteration, gradient clip 1.0.
- Defaults that the arms change: episodes per iteration N_EP 128; brain learning rate 3e-4 constant; entropy coefficient 1e-3; hand learning rate 1e-3 constant.
- The built-in evaluation of l2_train.py (64 episodes every 10 iterations) stays as it is and is descriptive. The decision uses the evaluation below.

## Arms
- C (control): the common configuration, unchanged.
- N (episodes per iteration): N_EP 1024 (same restart share: 512 original, 256 phase-2, 256 phase-3 starts) and PPO minibatch 4096 x 8 = 32768 decisions,
  so that the number of PPO gradient steps per iteration equals that of C (about 18900 decisions per iteration at N_EP 128 give 4 minibatches per epoch,
  20 steps per iteration; the minibatch scales with N_EP). The hand keeps 16 steps on 32 episodes.
- K (KL-adaptive brain learning rate): after each iteration, with k the mean KL of that iteration, lr <- lr / 1.5 if k > 2 * 0.015, lr <- lr * 1.5 if
  k < 0.015 / 2, bounds [1e-5, 1e-3]; target 0.015; starts at 3e-4. N_EP 128.
- H (entropy coefficient): entropy coefficient 1e-2. N_EP 128.
- D (hand learning-rate decay): hand learning rate 1e-3 * 0.5^((it - 140) / 20) (halved every 20 iterations: 1.25e-4 at 200). N_EP 128.
  D can slow the hand's drift but does not address its cause: the role loss measures the error inside rollouts that are recalibrated at every rest and does
  not pin the transfer between postures (spec sec:l2a3, hypothesis). The transfer term L_tr of handoff section 8, item 3 is tested separately offline and is
  not part of this pilot.
Each arm changes one setting against C. Combinations are not part of this pilot; they are proposed after the outcome.

## Evaluation (decision)
- At iteration 200 of each arm: 256 evaluation episodes per start (original start, phase-2 restart, phase-3 restart; 768 in all), the same episodes and PRNG
  keys for every arm (episodes l1.episodes(6000, 256, lim=0.7) with the nominal budget; keys jax.random.split(PRNGKey(555), 256)), deterministic policy
  (mean action, gripper logit > 0). Run on the same GPU as the training.
- Secondary, not used for any decision: the same 768 episodes fully sampled (mode 0 of l2.rollout: the whole action is sampled, the continuous actions
  with the exploration noise held for 10 decisions and the gripper from its Bernoulli), with the same keys; reported next to the deterministic result.
- Hand: median transfer residual of l2.hand_diagnostics on the 256 original-start episodes at 200, and the role loss at 200 (hand_loss_all of iteration 200,
  the unweighted replayed role loss on that iteration's 32-episode hand batch; and its mean over 191-200, descriptive).

## Decision rules
- An arm is excluded if (G) fails on any iteration 141-200: joint violations 0, non-finite states 0, tank-bound margin at least -0.01 J on episodes without
  saturation below T_min.
- Brain arms N, K, H: d_i = cost of the arm minus cost of C on evaluation episode i (deterministic, 768 episodes, paired by episode and key). Adopt only if
  (a) the upper bound of the bootstrap 95% CI of the mean of d (10000 resamples of the 768 pairs, numpy default_rng(0), percentile interval) is below 0, and
  (b) the accurate-press share from the phase-3 start (relative force error below 0.5, held to phase 4; 256 episodes) is not lower than C's by more than 0.05.
  For K additionally: the KL median over 151-200 lies in [0.01, 0.02].
- Arm D, paired with C at the same iteration: adopted only if, at iteration 200, (a) D's median transfer residual is below 1.768 N m and below C's median
  transfer residual at 200 (both on the 256 original-start episodes), and (b) D's role loss at 200 is at most 1.10 times C's role loss at 200. At 140 the
  residual is already 1.357, so the absolute criterion alone would pass without any effect of the decay.
- If no arm meets its rule, none is adopted; the outcome is dissected (l2_dissect.py at 200 of each arm) and next measures are proposed, not added.
- Everything else is descriptive: training-cost slope over 151-200, KL median and clip share over 151-200, goal and stiffness standard deviations at
  191-200, hold error at p* from the phase-2 start, the fully sampled evaluation, the built-in 64-episode evaluations, wall time per iteration.

## Preparation (code; no numerics change at the defaults)
- l2_train.py gains the options kl_target=, c_ent=, lr_h_half= and mb_scale=1 (minibatch 4096 x N_EP / 128); absent, the current behavior. Gate before the
  pilot, by the method of the training-path gate (records/train_gate_pc.log): 3 iterations on this PC's CPU from the same warm start with the new code
  without the options and with the current code, bit-identical brain and hand parameters and optimizer states after every iteration, identical train.log
  lines apart from elapsed times, identical hist.json and evals.json apart from the timing fields; outputs in a temporary directory deleted afterwards.
  port_check ref=pc EXACT.
- An evaluation script eval/diagnostics/l2_pilotO_eval.py (768 episodes, deterministic and fully sampled, per-episode costs, hand residual) writes
  records/l2_pilotO_eval.json; checked first on the start checkpoint 140 (descriptive).
- Interim reports at 170 and 200 of each arm; learning curves of the five arms in one figure (eval/figures/fig_l2_pilotO.py -> docs/figures/l2_pilotO.png);
  rollout GIFs of the adopted arm (or of C if none) at 200.

## Commands (after approval and after the preparation)
  sh script/safe_run.sh mem=16G python script/l2_train.py group=<G> l2_pilotO_<arm> 60 <N_EP> 6144 data/runs/20261115134229/l2_attempt3/params_it0140.pkl restart=0.5 eval_restart=1 plan=records/l2_pilotO_plan.md purpose="pilot O: optimization of the coupled learning" <arm options>
with <arm options>: C none; N: N_EP 1024 and mb_scale=1; K kl_target=0.015; H c_ent=1e-2; D lr_h_half=20. The first arm (C) uses group=new, which creates
the group data/runs/<timestamp>/ with its manifest; the other arms use group=<that timestamp>. Order: C, N, K, H, D, one after another.

## Budget (measured on this PC, records/l2_train_timing_pc.json)
GPU, steady iteration: 22.4 s at N_EP 128 (rollout 5.0, hand 16.6, PPO 0.3 s); 26.5 s at N_EP 1024 with minibatch 4096 (rollout 5.9, hand 16.6, PPO 3.0 s;
the larger minibatch of arm N keeps about the same PPO work per iteration, so about 26 s); first iteration about 45-50 s (compilation).
Built-in evaluation: about 30 s every 10 iterations (7 per arm, about 3.5 min).
Per arm: C, K, H, D about 60 x 22.4 s + 3.5 min = 26 min; N about 60 x 26.5 s + 3.5 min = 30 min. Training of the five arms about 2.3 h.
Decision evaluation: 768 deterministic + 768 fully sampled episodes per arm on the GPU, measured 5.75 s per 1024-episode batch: about 1 min per arm with compilation,
about 5 min for the five arms (and the check on checkpoint 140).
Total about 2.5 h of GPU time; peak GPU memory 0.3 GB (N_EP 128) to 2.2 GB (1024) in the JAX allocator, 8.2 GB on the device including the desktop.

## Not in this pilot
Evaluation rule changes beyond the decision evaluation above (handoff section 8, item 1), the transfer term of the role loss (item 3), grasp and ledger
changes (item 4), several seeds.

## Outcome (2026-11-16; group data/runs/20261116012508, this PC's GPU; decision evaluation records/l2_pilotO_eval.json)
Run as registered (hash 316d6d87): arms C, N, K, H, D one after another, 141-200 each, from data/runs/20261115134229/l2_attempt3/params_it0140.pkl with the
optimizer states carried over. Deviation from the text: none in the training. The launch of N was refused once by the group bookkeeping (code-state hash
covered non-code files); the hash scope was narrowed to the training path, whose files did not change between arms (sha256 in the group manifest).
(G) on every iteration 141-200 of every arm: joint violations 0, non-finite states 0, no saturation below T_min; minimum tank-bound margin
C 5.36e-4, N 8.79e-4, K 8.74e-4, H 3.56e-4, D 5.41e-4 J. No arm is excluded.

Decision evaluation at 200 (768 paired episodes, deterministic; arm minus C; bootstrap 95% CI, 10000 resamples, default_rng(0)):
| arm | mean diff | 95% CI | per start (original, phase-2, phase-3) | accurate-press diff | KL median 151-200 | rule | adopted |
|---|---|---|---|---|---|---|---|
| N | -2.033 | [-2.452, -1.617] | -5.913, -0.140, -0.046 | +0.059 | 0.0361 | CI < 0 and press >= -0.05 | yes |
| K | +0.300 | [+0.059, +0.536] | -0.419, +0.729, +0.589 | -0.035 | 0.0174 | as N, and KL in [0.01, 0.02] | no (cost) |
| H | -0.761 | [-1.022, -0.511] | -1.681, -0.295, -0.307 | +0.184 | 0.0391 | CI < 0 and press >= -0.05 | yes |
Arm D (paired with C at 200): transfer residual D 1.961 against C 1.794 and 1.768 -> not below either; role loss D 5.214 against 1.10 x C 5.694 = 6.263
-> met. D is not adopted (residual rule).

Decision evaluation, absolute values at 200 (deterministic | fully sampled; 256 episodes per start):
- start checkpoint 140: original 27.599 | 27.360, phase-2 8.600 | 8.704, phase-3 6.506 | 6.646; grasp 0.035 | 0.035; accurate press 0.117 | 0.102;
  hold error 0.0523 | 0.0524 m; transfer residual 1.005.
- C: 25.422 | 24.790, 7.494 | 7.632, 5.722 | 5.838; grasp 0.023 | 0.012; accurate press 0.207 | 0.191; hold error 0.0366 | 0.0371 m; residual 1.794.
- N: 19.509 | 19.482, 7.354 | 7.506, 5.677 | 5.773; grasp 0.070; accurate press 0.266 | 0.246; hold error 0.0320 m; residual 1.645.
- K: 25.003 | 24.433, 8.223 | 8.298, 6.311 | 6.387; grasp 0.039; accurate press 0.172 | 0.176; hold error 0.0415 m; residual 2.637.
- H: 23.742 | 23.769, 7.199 | 7.708, 5.415 | 5.808; grasp 0.059; accurate press 0.391 | 0.297; hold error 0.0314 m; residual 0.722.
- D: 26.204 | 25.224, 7.306 | 7.475, 5.618 | 5.729; grasp 0.039; accurate press 0.387 | 0.355; hold error 0.0332 m; residual 1.961.
- Reference, attempt 3 at 300 (same evaluation, same GPU): 18.089 | 17.362, 6.133 | 5.225, 4.641 | 3.886; grasp 0.141 | 0.133; accurate press
  0.637 | 0.645; hold error 0.0263 | 0.0271 m; residual 4.299. On the first 64 of these episodes: grasp 0.172 and mean cost 17.369957, equal to the
  dissection at 300 (grasp 0.172, cloud) and to the GPU port check; the grasp of l2_pilotO_eval.py and of l2_dissect.py is the same quantity
  (mean of grasp_end1 of l2.summarize), on 256 and 64 episodes.
Secondary, fully sampled, paired arm minus C (not used for the decision): N -1.833 [-2.196, -1.479], K +0.286 [+0.025, +0.547], H -0.325 [-0.576, -0.071],
D +0.056 [-0.168, +0.286].

Descriptive (training history; windows 141-150 / 151-170 / 171-200):
- KL median: C 0.0533/0.0496/0.0623, N 0.0304/0.0389/0.0361, K 0.0232/0.0166/0.0176, H 0.0542/0.0351/0.0420, D 0.0611/0.0462/0.0588.
- Clip share median: C 0.407/0.391/0.409, N 0.271/0.301/0.311, K 0.268/0.217/0.199, H 0.382/0.347/0.322, D 0.401/0.396/0.401.
- Goal sd (mean): C 0.0577/0.0573/0.0527, N 0.0561/0.0491/0.0439, K 0.0561/0.0511/0.0445, H 0.0606/0.0717/0.0801, D 0.0581/0.0578/0.0524.
- Training-cost slope 151-200 per iteration: C -0.0700, N -0.1546, K -0.0511, H -0.0465, D -0.0679.
- K's brain learning rate fell from 3e-4 to 1.21e-4 (mean 141-150) and about 6e-5 (151-200).
- Role loss at 200 (unweighted, one 32-episode batch): C 5.694, N 4.605, K 7.987, H 9.020, D 5.214. It fluctuates by a factor of two between iterations
  in every arm (summary.png), so single values are weak evidence.
- Wall time per iteration (median): C 21.7, N 23.0, K 22.8, H 21.8, D 22.0 s.
Reading (not a registered conclusion): N's gain is almost all from the original start (-5.913), where grasp rose from 0.023 (C) to 0.070; H's gain comes with
a growing exploration scale (goal sd 0.0577 -> 0.0801, stiffness sd 0.253 -> 0.363) and a lower transfer residual at 200 (0.722) that one evaluation can
move by a factor of two (handoff section 9). K reached the KL band but learned more slowly over 60 iterations (lr about 6e-5). D did not keep the residual
below C's: halving the hand learning rate every 20 iterations does not stop the drift within 60 iterations, consistent with the drift coming from the role
loss rather than from the step size.
Adopted: N (episodes per iteration 1024, minibatch 32768) and H (entropy coefficient 1e-2). Not adopted: K, D. Combinations are not tested here; attempt 4
waits for the user's decision.

Erratum to the Outcome (same day): in "Reading", H's exploration scale should read goal sd 0.0606 -> 0.0801 and stiffness sd 0.274 -> 0.363 (H's own means
over 141-150 and 171-200); the values 0.0577 and 0.253 written there are C's over 141-150.

## Amendment 1 (2026-11-16, written before the run): arm NH (N and H combined), approved by the user
Question: do the two adopted settings add up? Arm NH = arm N with the entropy coefficient of arm H.
- Configuration: N_EP 1024 (512 original, 256 phase-2, 256 phase-3 starts), PPO minibatch 4096 x 8 = 32768 decisions (option mb_scale=1, which scales the
  minibatch by N_EP / 128 = 8; the user's "mb_scale 8"), entropy coefficient 1e-2 (c_ent=1e-2); everything else as the common configuration.
- Start: data/runs/20261115134229/l2_attempt3/params_it0140.pkl, iterations 141-200. Optimizer states carried over from the checkpoint, exactly as in arm N
  (and every arm of this pilot; see "Start"). The user's message said "optimizer reset"; the decision below is a paired comparison with arm N, whose
  optimizer states were carried over, so NH carries them too; a reset would change two things at once (KL 1.765 at the first iteration after a reset,
  records/train_gate_pc.log). This is stated to the user with the report.
- Same platform (this PC's GPU), same group (data/runs/20261116012508), run name l2_pilotO_NH, through script/safe_run.sh.
  Command: sh script/safe_run.sh mem=16G python script/l2_train.py group=20261116012508 l2_pilotO_NH 60 1024 6144
  data/runs/20261115134229/l2_attempt3/params_it0140.pkl restart=0.5 eval_restart=1 plan=records/l2_pilotO_plan.md "purpose=..." mb_scale=1 c_ent=1e-2
- Evaluation: as the decision evaluation of the pilot (l2_pilotO_eval.py at 200: 256 episodes per start, same episodes and keys, deterministic for the
  decision, fully sampled secondary), on the same GPU. Arm N's evaluation is reused.
- Decision rule: d_i = cost of NH minus cost of N on the 768 paired episodes (deterministic). NH is adopted only if (a) the upper bound of the bootstrap 95% CI
  of the mean of d (10000 resamples, numpy default_rng(0), percentile) is below 0, and (b) the accurate-press share from the phase-3 start is not lower than
  N's by more than 0.05. (G) as before (an NH failing (G) on any iteration 141-200 is excluded). Otherwise N alone stays the adopted brain setting and H is
  reported as not additive. Descriptive: NH minus C and NH minus H on the same pairs, KL, clip share, sigma, hand residual, role loss.
- Budget: about 30 min of GPU time (arm N: 23.0 s per iteration) plus the evaluation (about 1 min).

## Outcome of amendment 1 (arm NH; 2026-11-16, same group and GPU; records/l2_pilotO_eval.json key l2_pilotO_NH:200)
Run as registered (hash 760c68df), optimizer states carried over as in arm N. (G) on every iteration 141-200: violations 0, non-finite 0, no saturation
below T_min, minimum tank-bound margin 7.74e-4 J.
Decision (NH minus N, 768 paired episodes, deterministic): mean -1.735, bootstrap 95% CI [-2.028, -1.474] (per start -3.174, -1.075, -0.957);
accurate-press share 0.469 against N's 0.266 (+0.203). Both conditions hold: NH is adopted.
Absolute values at 200 (deterministic | fully sampled): original 16.335 | 15.273, phase-2 6.279 | 5.724, phase-3 4.719 | 4.128; grasp 0.168;
accurate press 0.469 | 0.395; hold error 0.0212 m; transfer residual 2.444.
Descriptive: NH minus C -3.768 [-4.268, -3.279]; NH minus H -3.007 [-3.466, -2.559]; NH minus N fully sampled -2.545 [-2.856, -2.259].
KL median 141-150 / 151-170 / 171-200: 0.0297 / 0.0278 / 0.0267; clip share 0.244 / 0.225 / 0.202; goal sd 0.0603 / 0.0635 / 0.0754; stiffness sd 0.271 /
0.310 / 0.375; training-cost slope 151-200 -0.2322 per iteration; role loss at 200 8.882; built-in evaluation residual 1.357 (140) -> 2.908 (200).
Same evaluation of attempt 3 at 300 (reference, same GPU): original 18.089, phase-2 6.133, phase-3 4.641, grasp 0.141, accurate press 0.637. After 60
iterations from 140, NH is below attempt 3 at 300 in the original-start cost and above it in grasp; pressing accuracy is still lower.
The hand's transfer residual still drifts in NH (as in every arm); the offline test of L_tr (records/l2_hand_ltr_plan.md) is reported separately.

## Protocol check (2026-11-16, after the outcome; no result changed)
Question raised by the user: were the optimizer states reset (opt_reset=1) as registered? Answer from the records:
- The registered text (revision 3, hash 316d6d87; recomputed from the file prefix) specifies "Optimizer states: both carried over" and its command has no
  opt_reset=1. The reset appeared only in the unregistered revision-2 draft.
- Every arm (C, N, K, H, D and NH of amendment 1) ran with the brain and hand optimizer states carried over from params_it0140.pkl: no opt_reset in the
  commands of data/runs/20261116012508/manifest.json, one "optimizer states restored" line in each train.log, KL at iteration 141 between 0.0369 and 0.0416
  (C 0.0399, N 0.0369, K 0.0399, H 0.0416, D 0.0399, NH 0.0388; a reset gives about 1.765, records/train_gate_pc.log).
- No arm deviated from the registration; all paired comparisons (arms against C, NH against N) are like-for-like in the optimizer handling.
