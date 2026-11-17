# L2 attempt 2: coupled learning under spec revision 14, one seed, from initialization

Written 2026-11-15 before the run, after the user's approval of (1) the task cost of revision 14, (2) the role-loss weighting, (3) a restart from initialization
(no gripper head reset, no warm start from runs trained under the cost or loss of revision 13), (4) the budget, criteria and stop rule below.
Standing instructions: one seed only; dissect each component when learning goes badly; design changes only with approval.

## Configuration
- Task cost of revision 14 (code/stageb/env.py, COST_V = 3): in phases 2 and 3 the rate is the progress cost of the object at its pose (the tip while held, zero force
  while not held) plus (1 - xi)(1 + |p - o|/0.1); without top-face force the press term is 1 + |o - p_c|/0.1, p_c the center of the top face. Phases 1 and 4 unchanged.
- Hand role loss of revision 14 (code/stageb/l2.py, replay_loss w_mode 1): the squared error at t is weighted by 1 - exp(-lam_s r(t)), r(t) the rest time of the
  memory since the last change of the attached load (r = infinity before the first change). The unweighted loss is logged as a diagnostic.
- Brain, critic, hand networks, exploration and PPO as at the end of attempt 1 (amendments 1-3): actor MLP 31-128-128-13 (absolute goal c + h tanh(u), stiffness
  k_lo (k_hi/k_lo)^sigmoid(u), state-dependent standard deviation bounded in [0.005, 0.3] (goal) and [0.05, 0.5] (stiffness), initial 0.04 and 0.3; gripper logit bias 6);
  exploration noise of the six continuous actions held for 10 decisions (the PPO log-probability is the per-decision Gaussian of the held sample: an approximation);
  critic MLP 37-128-128-1 with six privileged inputs; hand G_W MLP 6-64-64-21 of (sin q, cos q), a_bar = 50 tanh(raw).
  gamma 0.99, lambda 0.95, clip 0.2, value coefficient 0.5, entropy 1e-3, brain lr 3e-4, 5 epochs, minibatch 4096; hand lr 1e-3, 16 steps of 32 episodes per
  iteration before the brain step. Initialization l2.init_params(PRNGKey(0)); pool of 6144 episodes from seed 10000, 128 per iteration; 300 iterations.
- Command: python3 l2_train.py ../../records/l2_attempt2 300 128 6144 (defaults cost=3, role=1).

## Checks every iteration (C1)
Joint violations 0; non-finite states 0; tank-bound margin >= -0.01 J on episodes without saturation below T_min. A failure stops the run (implementation or theory error).

## Evaluation every 10 iterations (deterministic policy, 64 episodes from seed 6000)
Cost, grasp, hold error, force error, class decision among held episodes with contact in the force window, the same with the learned load hidden from the brain,
budget episodes, hand features (transfer residual, prior error).

## Dissection and visual record at iterations 50, 100, 150, 200, 250, 300
code/stageb/l2_dissect.py (reach, aim, gripper, stiffness, task, ledger use, press, face of the block contact, carry, hand error by phase, hand features,
exploring-policy statistics, critic explained variance, channel test with the L1 reference brain driven by the learned hand), and an animation of four evaluation
episodes (code/stageb/anim_rollout.py -> figures/l2_attempt2_rollout_itXXXX.gif) plus the learning curves (code/stageb/fig_l2_attempt2.py).

## Criteria at iteration 300
- (G) (C1) on every iterate.
- (H1) median transfer residual of the hand below 1.768 N m (class basis with the nominal prior).
- (H2) class decision of the channel test at least 0.95.
- Brain, descriptive: grasp, hold error, share of held phase-3 steps on the top face and on the side, contact in the force window, force error, class decision
  with and without the learned load, release.

## Stop rule
At iteration 300, if fewer than 25% of the evaluation episodes are held with top-face contact in the force window, the run stops and is dissected; further measures
(for example a curriculum from intermediate states) are proposed with their rationale and not added. Divergence (non-finite states) stops the run at once.

## Amendment 1 (after iteration 60; written before the restart)
Observed at iteration 50 (records/l2_attempt2a/dissect_it0050.json, figures/l2_attempt2a_rollout_it0050.gif): the brain's goals have barely left the zero-action
posture. At the end of phase 1 the goal tip is 0.106 m from the tip of the box-center posture and 0.219 m from the grasp point (median); the spread of the goal tips
across the 64 evaluation episodes is 0.110 m against 0.197 m for the grasp points; evaluation grasp 0.016, nearest distance median 0.127 m; evaluation grasp 0.02 at 60.
In attempt 1, trained from initialization with the initial goal standard deviation 0.1 (amendment 2), the same quantities were 0.197, 0.073 and 0.162 m at iteration 30
(grasp 0.125). The pre-registered initial standard deviation 0.04 was the value set at the warm start of attempt 1 at iteration 70 (amendment 3), after reaching had
been learned, not the value under which attempt 1 learned to reach from initialization. The hand meets (H1) and (H2) already (transfer residual 1.238 N m; class
decision of the channel test 0.984).
Change: initial goal standard deviation 0.1 (pre-squash; bounds unchanged); everything else unchanged with the same seed (identical initialization of all other weights,
identical episode and noise sequences). The run restarts from iteration 0 in records/l2_attempt2; the stopped run is kept in records/l2_attempt2a. Budget, criteria and
stop rule apply to the restarted run. Decisive check of the cause: at iteration 50 the goal tip to grasp point distance and the evaluation grasp of the restarted run
against records/l2_attempt2a (iteration 50: 0.219 m, 0.016).
Command: python3 l2_train.py ../../records/l2_attempt2 300 128 6144 sd_goal=0.1

## Amendment 2 (after iteration 50 of the restart; written before the next runs)
Decisive check of amendment 1: at iteration 50 the restarted run (initial goal standard deviation 0.1) has the goal tip 0.233 m from the grasp point at the end of phase 1
(0.219 m with 0.04; attempt 1: 0.079 m), goal spread 0.179 m (0.110 m with 0.04), evaluation grasp 0.02. The exploration magnitude is refuted as the cause; the restarted
run is stopped and kept in records/l2_attempt2b.
Remaining differences to the start of attempt 1: (i) the four relative-position inputs, scaled by 1/0.05 and clipped at +-5, enter with random weights from initialization
(in attempt 1 they were added at iteration 70 with zero weights); at initialization they give a mean |contribution| of 1.02 to the first-layer pre-activations against 0.41 from
the other 27 inputs together, and 14.0% of the first-layer units are saturated (|pre-activation| > 2) against 0.1% for attempt 1; (ii) the task cost of revision 14 (identical in
phase 1, larger costs while the object is not held); (iii) the role-loss weighting (hand only); (iv) the state-dependent standard deviation head (near-constant at initialization).
Minimal decisive experiments (pilots, 40 iterations each from initialization, same seed, initial goal standard deviation 0.1, role-loss weighting on):
- E1: cost of revision 14, first-layer weights of the four relative-position inputs set to zero in the actor and the critic (option rel_zero=1): isolates (i).
- E2: cost of revision 13 (option cost=1), weights as in the restarted run: isolates (ii).
Measure: goal tip to grasp point at the end of phase 1 and evaluation grasp at iterations 30 and 40, against attempt 1 (0.073 m and 0.125 at 30; grasp 0.172 at 40).
Expected reading: the factor whose removal restores attempt-1-like reaching is the cause; if both do, both contribute; if neither, (iv) or an interaction is examined next.

### Outcome of E1 and E2 (40 iterations each; records/l2_diag_relzero, records/l2_diag_cost13, records/l2_goal_check.json)
Goal tip to grasp point at the end of phase 1: E1 0.176 m (iteration 30) and 0.170 m (40); E2 0.172 m and 0.167 m; restarted run of amendment 1 0.206 m at 30;
attempt 1 0.073 m at 30 and 0.062 m at 40. Evaluation grasp 0.02 at 30 and 40 in both (attempt 1: 0.125, 0.172). Neither factor restores attempt-1-like reaching.

## Amendment 3 (written before the runs): decisive experiments on the initial network
Remaining differences: the initial weights (attempt 2 draws a 31-input, 13-output network; attempt 1 drew a 27-input, 7-output network with a global standard deviation,
so with the same seed the hidden features differ), the state-dependent standard deviation head, and the interaction of these with the cost and the role-loss weighting.
- E3: attempt 1's initial network (records/l2_attempt1/params_it0000.pkl, function-preserving upgrade: zero weights for the relative-position inputs and for the
  standard-deviation head, whose biases reproduce the global standard deviation 0.1), cost of revision 14, role-loss weighting on, 40 iterations.
- E4: the same initial network with the cost of revision 13 (reproduction check of attempt 1 with the current code), 40 iterations.
Same episode and noise sequences as attempt 1 (seed 0). Reading: E4 close to attempt 1 and E3 close to attempt 1 -> revision 14 is not the cause and attempt 2 is restarted
from attempt 1's initial network; E4 close and E3 not -> revision 14 slows the learning of reaching (to be dissected); E4 not close -> the current code does not reproduce
attempt 1 (to be dissected before anything else).

### Outcome of E3 and E4 (40 iterations each; records/l2_diag_init1_rev14, records/l2_diag_init1_rev13)
E4 reproduces attempt 1's evaluation at iteration 0 exactly (cost 32.822). Goal tip to grasp point at the end of phase 1: E4 (revision 13) 0.101 m at 30 and 0.089 m at 40
(evaluation grasp 0.09 at 40), E3 (revision 14) 0.195 m and 0.171 m (grasp 0.02); attempt 1 0.073 m and 0.062 m (grasp 0.172). Reading of amendment 3: E4 is close to
attempt 1 and E3 is not, so the cost of revision 14 slows the learning of reaching at equal initialization; under revision 13 the initial network of attempt 1 also learned
faster than that of attempt 2 (E4 0.089 m against E2 0.167 m at 40).
Dissection at iteration 30 (exploring rollouts, records/l2_adv_check.json): critic explained variance at phase-1 decisions 0.63 under revision 14 (E3) against 0.95 under
revision 13 (E4); the global-norm gradient clipping scales the actor's step by about 0.1 in both (not a difference). The same rollouts scored under both costs have
episode-cost standard deviations 11.15 (revision 14) and 9.83 (revision 13); 29% of the phase 2+3 cost variance under revision 14 is explained by the episode geometry and
the phase durations (8% under revision 13). Of the critic's residual at the first decision under revision 14 (standard deviation 4.47), the later phase durations explain 7%,
the grasp outcome 5%, geometry and grasp outcome together 11%. Reading: the returns of revision 14 are larger in magnitude (mean episode cost 50.4 against 27.7 on the same
rollouts) with modestly larger spread, and the critic fits them worse early in training; with Adam's scale-free steps a critic learning targets of larger magnitude
converges more slowly.

## Amendment 4 (written before the run): decisive experiment on the critic's return scale
- E5: as E3 (initial network of attempt 1, cost of revision 14, role-loss weighting), and the critic predicts the return divided by a scale set each iteration to an
  exponential average (weight 0.1) of the standard deviation of the GAE returns (option vnorm=1; the value used in GAE is the scale times the network output; the value loss
  is computed in scaled units; the scale is not trained). The policy objective is unchanged. 40 iterations.
Reading: E5 close to E4 (goal tip to grasp point about 0.09-0.10 m at 30-40) -> the critic's return scale is the mechanism and the normalization is adopted for attempt 2;
otherwise the mechanism remains open and is reported.

### Outcome of E5 (40 iterations; records/l2_diag_init1_rev14_vnorm)
Goal tip to grasp point at the end of phase 1: 0.167 m at 30 and 0.134 m at 40 (E3 0.195 and 0.171; E4 0.101 and 0.089); evaluation grasp 0.05 at 30, 0.02 at 40; critic
explained variance at phase-1 decisions 0.53 at 30 (E3 0.63). Reading of amendment 4: not close to E4, so the return-scale normalization is not adopted and the mechanism by
which the cost of revision 14 slows the learning of reaching remains open (single-seed comparisons).

## Amendment 5 (written before resuming)
Attempt 2 continues with the pre-registered configuration and amendment 1 (initial goal standard deviation 0.1): the run of records/l2_attempt2b is resumed from its
iteration-60 checkpoint with fresh optimizer states (the checkpoint holds the weights only) and a fresh episode-sampling state, up to iteration 300, in records/l2_attempt2.
Budget, criteria and stop rule unchanged; the dissections and animations at 100, 150, 200, 250 and 300 as planned.
Command: python3 l2_train.py ../../records/l2_attempt2 240 128 6144 ../../records/l2_attempt2b/params_it0060.pkl
Correction to amendment 5 (before any iteration ran; the launch failed on the missing file): the run of records/l2_attempt2b was stopped after its iteration-50 evaluation,
so its last checkpoint is iteration 50 (amendment 2 is right; "iteration 60" in amendment 1 refers to the first run, records/l2_attempt2a). Attempt 2 resumes from
records/l2_attempt2b/params_it0050.pkl up to iteration 300. Command: python3 l2_train.py ../../records/l2_attempt2 250 128 6144 ../../records/l2_attempt2b/params_it0050.pkl
Relaunch note: the corrected launch ran to iteration 55 and was then terminated together with a monitoring shell (a tooling error, no change of the run itself;
records of iterations 51-55 discarded). Attempt 2 is relaunched from records/l2_attempt2b/params_it0050.pkl with the same command, in a detached session.

## Amendment 6 (written before the evaluations of iterations 250 and 300): additional dissection at iteration 300
At iteration 200 the brain reaches (nearest distance median 0.050 m; 21.9% of the evaluation episodes within 2 cm) but has not learned to carry (6 evaluation episodes
held through phase 2; goal tip 0.388 m from p* at the end of phase 2) and has made no top-face contact; the hand meets (H1) and (H2) (transfer residual 1.046 N m,
channel class decision 0.953).
Question: are phases 2 and 3 not learned because their learning signal is right but rare (held states occur only after a grasp in phase 1), or because the signal is wrong?
Measure (code/stageb/l2_signal_check.py; exploring rollouts of the checkpoint on the first 256 episodes of the pool, key 4242, cost of revision 14). Three classes of decision:
phase-1 decisions without the object (target: the grasp point), held phase-2 decisions (target p*) and held phase-3 decisions (target p_c). For each class:
- the number of decisions per 128 episodes;
- the correlation between the GAE advantage (lambda 0.95 as in training, and lambda 1) and the projection of the goal-tip deviation (goal tip of the sampled action minus
  that of the mean action, by the forward kinematics) on the unit vector from the mean goal tip to the target;
- 95% bootstrap intervals over episodes (1000 resamples).
Computed at iterations 200, 250 and 300; the reading uses 300.
Reading:
- If the lambda-0.95 intervals for phases 2 and 3 lie above zero, the signal is right but rare. The measure proposed is then a start-state distribution that covers held states.
- If the interval for a phase contains zero or lies below it, the signal of that phase is wrong, and a start-state distribution alone would not help. The critic and the
  cost of that phase are dissected before any measure is proposed.
- If the intervals lie above zero with lambda 1 but not with lambda 0.95, the critic is the defect.
The stop rule and the criteria at iteration 300 apply as registered.
Correction to amendment 6 (written after the computation at iteration 200 and before those at 250 and 300).
- Iteration-200 result, phase 1: correlation (lambda 0.95) 0.055 [0.012, 0.104], 4999 decisions per 128 episodes.
- Phase 2: 0.216 [-0.005, 0.354], 375.5 decisions per 128 episodes; the held decisions came from 16 of the 256 episodes.
- Phase 3: 0.092 [-0.132, 0.257], 385.0 decisions per 128 episodes, from 17 of the 256 episodes.
- With lambda 1 the correlations were -0.001, 0.016 and 0.025, and all three intervals contained zero.
With so few held episodes, an interval containing zero cannot tell a wrong signal from a rare one, so the reading as written would call a rare signal wrong.
Changes:
(1) The measure uses the first 2048 episodes of the pool (key 4242, in chunks of 256). This gives about eight times as many held episodes.
(2) Reading: an interval below zero means the signal of that phase is wrong. An interval containing zero is unresolved: it is reported as such, and no measure is proposed
on that evidence. The other readings are unchanged.
Iteration 200 is recomputed with the same measure.

## Amendment 7 (written after the evaluation of iteration 250, before that of 300): hand diagnostic at iteration 300
Observed in the evaluations every 10 iterations:
- The median transfer residual of the hand fluctuates between 0.91 and 2.06 N m from iteration 60 to 250, without a trend.
- It is above 1.768 at iterations 180 (2.059) and 240 (1.836).
- The weighted role loss in training rose from 0.72 (mean over iterations 126-150) to 0.92 (226-250) while the held share grew.
(H1) at iteration 300 is a single evaluation of this fluctuating quantity; it is applied as registered.
Question: does the fluctuation come from the optimizer's noise around better features, or from the loaded target posture (at p*) lying outside the hand's training data?
In the second case the residual there is an extrapolation of the features.
Measures (code/stageb/l2_hand_check.py and the signal check of amendment 6):
- (a) The median transfer residual on the 64 evaluation episodes, for the hand of each checkpoint 210, 220, ..., 300, and for the hand whose weights are the average of
  those ten.
- (b) In the exploring rollouts of the signal check at iteration 300 (2048 pool episodes): held decisions per 128 episodes with the measured tip within 0.05 m of the grasp
  point at the start of the episode, and within 0.05 m of p*.
Reading:
- (a) If the averaged hand is below the 25th percentile of the ten checkpoint values, the fluctuation is optimizer noise, and averaging or a smaller step would stabilize
  the features. Otherwise it is not.
- (b) If there is less than 1 held decision per 128 episodes within 0.05 m of p*, the residual at the target posture is an extrapolation outside the training data. A
  start-state distribution with held states near p* would supply that data.

## Outcome at iteration 300 (records/l2_attempt2/dissect_it0300.json, records/l2_signal_check.json, records/l2_hand_check.json)
Registered criteria and stop rule:
- (G) met. (C1) holds on all 300 iterations of the lineage (records/l2_attempt2b for iterations 1-50, records/l2_attempt2 for 51-300): joint violations 0, non-finite
  states 0, minimum tank-bound margin 1.5e-7 J.
- (H1) not met. The median transfer residual is 2.947 N m at iteration 300 (criterion: below 1.768). 6 of the 25 evaluations from iteration 60 to 300 were above 1.768
  (range 0.910-2.947), including 4 of the last 5.
- (H2) met. Class decision of the channel test 0.969.
- Stop rule applied. 0% of the evaluation episodes were held with top-face contact in the force window (threshold 25%), so the run stops and is dissected.
Brain, descriptive (evaluation unless noted):
- Grasp 0.109. Nearest-distance median 0.033 m; 29.7% of the episodes within 2 cm.
- 7 episodes held through phase 2; at the end of phase 2 the goal tip is 0.263 m from p*.
- 10 episodes held in phase 3; the tip is 0.106 m below the top face and 0.186 m lateral to p_c. No top-face contact.
- Training rollouts: open probability while held 0.0007 (phase 2) and 0.0012 (phase 3); releases before phase 4 in 3.1% of the episodes.
- Critic explained variance by phase: 0.936, 0.916, 0.948, 0.649.
Amendment 6 (corrected reading):
- Phase 3 right: 0.061 [0.006, 0.118].
- Phase 2 unresolved: 0.013 [-0.033, 0.071].
- The phase-1 reference also fell, to 0.007 [-0.006, 0.022]; all lambda-1 intervals contain zero.
- The outcome is not "right and rare" for both phases, so no measure is proposed on this evidence.
Amendment 7:
- (a) The averaged hand gives 1.874 N m, above the 25th percentile of the ten checkpoints (1.507). Not optimizer noise.
- (b) 0.06 held decisions per 128 episodes within 0.05 m of p* (114.5 near the start grasp point). The residual at the target posture is an extrapolation outside the
  training data; a start-state distribution with held states near p* would supply that data.
Descriptive, examined after iteration 250 (not registered):
- In held decisions, the sd of the goal-tip deviation along the target direction fell from 0.029 and 0.025 m (iteration 200, phases 2 and 3) to 0.011 and 0.010 m (300).
- Over the same interval the median action standard deviation fell from 0.055 and 0.058 to 0.038 and 0.036, and the median tanh slope from 0.963 and 0.975 to 0.780 and
  0.879.
- Meanwhile the mean goal tip approached the targets: from 0.353 to 0.266 m from p*, and from 0.267 to 0.216 m from p_c.
Proposal (not added; awaiting approval):
- Measure: a training start-state distribution mu = 0.5 mu_0 + 0.25 mu_2 + 0.25 mu_3, where mu_2 is the phase-2 start at the grasp posture q_o with the object attached, and
  mu_3 is the phase-3 start at the posture q_s of p* with the object attached. Velocity zero, memory zero, tank energy E0, the episode's remaining phase durations. Cost,
  networks, guarantee layers and evaluation unchanged.
- First step: a decisive pilot from this run's iteration-300 checkpoint, two arms of 50 iterations with the same seed: unchanged, and with mu. Criteria are to be registered
  before the pilot.
