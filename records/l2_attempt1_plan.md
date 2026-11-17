# L2 attempt 1: coupled learning, one seed (Stage B, spec revision 13)

Written 2026-11-14 before the run. Instructions of the user: train one seed only (no seed repetitions without an explicit instruction); go straight to L2;
small pilots allowed; if learning goes badly, dissect whether each component learned or acted as intended.

## Configuration (Definition coupled learning)
- c_r = 0: the hand learns only its role loss; Gamma = I; damping D_W fixed; memory dimension n_a = 7; tank, gate, guard, constraint layer, saturation fixed (L0 constants).
- Brain: MLP 27-128-128-7 (tanh) of o_k = (q, dq, tau_ext, context, ledger) without q_d and K. Gaussian on six pre-squash actions: goal offset 0.5 tanh(u) rad from the measured posture,
  clipped to the joint box; stiffness k_lo (k_hi/k_lo)^sigmoid(u). Bernoulli gripper. Critic MLP 33-128-128-1 with six privileged inputs (time left in the phase, payload,
  active force target, block depth, class, budget flag; training only). PPO: gamma 0.99 per decision, lambda 0.95, clip 0.2, 5 epochs, minibatch 4096, lr 3e-4, entropy 1e-3.
- Hand: G_W = MLP 6-64-64-21 of (sin q, cos q), entries tanh-bounded in [-1, 1]; a_bar bounded by 50 tanh(./50); u_ad = G_W^T (a_bar + z) (no nominal model).
  Role loss: mean |u_ad - u*|^2 replayed on the recorded rollouts with the plant trajectory held fixed (the rest-gated normalized memory law is replayed exactly).
  Adam lr 1e-3, 4 steps on 32 episodes per iteration, before the brain step (the hand is the faster time scale).
- Episodes: a pool of 6144 task-version-2 episodes from seed 10000 (30% hard; 30% budget, E0 = Delta T + rho E_task with E_task from the L1 reference brain and hand);
  128 episodes per iteration; 300 iterations planned; stopped early on divergence.

## Checks every iteration (C1)
Joint violations 0; non-finite states 0; tank-bound margin >= -0.01 J on episodes without saturation below T_min.

## Component diagnostics every 10 iterations (deterministic policy; 64 evaluation episodes from seed 6000)
- Task: mean cost, grasp, hold error, force error, class accuracy, release, against the L1 reference (cost 2.39, grasp 1.000, hold 0.0014 m, force error 0.032, class 0.988).
- Brain's use of the ledger: the same episodes with the learned load hidden from the brain (the hand unchanged). Intended role: class accuracy falls toward chance
  while the hold error does not change (the brain should not move its hold goal with the learned load).
- Hand features: residual at the target posture after rest calibration at the grasp posture with the object (minimum-norm calibration), against the fixed class basis
  with the nominal prior of L1; error of the learned prior at the start posture against zero prior and the nominal prior.
- Budget episodes (all 64 as budget episodes): cost and gate activity.
Attempt 1 is descriptive. The brain is judged to have learned its role if the task metrics approach the L1 reference and the blind test shows ledger use for the class only;
the hand, if the transfer residual falls below its initial value toward the class basis and the prior error below the zero-prior value. Failures are dissected per component.

## Amendment 1 (after 15 iterations; written before the restart)
Observed: with the goal parametrized relative to the measured posture, the zero action lets the arm follow its own sag (no spring force), so the arm rarely rests,
the rest-gated memory rarely calibrates (hold-window mismatch about 9-10 N m) and grasps do not occur (grasp rate 0.00-0.02); exploration noise of 0.15 in the
pre-squash goal (about 0.075 rad per decision) also keeps joint speeds above the grasp speed of 0.1 rad/s.
Changes: the goal is the absolute posture c + h tanh(u) (the brain's action space C_q itself), so the zero action holds the arm at the box center;
initial standard deviation 0.02 for the goal and 0.3 for the stiffness pre-squash actions; gripper logit bias 6 (closed with probability 0.9975 per decision).
The run restarts from iteration 0 with the same seed; everything else unchanged.

## Amendment 2 (after 12 more iterations; written before the restart)
Observed: the memory now calibrates at rest (hold-window mismatch 0.37-0.83 N m), but the learned prior did not move (prior error 16.08 against 16.22 N m with zero prior):
with a_bar = 50 tanh(raw/50), an Adam step of 1e-3 changes a_bar by about 1e-3, far too little for targets of order 10. The brain did not improve (cost 31-37, grasp 0.01-0.04):
with goal noise of 0.02 the exploration is too small to learn reaching, and per-decision noise keeps joint speeds near the grasp limit.
Changes: a_bar = 50 tanh(raw) (raw of order one); 16 hand steps per iteration (the hand stays the faster time scale); exploration noise of the six continuous
pre-squash actions held for 10 decisions (0.5 s, resampled at decisions 0, 10, 20, ...; the PPO log-probability is the per-decision Gaussian of the held sample);
initial standard deviation 0.1 for the goal and 0.3 for the stiffness. Restart from iteration 0, same seed.

## Amendment 3 (after iteration 74; written before continuing)
Observed: in the training rollouts the grasp rate stayed at 0.003-0.023 for 74 iterations, while the deterministic evaluation reached 0.34 at iteration 60.
The held exploration noise of the goal (standard deviation 0.07-0.10, about 5-7 cm at the tip) rarely leaves the tip within 2 cm of the grasp point at rest,
so the later phases (carry, press, class decision, release) receive almost no training data. Dissection at iteration 50: the brain aims 5.5 cm from the grasp
point and the hand tracks the goal within 1 cm; the hand learned its role (prior error 0.96 N m against 16.2 N m with zero prior; residual after calibration
at another posture 1.55 N m against 1.77 N m for the class basis with the nominal prior).
Changes: state-dependent standard deviation output by the actor (bounded in [0.005, 0.3] for the goal and [0.05, 0.5] for the stiffness pre-squash actions);
four relative-position inputs, o - p_hat and p* - p_hat scaled by 1/0.05 and clipped at +-5 (functions of existing entries, so the information structure is unchanged);
warm start from the iteration-70 checkpoint through a function-preserving upgrade (zero weights for the new inputs and outputs; checked to reproduce the actor
and the critic exactly), with the goal standard deviation reset to 0.04. Iterations continue from 71 in records/l2_attempt1b.

## Diagnosis at iteration 100 (run stopped; written before the pilot)
Evaluation at iteration 100: cost 15.158, grasp 0.16, hold error 0.2263 m, force error 1.000 (no press). Component dissection (records/l2_attempt1b/dissect_it0080.json,
code/stageb/l2_dissect.py) and the mechanism check below.
- Hand, channel test: the L1 reference brain driven with the learned hand of iteration 80 on the 64 evaluation episodes gives cost 2.558, grasp 1.000, hold error 0.00157 m,
  force error 0.0313, class decision 0.984 (64 episodes with contact), 0 violations; with the L1 hand: 2.570, 1.000, 0.00161 m, 0.0311, 1.000. The learned hand serves both
  as compensator and as the source of the class information in the ledger.
- Critic: explained variance 0.965, 0.956, 0.954, 0.972 by phase (exploring rollouts).
- Brain: reaching partly learned (deterministic: nearest distance median 0.017 m, 58% within 2 cm, goal tip 0.038 m from the grasp point). In the exploring rollouts grasps
  occur (1.05 per episode) but the object is released before phase 4 in 40% of the episodes; open probability per decision while attached 0.337, 0.565, 0.688, 0.646 by phase.
- Mechanism: the task cost of spec rev. 13 makes dropping cheaper than holding. At the grasp point the held rate of phase 2 is |p - p*|/0.1 >= 2 (|p_o,x - p*_x| >= 0.2 m),
  the dropped rate is 1 + |p - o|/0.1 = 1; in phase 3 the held rate before contact is 1 + (p_y - y_e)/0.1 + |p_x - p*_x|/0.1 >= 1. In the exploring rollouts of iterations
  70, 80, 90 the held rate exceeded the dropped rate at every attached decision of phases 2 and 3 (median 2.26-2.52 in phase 2, 3.98-4.18 in phase 3). The critic learned it
  first: V(xi=0) - V(xi=1) at attached decisions (phase 2 / phase 3) -0.091 / -0.078 at iteration 0, +1.698 / +2.534 at 10 (positive at every decision), +2.195 / +2.397 at 20,
  +0.390 / +0.589 at 40; the actor followed: open probability while attached 0.002 / 0.002 at 0, 0.004 / 0.003 at 10, 0.009 / 0.008 at 20, 0.162 / 0.149 at 40,
  0.500 / 0.591 at 70, 0.567 / 0.663 at 80, 0.577 / 0.606 at 90. Carry, press, class decision and release therefore receive almost no data.
- Metric correction: the logged class accuracy counted grasped episodes without contact (then the lighter target is "closer"); from now on it is conditioned on contact.
The run is stopped at iteration 100 (of 300): the brain sits in a local optimum of the task cost.

## Pilot P1: proposed cost (not adopted without the user's approval; written before the run)
Cost version 2 (code/stageb/env.py, COST_V): while the object is not held, phases 2 and 3 add the held cost evaluated at the object's own pose with zero force,
|o - p*|/0.1 in phase 2 and 1 + max(0, o_y - y_e)/0.1 + |o_x - p*_x|/0.1 in phase 3; so at a release the rate rises by exactly 1. Unchanged for the reference brain
(checked: mean cost by phase identical on 16 evaluation episodes). Warm start from records/l2_attempt1b/params_it0100.pkl (brain and hand as they are, fresh optimizer states),
30 iterations, 128 episodes per iteration, everything else unchanged; output records/l2_pilot_cost2.
Expectations: (E1) open probability while attached in phases 2-3 of the training rollouts below 0.05 by iteration 115; (E2) fraction of training episodes attached
in phase 3 above 0.3 by iteration 130; (E3) 0 violations on every iterate. Otherwise descriptive.

### Pilot P1 outcome (iterations 101-130; records/l2_pilot_cost2)
Open probability while attached (phases 2-3, training rollouts) 0.662 at iteration 101, 0.611 at 105, 0.404 at 115, 0.341 at 120, 0.163 at 130: the trend reverses,
but (E1) fails (0.404 > 0.05 at iteration 115). Fraction of training episodes attached in phase 3: 0.12-0.30, 0.20 at 130; (E2) fails. Contact while attached in phase 3
rises from 0.00 (101) to 0.56 (130). 0 violations on every iterate; (E3) holds. Evaluation (deterministic) at 130: cost 28.710 (version 2), grasp 0.34, hold error 0.2859 m,
force error 1.000. Reading: under cost version 2 a release is followed by an immediate re-grasp (the tip stays at the object and the gripper closes), so the penalty of a
release is about 1 per second times the short time until the re-grasp; the learned release unlearns slowly, and with an open probability of 0.2-0.6 per decision the
object cannot be carried for the 20 or more decisions a carry takes.

## Pilot P2: proposed cost with the gripper head reset (not adopted without the user's approval; written before the run)
As pilot P1 (cost version 2, warm start from records/l2_attempt1b/params_it0100.pkl, 30 iterations), and in addition the gripper head of the actor is set back to its
initial value (output weights 0, bias 6: closed with probability 0.9975 per decision in every state), since its learned weights encode the release incentive of the old cost.
Goal, stiffness and standard-deviation heads, the hidden layers, the critic and the hand are unchanged. Output records/l2_pilot_cost2_grip.
Expectations: (E1') open probability while attached in phases 2-3 of the training rollouts below 0.05 at every iteration; (E2') fraction of training episodes attached in phase 3
above 0.3 by iteration 115; (E3) 0 violations on every iterate. Otherwise descriptive, compared with P1 (same start, same seed; the only difference is the gripper head).

### Pilot P2 outcome (iterations 101-130; records/l2_pilot_cost2_grip)
(E1') holds: open probability while attached 0.002-0.004 at every iteration. (E2') holds: fraction of training episodes attached in phase 3 0.33-0.53 (0.52 at 101).
(E3) holds: 0 violations. Training grasp at the end of phase 1 0.19-0.38 (P1: 0.01-0.12); evaluation grasp 0.42 at 100 (reset only), 0.34, 0.36, 0.38.
Two new failures, dissected at iteration 130 (records/l2_pilot_cost2_grip/dissect_it0130.json and the checks below):
- Side-wall exploit of the press term. Among the 32 evaluation episodes held in phase 3, the tip touches the block in about 90% of the phase-3 decisions, but 91.5% of the
  held phase-3 steps are in side-face contact (median 0.205 m below the top face, 4.9 mm side penetration) and 0% in top-face contact; no evaluation episode has top-face
  force in the force window. Cause: without top-face force, Omega_press = 1 + max(0, p_y - y_e)/0.1 is minimal anywhere below the top face, also beside the block, so the
  brain descends where it is (hold error 0.258 m) and presses on the side wall, where the nearest-face contact law gives a horizontal force only.
- Hand degradation. Transfer residual 4.413 N m (1.417 at 100), prior error 1.061 N m (0.723 at 100); channel test with the reference brain: class decision 0.719,
  force error 0.0476, cost 2.954 (it 80 hand: 0.984, 0.0313, 2.558). Cause: in the exploring rollouts the segments after a load change (grasp or release) before the memory
  has rested 0.1 s take 14.2% of the time but 74.2% of the role loss (mean squared error 24.1 against 1.39 elsewhere; at attempt-1b iteration 80: 8.5%, 77.7%, 8.32 against 0.221).
  The rest-gated memory cannot know a new load before it rests, so these errors are not the hand's to fix; minimizing them pulls the prior and features toward the
  average payload of the brain's current behavior (calibrated error 0.221 -> 1.386).

## Pilot P3: cost version 3, gripper head reset, role-loss weighting (not adopted without the user's approval; written before the run)
Cost version 3: as version 2, and without top-face force the press term is the distance to the center of the top face, Omega_press = 1 + |p - p_c|/0.1 with
p_c = (p*_x, y_e) (also inside the dropped term of phase 3). Checked: reference brain cost by phase on 16 evaluation episodes 0.8165, 0.0871, 1.5147, 0.2569 (version 1:
phase 3 1.5116). Role-loss weighting: the error at t is weighted by 1 - exp(-lam_s r(t)), r(t) the rest time of the memory since the last change of the attached load,
r = infinity before the first change (checked: unweighted mode reproduces the previous loss exactly). Gripper head reset as in P2. Warm start from
records/l2_attempt1b/params_it0100.pkl, 30 iterations, otherwise unchanged; output records/l2_pilot_v3.
Expectations: (E1'') open probability while attached below 0.05 at every iteration; (E2'') at iteration 130, below 0.2 of the held phase-3 steps of the evaluation in
side-face contact (P2: 0.915); (E3'') at iteration 130, transfer residual at most 2.0 N m (P2: 4.413) and class decision of the channel test at least 0.9 (P2: 0.719);
(E4) 0 violations on every iterate. Otherwise descriptive, compared with P2.

### Pilot P3 outcome (iterations 101-130; records/l2_pilot_v3, dissect_it0130.json)
(E1'') holds: open probability while attached 0.002-0.007. (E2'') fails: 0.834 of the held phase-3 steps of the evaluation in side-face contact (P2: 0.915), 0.006 on the
top face. (E3'') fails on one of its two parts: transfer residual 1.163 N m (P2: 4.413; holds), class decision of the channel test 0.891 (P2: 0.719; required 0.9).
(E4) holds: 0 violations. Evaluation at 130: grasp 0.44, nearest distance in phase 1 median 0.014 m, hold error 0.2766 m, force error 1.000, prior error 0.733 N m.
Carry check (deterministic, episodes held through phase 2): at 130 the goal tip is 0.387 m from p* in the last 40% of phase 2 and the tip 0.328 m (0.395 m at the start of
phase 2), so the brain has not yet learned to carry the object toward p*; phase 3 therefore starts far from the block center and the descent ends on the side wall.
Same for P2 at 130 (0.397, 0.332, 0.401 m).

## Self-review before the approval request (2026-11-15; no run)
Cost audit (code/stageb/cost_audit.py; 200000 random postures, plant of the first evaluation episode). (i) Dropping minus holding at the same pose without top-face force:
version 1 minimum -8.174 (phase 2) and -14.049 (phase 3), negative in 0.982 and 1.000 of the states; versions 2 and 3: +1.000 everywhere. (ii) Held phase-3 rate along the
side wall of the block (1035 states): version 1 and 2 slope with depth +0.01 per m (range 2.500-2.600, flat), version 3 +6.62 per m (range 4.015-5.978).
Judgement: the two cost changes are one principle (the rate measures the object's progress; holding is the means of progress) checked by two properties stated
independently of the learner; the role-loss weighting follows the rest-gated memory law; the gripper head reset and the warm starts from runs trained under the flawed
cost and loss are withdrawn as methods (kept as diagnostic evidence).
