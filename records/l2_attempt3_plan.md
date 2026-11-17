# L2 attempt 3: the adopted configuration to iteration 300 (pre-registration)

Written before the run. The user approved on 2026-11-15: continue arm T of pilot T to iteration 300 as attempt 3 (records/l2_pilotT_plan.md, outcome).

## Configuration
- Cost revision 15; task-space action (revision 16, l2.goal_of in task mode); role weighting with the corrected start of r(t); restart share 0.5 (env.restart);
  128 episodes per iteration from the same pool. This is the configuration of arm T.
- Iterations 1-100 are arm T (records/l2_pilotT_T). Iterations 101-300 continue from records/l2_pilotT_T/params_it0100.pkl (records/l2_attempt3).
- The optimizer states start fresh at 101; the pilot-T checkpoints do not contain them. From this run on, checkpoints also store the optimizer states.
- Command: python3 l2_train.py ../../records/l2_attempt3 200 128 6144 ../../records/l2_pilotT_T/params_it0100.pkl restart=0.5 eval_restart=1
  (action mode taken from the checkpoint; cost 4 by default).
- Evaluations every 10 iterations, from the original start and from both restart starts.
- Dissections, animations and figures at 150, 200, 250 and 300, reported to the user.

## Criteria at iteration 300
- (G) (C1) on every iterate, 1-300: joint violations 0, non-finite states 0, tank-bound margin at least -0.01 J on episodes without saturation below T_min.
- (H1) Hand, changed from attempt 2: the mean over the evaluations 260-300 (five) of the median transfer residual is below 1.768 N m (class basis with the
  nominal prior). Attempt 2 showed that a single evaluation of this quantity fluctuates by a factor of two.
- (H2) Hand: the class decision of the channel test (L1 reference brain with the learned hand), in the dissection at 300, is at least 0.95.
- Brain, descriptive:
  - original start: grasp, nearest distance, the share within 2 cm, and the share slow enough to grasp at the nearest point;
  - phase-2 start: hold error at p*;
  - phase-3 start: accurate presses (relative force error below 0.5, held to phase 4), contact share and force ratio.
- Learning signal (l2_signal_check.py, 512 episodes, cost 4) at 200 and 300, from both restart starts.
- Stop rule (as in attempt 2): if fewer than 25% of the evaluation episodes from the original start are held with top-face force in the force window at 300, the
  run stops and is dissected. Further measures are then proposed with their rationale and not added.

## Outcome at iteration 300 (records/l2_attempt3; dissect_it0300.json; records/l2_restart_behavior.json, l2_signal_check.json)
Registered criteria and stop rule:
- (G) met. (C1) holds on all 300 iterations (records/l2_pilotT_T for 1-100, records/l2_attempt3 for 101-300): joint violations 0, non-finite states 0, minimum
  tank-bound margin 2.6e-4 J, no episode saturated below T_min.
- (H1) not met. Median transfer residual at the evaluations 260-300: 3.130, 3.408, 2.408, 2.609, 4.427 N m; mean 3.196 (criterion: below 1.768). The mean over
  the evaluations 110-150 was 1.400.
- (H2) not met. Class decision of the channel test 0.922 (59 of 64; criterion at least 0.95). It was 0.969, 0.953, 0.938, 0.922 at 100, 150, 200, 250.
- Stop rule not applied. 68.8% (44 of 64) of the evaluation episodes from the original start were held with top-face force in the force window (threshold 25%);
  0, 12.5, 21.9, 45.3% at 100, 150, 200, 250.
Brain, descriptive (deterministic, the 64 evaluation episodes):
- Original start (dissection at 300): grasp at the end of phase 1 0.172; nearest-distance median 0.0291 m; share within 2 cm 0.375; share slow enough to grasp
  at the nearest point 0.062; 45 episodes held in phase 3 (grasps after phase 1 included); class decision of the grasped episodes with top-face force 0.455
  (5 of 11).
- Phase-2 start: all 64 attached at the start of phase 3; median hold error at p* 0.0278 m (0.0642, 0.0458, 0.0300, 0.0256 at 100, 150, 200, 250).
- Phase-3 start: accurate presses 0.641 (0.078, 0.141, 0.266, 0.328 at 100, 150, 200, 250); held with top-face force in the force window 0.984; contact in 63
  of 64 episodes, median force ratio in contact 0.74 (90th percentile 1.51).
- Learning signal (512 episodes, cost 4): phase 2 toward p* 0.024 [-0.024, 0.078] at 200 and 0.075 [-0.006, 0.187] at 300; phase 3 toward p_c -0.010
  [-0.071, 0.046] at 200 and 0.022 [-0.012, 0.057] at 300. All four intervals contain zero (unresolved). Sd of the deviation along the target 0.0052-0.0070 m.
Hand, descriptive (dissections): median error of the hand while moving in phase 2, 0.571, 0.819, 0.901, 1.536, 1.560 N m at 100, 150, 200, 250, 300.
Post-hoc analyses (not registered; descriptive):
- Class decision at 300 among episodes with top-face force: 0.452 (62, phase-2 start), 0.460 (63, phase-3 start), 0.455 (11, original start). The ledger read
  with the L1 class rule at the last held decision of phase 3 was correct in 0.911 of 45 held evaluation episodes (records/l2_ledger_check.json).
- Hand replay (records/l2_hand_replay.json): on rollouts of the brain at 200, the role loss of the hands 100, 140, 200 was 1.153, 1.146, 1.188 (original start)
  and 0.747, 0.816, 0.647 (phase-3 start), while their transfer residuals were 1.366, 1.357, 2.241 N m.
- Offline hand training with fresh data (records/l2_hand_fresh_b140.json, l2_hand_fresh_b300.json): the hand of 140 trained 30 more iterations reached a mean
  transfer residual (every 5 iterations) of 1.934 on data of the brain at 140, 3.083 on data of the brain at 300, and 3.019 there with contact steps masked out
  of the role loss. Contact is not the cause; the drift occurs on both data sets and is larger on the later one. The reused-batch design
  (records/l2_hand_contact.json) agreed.
- Transfer exposure (records/l2_transfer_exposure.json): share of the role weight on held, frozen-memory steps more than 0.2 rad from the last rest posture
  0.018, 0.028, 0.077, 0.025 for the brains at 100, 140, 200, 300.
