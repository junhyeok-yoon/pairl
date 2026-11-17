# L2 attempt 4: integrated configuration I against the control N (pre-registration)

Written 2026-11-17 before any training of attempt 4, under the user's approval of 2026-11-17 (Parts B and C of that request). Platform: this PC's GPU with
XLA_FLAGS=--xla_gpu_deterministic_ops=true for training and decision evaluations, this PC's CPU for phase success, the hand pairing and contact peaks;
script/safe_run.sh. One seed (SEED 0 in both arms).

## Protocol change and status of the earlier pilots
The user discontinued the single-factor 60-iteration pilots: in pilot G the paired differences of phases that an arm did not target were about 0.1
(P2, which neither arm targeted: S - C +0.109 [+0.070, +0.152], B - C +0.090 [+0.047, +0.137]; records/l2_pilotG_decision.json, key phase), as large as
the effects being tested, while the baseline was still improving. The registered outcomes of pilots L (records/l2_pilotL_plan.md, outcome hash 485e1d7b) and G (records/l2_pilotG_plan.md, outcome
hash fa18f6e5) stand unchanged. Attempt 4 is a newly registered test, not a reinterpretation of either.

## Facts from existing records that motivate the items (Part B0, CPU)
- Rule (b) of pilot L compared hand_loss_all, the unweighted replayed role loss (script/l2_train.py, hand_loss_all; eval/diagnostics/l2_pilotL_decide.py),
  mean over iterations 251-260: C 9.125, L1 11.344, L05 10.993, L025 10.742 (ratios 1.243, 1.205, 1.177). The weighted training loss "hand" (hist
  hand_loss, the quantity printed first in the training log) over the same iterations: C 1.196, L1 1.423, L05 1.336, L025 1.282 (ratios 1.190, 1.118,
  1.072). Under the weighted quantity L025 would have been within 1.10; this does not change the registered outcome.
- l2_pilotG_C at 320 (eval/diagnostics/l2_gripper_hold.py, records/l2_gripper_hold_pilotG_C_320.json; 256 episodes per start, key 555, deterministic
  trajectories): open probability while holding, mean per decision, from the original start: phase 1 0.0008, phase 2 0.0000, phase 3 0.0000, phase 4
  0.5138 (share of holding decisions above 0.5: 0.3581); from the phase-4 start 0.3490 (0.0777). From the phase-4 start, deterministic: released by the
  end of phase 4 0.676, tip within 2 cm of p* at the last decision 0.277, released and within 0.254, released away from p* 0.422, still holding 0.324;
  gripper-sampled: released 0.887, released and within 0.359, released away 0.527. P4 0.152 deterministic, 0.160 gripper-sampled. The gripper does open in
  phase 4 most of the time; the larger loss is releasing away from p* (position at release) and the indecisive open probability (about 0.35-0.55).

## Roles served
- (1) L_tr: the hand's role of carrying a calibration from one posture to another (transfer residual; pilot L: 6.081 without it at 260).
- (2) Ledger tip-load observation: the hand-to-brain channel (handoff section 3, "손->뇌로는 에너지 장부"; pilot G arm B passed the class decision and the
  class gap but not P3).
- (3) Class-reading auxiliary loss: the brain's reading of the object class from what it observes (class decision about 0.5-0.6 at 320).
- (4) Entropy decay of the continuous actions: the brain's precision late in training (exploration noise 1.6 cm against a commanded 2.5 cm near the object,
  records/l2_sigma_near.json).
- (5) Gripper exploration: the brain's release skill (P4 gripper-sampled below deterministic for pilot L C, 0.176 against 0.266).

## Arms (both from l2_pilotG_C at iteration 320, iterations 321-920, optimizer states carried over)
- N (control): NH only, exactly the configuration of pilot G arm C: N_EP 1024, restart=0.5, eval_restart=1, mb_scale=1, c_ent=1e-2, revision-15 cost.
- I (integrated): N plus
  (1) ltr=0.25 (l2.transfer_loss, as arm L025: 64 pairs per episode, keys PRNGKey(9000 + hand step));
  (2) obs_tip=1 (as pilot G arm B: two inputs l1.tip_force(q, u_load)/10 appended to the actor and critic inputs, zero rows in the first layers and in their
      Adam moments; l2_train asserts the outputs do not change);
  (3) cls_aux=0.1: a zero-initialized linear head (128 -> 1) on the actor's last hidden layer (l2.add_cls_head, l2.actor_hidden), binary cross-entropy
      against the privileged heavy label (critic input entry task['heavy']) on decisions with the object held (observation entry xi), mean over held
      decisions of the minibatch, weight 0.1, added to the PPO loss (l2.ppo_loss_i), so its gradient reaches the actor trunk; the head is not an input of the
      policy (l2.policy does not read it) and plays no part in any rollout or evaluation; new parameters start with zero Adam moments;
  (4) c_ent_cont_end=1e-3: entropy coefficient of the continuous actions c(it) = 1e-2 (1e-3 / 1e-2)^((it - 321)/599), 1e-2 at 321 and 1e-3 at 920; the
      gripper entropy term keeps the coefficient of N, 1e-2 (the request names only the continuous actions);
  (5) grip_eps=0.05: in training rollouts (l2.rollout mode 0) the gripper is sampled from p' = (1 - eps) p + eps/2, p = sigmoid(logit), and the recorded
      log-probability and the PPO log-probability and gripper entropy use p' (l2.GRIP_EPS, l2.grip_prob, l2.logp_mix); evaluation modes 1 and 2 are unchanged.
      Known consequence: while holding, the training gripper opens with probability at least eps/2 = 0.025 per decision whatever the policy, so training
      episodes contain drops (I smoke at 321: held at the start of phase 3 from the phase-2 restarts 0.75 against 1.00 for N). Reported, not a rule.
  Not included: S, S2, K, D.
- Run names l2_a4_N and l2_a4_I, one group (group=new by N, I joins). Commands (both at the same time, two GPU processes):
  sh script/safe_run.sh mem=12G python script/l2_train.py group=<new|G> l2_a4_<arm> 600 1024 6144 data/runs/20261116165912/l2_pilotG_C/params_it0320.pkl
  restart=0.5 eval_restart=1 mb_scale=1 c_ent=1e-2 plan=records/l2_attempt4_plan.md "purpose=..." [ltr=0.25 obs_tip=1 cls_aux=0.1 c_ent_cont_end=1e-3 grip_eps=0.05]

## Checks done before hashing (records/attempt4_tests.log, records/attempt4_check.log)
- eval/verify/attempt4_tests.py (CPU): the head leaves the action distribution bit-identical; at initialization the class loss gives zero actor gradient;
  ppo_loss_i with eps 0, equal coefficients and w_cls 0 equals ppo_loss (loss difference 0, gradient difference 3.3e-16); the sampler frequency matches p'
  for five logits; in mode-0 rollouts the recorded log-probability equals logp_mix(0.05) recomputed from the recorded observation, action and gripper (max
  difference 7.5e-15), so the PPO ratio is 1 before the update; modes 1 and 2 are unchanged by GRIP_EPS.
- Arm N with the changed code reproduces the pre-change code bit-identically for one iteration from 320 (GPU, deterministic flag): hist, evaluation
  record and parameters at 321 identical. script/port_check.py ref=pc: EXACT (192 episode costs).

## Evaluation (at 920; also at 620 and every 100 iterations descriptively)
- Decision evaluation: eval/diagnostics/l2_pilotO_eval.py (256 episodes per start from the original, phase-2 and phase-3 starts, the pilot-O episodes and
  PRNG key 555, deterministic for decisions; revision-15 cost; GPU, deterministic flag), records/l2_pilotO_eval.json, as in pilot G.
- Phase success: eval/diagnostics/phase_success.py (definitions P1, P2, P3, P3', P4, chain, as fixed on 2026-11-16 in that file; 256 episodes per start,
  key 555; CPU), deterministic and gripper-sampled, records/phase_success.json; I at 320 is l2_pilotG_C:320 (same parameters).
- Bootstrap: paired over episodes, 10000 resamples, numpy default_rng(0), percentile 95% CI (eval/diagnostics/phase_diff.py, l2_pilotG_decide.py).
- All differences are I - N. Decisions by eval/diagnostics/l2_attempt4_decide.py into records/l2_attempt4_decision.json.

## Fixed rules
- Guarantees: violations 0 and non-finite states 0 on every iteration 321-920 of both arms (hist viol, nan); the minimum tank margin (hist
  tank_bound_min) is reported. A violation or a NaN stops the runs.
- (T) Task, on P1, P2, P3, P3', P4 alone and the full chain (chain P1-P4), deterministic, phase_diff.diff: a quantity "improves" if the difference is at
  least +0.10 and its CI lower bound is above 0; it is "non-inferior" if the difference is at least -0.10. The margin 0.10 is the size of the untargeted
  between-arm differences of pilot G (above).
- (Hd) Hand role on common rollouts (eval/diagnostics/l2_attempt4_hd.py, records/l2_attempt4_hd.json): each arm's final hand (hp at 920) paired with the same
  brain, arm N's final brain (bp, action mode and observation layout of l2_a4_N at 920), on the same evaluation episodes and keys. Pass if all three hold:
  (i) transfer residual (l2.hand_diagnostics res_learned, median over the 256 original-start episodes, as the 'hand' entry of l2_pilotO_eval) of I's hand
  at most 0.5 times that of N's hand; (ii) hand error while holding, mis_hold (l2.rollout and l2.summarize: |u_ad - u*| averaged over the last 40% of
  phase 2), per episode from the phase-2 start (object held from the start), deterministic, 256 pairs, I's hand minus N's hand: CI lower bound at most 0
  (the CI includes or lies below 0); (iii) (T) non-inferior under that pairing: phase success (deterministic) of N's brain with I's hand minus N's brain
  with N's hand, every quantity of (T) at least -0.10. This replaces the role-loss ratio of pilot L for this test: there each arm was measured on its own
  brain's rollouts, so the role loss mixed the hand with the states the brain visited, and the 1.10 threshold had no stated basis.
- (Ld) Ledger reading, from the phase-3 start of the decision evaluation, as defined in pilot G (eval/diagnostics/l2_pilotG_decide.py, function ledger):
  class decision (class_ok on episodes with top-face force in both compared evaluations) and class gap g (coefficient of h (f_heavy - f_light) in
  fbar ~ a + b f_light + c f_heavy + g h (f_heavy - f_light)). Pass if both hold: (i) class decision I - N, CI lower bound above 0; (ii) masking the load
  entry of the ledger in I's final policy (l2.rollout blind=1: the brain sees u_load = 0, hence also a zero tip load; the hand is unchanged; same episodes
  and keys, deterministic) lowers its class decision: I minus I masked, CI lower bound above 0. g is reported with its CI and is not part of the rule.
- Adoption: I becomes the baseline configuration if (T) is non-inferior on every quantity and either (T) improves at least one of P3, P4, full chain, or
  (Hd) and (Ld) both pass. Otherwise nothing is adopted and the ablations to run are proposed (not started).

## Descriptive (no rule)
Gripper-sampled phase success next to deterministic; first-contact peak normal force per start for both final policies (eval/verify/contact_peak_record.py
generalized to a run directory; shares above 60 N and 70 N); the class-reading head's accuracy; training drops under grip_eps; KL, clip share, sigma,
role loss, residual over iterations; the gripper diagnostic of B0 for both final policies.

## Budget
One arm alone takes about 35-43 s per iteration at N_EP 1024 (pilot G; I smoke 42.7 s); two concurrent processes on one GPU, estimated 10-12 h for 600
iterations, then about 1 h of evaluations.

## Outcome

## Amendment 1 (2026-11-17, approved by the user before restarting arm I; arm N untouched)
- Change: item (5) of arm I applies only to decisions in phase 4: eps = 0.05 on phase-4 decisions of training rollouts (mode 0), eps = 0 in phases 1-3.
  The recorded log-probability and the PPO log-probability and gripper entropy use the distribution actually sampled (p' on phase-4 decisions, sigmoid(logit)
  elsewhere; l2.rollout and l2.ppo_loss_i, phase from the observation's phase indicator, l2.PH4). Evaluation unchanged.
- Reason: with eps on every decision the training gripper opened with probability at least 0.025 per decision while holding, whatever the policy, and
  forced drops. This confounds the test of L_tr and the ledger, whose effects need held carries and presses. B0 above shows that phase-4 losses come
  mostly from the release position (0.422 released outside 2 cm at the phase-4 start), not from failing to open, so the exploration is kept where release
  happens.
- Iterations of I already run under the original item (5): 321-329 (9 iterations, run folder deleted). Seen: the share of training episodes (original
  starts) that released before phase 4 was 0.79 at iteration 321 (rel_before4 in the training log). Nothing else was examined.
- Restart: I from l2_pilotG_C at 320 in the same run group (20261117004739), iterations 321-920, same command. I finishes after N; the evaluation and
  decisions run when both are done. All decision rules stay as registered.
- Code: the phase mask is the only code change. Tests (eval/verify/attempt4_amend_tests.py, records/attempt4_amend_tests.log): with eps 0.05, phases 1-3
  reproduce the unmodified sampling bit-identically (gripper draws, actions, observations, log-probabilities); phase-4 log-probability equals logp_mix(0.05)
  and the gripper frequency matches p'; ppo_loss_i reproduces the recorded log-probability (approximate KL -1.1e-17, no clipped sample).
  script/port_check.py ref=pc: EXACT. Arm N's code path (GRIP_EPS 0) is unchanged by the amendment, so N keeps running on the code it started with; the
  group manifest records both code states (key code_history).
