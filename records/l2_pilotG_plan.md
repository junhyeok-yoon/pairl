# L2 pilot G: the grasp gate and the hand-to-brain ledger channel (pre-registration)

Written 2026-11-16 before running, under the user's standing approval of 2026-11-16 for the chain pilot L -> pilot G -> attempt 4 -> evaluation -> spec
revision 17. Platform: this PC's GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true for training and evaluation (records/l2_pilotL_plan.md, amendment 1),
script/safe_run.sh. One seed.

## Roles served
- Arms S and S2: the brain's role in phase 1, aiming at the grasp point and stopping there (the grasp gate; spec sec:l2status: the phase-1 cost measures
  distance only, so stopping is rewarded only through the sparse grasp event; records/l2_grasp_design.md).
- Arm B: the hand-to-brain channel, the ledger (handoff section 3: "손->뇌로는 에너지 장부"). The learned brain does not use it: class decision 0.452-0.460
  at 300 in attempt 3 (spec sec:l2a3, dissection 1), class-gap coefficient -0.038 (spec sec:l2status), while the ledger read with the L1 rule is right in
  0.911. The joint-space ledger load u_load is in the observation; arm B adds it as a tip load through the nominal kinematics (class-level, as the L1
  reference brain reads it).

## Start and common configuration
- BASE = the configuration adopted by pilot L: NH (N_EP 1024, mb_scale=1, c_ent=1e-2) plus ltr=lambda of the adopted L_tr arm, or NH alone if no L_tr
  arm is adopted (records/l2_pilotL_plan.md, outcome). Start checkpoint: the adopted pilot-L arm at 260 (l2_pilotLd_L05, l2_pilotLd_L1 or l2_pilotLd_C),
  brain and hand optimizer states carried over. Iterations 261-320.
- Arms, one after another in this order, in one group (group=new by the first):
  - C: BASE continued.
  - S: BASE + cost revision 17 (cost=5): in phase 1, while the object is not held, kappa max(0, 1 - d/0.05) min(max(0, |dq|_inf - 0.1)/0.1, 3) is added to
    the rate, kappa = 1/6 (env.cost_rate; the audit of records/l2_grasp_design.md: with kappa <= 1/6 being closer is never costlier than hovering at 5 cm).
  - S2: S + near-object starts (near=0.125): 1/8 of the episodes of each batch (128 of 1024) start phase 1 at rest with the tip 2-8 cm from the grasp
    point (env.near_start), taken from the first-start share (the batch keeps 384 original, 128 near, 256 phase-2, 256 phase-3 starts); training
    statistics on the original starts only.
  - B: BASE + obs_tip=1: two inputs, l1.tip_force(q, u_load)/10, appended to the brain's observation (actor and critic), zero-initialized (rows of the
    first layers and of their Adam moments); l2_train checks at the start that the actor and critic outputs do not change (fourth gate: 0.000e+00).
- Run names l2_pilotG_<arm>. Command: sh script/safe_run.sh mem=16G python script/l2_train.py group=<new|G> l2_pilotG_<arm> 60 1024 6144 <start checkpoint>
  restart=0.5 eval_restart=1 mb_scale=1 c_ent=1e-2 [ltr=lambda] plan=records/l2_pilotG_plan.md "purpose=..." [cost=5 | cost=5 near=0.125 | obs_tip=1]

## Evaluation
eval/diagnostics/l2_pilotO_eval.py at 320 of each arm (256 episodes per start, the pilot-O episodes and keys, deterministic for decisions, fully sampled
secondary), on the GPU with the same flag, always under the revision-15 cost (env.COST_V 4 in the evaluation script) so that S and S2 are comparable
with C and B. Per-episode grasp, top-face force, class decision and targets are stored. Decisions by eval/diagnostics/l2_pilotG_decide.py
(records/l2_pilotG_decision.json): paired bootstrap over episodes, 10000 resamples, numpy default_rng(0), percentile 95% CI.

## Decision rules
- (G) on every iteration 261-320 of an arm (violations 0, non-finite 0, tank-bound margin at least -0.01 J on episodes without saturation below T_min);
  an arm failing (G) is excluded, and the chain stops for the user.
- S and S2 (each against C): adopted if (a) the grasp at the end of phase 1 from the original start, arm minus C on the 256 paired episodes, has a CI
  lower bound above 0, and (b) the cost, arm minus C on the 768 paired episodes (revision-15 cost), has a CI upper bound below +0.5.
  If both pass, S2 is preferred over S only if S2 minus S in grasp has a CI lower bound above 0; otherwise S. If only one passes, that one.
- B (against C), all three: (a) class decision on the phase-3 start (class_ok on the episodes with top-face force in both arms), B minus C, CI lower bound
  above 0; (b) class dependence of the press force: the coefficient g of fbar ~ a + b f_light + c f_heavy + g h (f_heavy - f_light) on the phase-3-start
  episodes with top-face force (g = 1 presses the class target, g = 0 ignores the class; -0.038 in attempt 3), B minus C, CI lower bound above 0 (episodes
  resampled jointly for both arms); (c) cost B minus C, CI upper bound below +0.5.
- What follows (no stop): attempt 4 integrates BASE with the adopted grasp arm (if any) and B (if adopted), as instructed by the user; arms not adopted are
  reported and left out. No combination pilot.

## Descriptive
For C and S: the exploration standard deviation of the goal action (tip displacement, pre-squash and as a tip distance) at phase-1 decisions within 5 cm
of the grasp point against beyond 10 cm, next to the commanded displacement |dp| (about 2.4 cm near the object at NH 200), to say whether the exploration
noise limits aiming. For B: class decision from the original and phase-2 starts, the ledger read with the L1 rule, and the B weights of the new inputs
(norm of the two rows of the actor's first layer at 320). For all: KL, clip share, sigma, hand residual, role loss, grasp, accurate presses, hold error.

## Budget
GPU with the deterministic flag: about 35 s per iteration at N_EP 1024 (measured in pilot L), 60 iterations and evaluations per arm, about 40 min per arm,
about 2.7 h for four arms.

## Outcome
(to be appended)

## Amendment 1 (2026-11-16, before any arm ran; requested by the user)
Arm B's rule gains two conditions on the phase success metrics (eval/diagnostics/phase_success.py, records/phase_success.json; deterministic mode, this
PC's CPU, the decision-evaluation episodes and keys; definitions fixed on 2026-11-16), paired with C at 320:
  (d) P3' (class, alone from the phase-3 start; episodes with top-face contact in the last 1 s of phase 3 in both arms): B minus C, bootstrap 95% CI lower
      bound above 0;
  (e) P3 (press, alone from the phase-3 start: attached, pressing the top face for the last 1 s, mean force within +-10% of f*): B minus C, CI lower bound
      above 0.
Rationale: pressing between the two class targets misses +-10% for both classes, so P3 is largely limited by the unused ledger; if the brain uses the
ledger, P3 should rise. B is adopted only if (a)-(e) all hold. The paired differences of every phase metric (P1, P2, P3, P3', P4 alone, the chain) for
every arm against C are reported with CIs (descriptive except (d) and (e) for B). Bootstrap: 10000 resamples of episodes, numpy default_rng(0), percentile.

## Outcome (2026-11-16; group data/runs/20261116165912; GPU with the deterministic flag; records/l2_pilotG_decision.json, records/l2_pilotO_eval.json,
records/phase_success.json)
Run as registered from l2_pilotLd_C at 260 (BASE = NH; pilot L adopted no L_tr arm), arms C, S, S2, B, 261-320. (G) on every iteration of every arm:
violations 0, non-finite 0, no saturation below T_min; minimum tank-bound margin C 0.0100, S 0.0063, S2 0.0184, B 0.0121 J. S2: 128 near starts per
batch, none not found. B: the zero-initialized inputs changed the actor and critic outputs by 0.000e+00 at the start.
Decisions (paired bootstrap 95% CI):
- S: grasp minus C -0.102 [-0.141, -0.062], cost +0.258 [+0.131, +0.409]: not adopted ((a) fails).
- S2: grasp minus C -0.066 [-0.105, -0.027], cost +0.335 [+0.211, +0.480]: not adopted ((a) fails); S2 minus S +0.035 [-0.008, +0.078].
- B: (a) class decision B minus C +0.036 [+0.012, +0.064] (0.610 against 0.574) holds; (b) class-gap coefficient 0.148 against 0.120, difference CI
  [+0.011, +0.047] holds; (c) cost +0.203 [+0.052, +0.380] holds; (d) P3' +0.032 [+0.008, +0.060] holds; (e) P3 +0.016 [-0.020, +0.051] fails:
  not adopted.
No pilot-G change is adopted; attempt 4 has no pilot-G change to add.
Phase success, deterministic, at 320 (C / S / S2 / B): P1 0.504 / 0.453 / 0.465 / 0.461; P2 0.594 / 0.703 / 0.582 / 0.684; P3 0.215 / 0.168 / 0.176 /
0.230; P3' 0.572 / 0.613 / 0.610 / 0.606; P4 0.152 / 0.125 / 0.188 / 0.121; chain P1-P3 0.070 / 0.039 / 0.059 / 0.082; full chain 0.004 / 0.004 /
0.004 / 0.008 (L1 reference 0.922). Paired against C: S P1 -0.051 [-0.094, -0.012], P2 +0.109 [+0.070, +0.152], P3' +0.029 [+0.004, +0.058]; S2 P4
+0.035 [+0.004, +0.070]; B P1 -0.043 [-0.082, -0.004], P2 +0.090 [+0.047, +0.137], P3' +0.032 [+0.008, +0.060]; the other differences have CIs
containing 0.
Descriptive: exploration near the object (records/l2_sigma_near.json, CPU): within 5 cm the held goal noise as a tip distance is 1.60 cm (C) and 1.67 cm
(S) against commanded displacements of 2.47 and 2.59 cm (ratio 0.64, 0.65; beyond 10 cm 0.27, 0.29). B's new input rows have norms 0.284 and 0.208 at
320 against a median row norm of 2.233 for the other inputs: the brain started to use the ledger, and its press force depends more on the class
(g 0.148), far from g = 1.
Reading (not registered): the speed term and the near starts did not raise grasping in 60 iterations; the held exploration noise near the object is
two-thirds of the command and comparable to the 2 cm grasp radius, which blurs the signal they add (hypothesis). The ledger input moves the class
decision in the intended direction but too little to change the press success within 60 iterations.
