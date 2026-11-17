# Pilot R: restart start-state distribution against plain continuation (pre-registration)

Written before either arm was run. The user approved:
- the recommended direction of the attempt-2 outcome on 2026-11-15 (records/l2_attempt2_plan.md, outcome at iteration 300);
- cost revision 15 with the critic warm-up, the same day, after the findings below.
Precondition: the environment check passes under revision 15 (records/stageb_env_criteria.md, amendment 7).

## Question
Does a training start-state distribution that covers held states speed up the brain's learning of the later phases (carry, press), and give the hand data at
the loaded target posture, without slowing the first phase?

## Arms (one seed each, the same seed)
Both arms:
- start from records/l2_attempt2/params_it0300.pkl with fresh optimizer states;
- use cost revision 15 (env.COST_V 4), role weighting (w_mode 1), and 128 episodes per iteration from the same pool;
- run 55 iterations (301-355). Iterations 301-305 are a critic warm-up: only the critic is trained, the actor and the hand are frozen, and the brain's optimizer state
  is reset afterwards. Iterations 306-355 train everything;
- are evaluated at 300, 310, ..., 350 and 355;
- run in parallel on the same machine.
Arm A (records/l2_pilotR_A) keeps the start distribution unchanged.
  Command: python3 l2_train.py ../../records/l2_pilotR_A 55 128 6144 ../../records/l2_attempt2/params_it0300.pkl eval_restart=1 vwarm=5
Arm B (records/l2_pilotR_B) uses restart share 0.5 (env.restart). Of each batch:
- 64 episodes start from the original start;
- 32 start at the beginning of phase 2, at the grasp posture q_o (best-margin inverse kinematics of the grasp point), with the object attached at the tip;
- 32 start at the beginning of phase 3, at the posture q_s of p*, with the object attached as if grasped at q_o.
Restart states have velocity zero, memory zero, tank energy E0, and the episode's remaining phase durations.
  Command: python3 l2_train.py ../../records/l2_pilotR_B 55 128 6144 ../../records/l2_attempt2/params_it0300.pkl restart=0.5 eval_restart=1 vwarm=5

## Changes made before the pilot, and the findings behind them
1. Restart states (env.init_state takes an attached start; env.restart). Checked on the 6144 pool episodes:
   - q_o and q_s lie inside the joint set, with minimum margins 0.259 and 0.119 rad.
   - The grasp point is at the tip, and the offset norm equals the object's length.
   - Rollouts of the iteration-300 checkpoint from 512 restart states of each kind, deterministic and exploring: 0 joint violations, 0 non-finite states,
     minimum tank-bound margin 0.029 J.
2. Role-loss weighting defect (a code error against the definition; fixed).
   - The defect: replay_loss took the attachment at the first control step as the state before the episode. A grasp at the first decision was therefore not a
     load change, and its whole loaded segment had weight 1.
   - How often: 36 of the 6144 pool episodes (0.59%) start with the tip within 2 cm of the grasp point.
   - The fix: the state before the episode is detached.
   - Check on four batches of 32: the unweighted loss is unchanged. The weighted loss is unchanged on the two batches without such an episode, and changed on the
     two with one (1.842 to 0.761, 1.481 to 0.903).
   - Attempt 2 trained the hand with the defect.
3. Critic warm-up. Smoke test of arm B as first registered (revision 14, no warm-up, 2 iterations):
   - The first PPO update had KL 4.12, clip fraction 0.720 and value loss 87.0. Attempt 2 after its resumption had 0.133, 0.425 and 1.41.
   - With 5 critic-only iterations, the critic's explained variance on phase-3 restarts rose from 0.468 to 0.698, and the first two updates had KL 0.767 and 0.190.
4. Revision 14 rewarded releasing under overpressing (spec, fact release14).
   - Releasing at the same configuration changes the phase-3 task rate by 2 + |p - p_c|/0.1 - ((f - f*)/f*)^2, which is negative above (1 + sqrt 2) f* at p_c.
   - Phase-3 restarts of the iteration-300 checkpoint (exploring, 256 episodes): 4039 of 11927 held phase-3 decisions (0.339) lay in that region; the contact force
     had median 4.24 f* and 90th percentile 9.72 f*.
   - Revision 15 bounds the press term by 1; cost audit with top-face force: share of states where dropping is cheaper 0.885 (revision 14), 0.000 (revision 15).

## Baseline at iteration 300 (deterministic, the 64 evaluation episodes; both arms start here)
These measures do not depend on the cost version.
- Original start: grasp 0.109; median transfer residual 2.947 N m.
- Phase-2 start: all 64 episodes attached at the start of phase 3; median hold error at p* among them 0.249 m; top-face force in the phase-3 force window 0.109.
- Phase-3 start: all 64 attached.
  - Top-face force in the force window 0.4375 (28 of 64); median relative force error among those 1.291.
  - Accurate press (relative force error below 0.5 over the force window, and attached at the start of phase 4): 0 of 64.

## Readings at iteration 355 (deterministic, the 64 evaluation episodes)
- (R3, primary) From the phase-3 restart, the share of accurate presses. Also reported: the contact share in the force window, and the median relative force error
  among contacts.
- (R2) From the phase-2 restart: the share attached at the start of phase 3, and the median hold error at p* among them.
- (R0) From the original start: grasp at the end of phase 1, and the median nearest distance to the grasp point in phase 1 (dissection at 355).
- (RH) Hand: mean over the evaluations 310-355 (six) of the median transfer residual.
- (C1) on every iterate of both arms.
- Descriptive. From phase-3 restarts of each arm's iteration-355 checkpoint (exploring, 256 episodes): the share of held phase-3 contacts above 2 f*, the open
  probability in held phase-3 states, and the share of episodes that release in phase 3.

## Decision rule
- Adopt the start-state distribution for attempt 3 if both hold:
  - (P) B minus A is at least 0.25 in (R3);
  - (G0) B's grasp is at least A's minus 0.05, and B's nearest-distance median is at most A's plus 0.01 m.
- Otherwise it is not adopted, and the dissection of both arms at 355 decides what is proposed next.
- (R2) and (RH) are reported. B below A in (RH) supports the coverage reading for the hand.
- A failure of (C1) stops that arm.
Products:
- dissection at 355 of both arms (code/stageb/l2_dissect.py);
- animations from the original start and from both restart states;
- the comparison figure (code/stageb/fig_l2_pilotR.py).

## Outcome at iteration 355 (records/l2_pilotR_A, l2_pilotR_B; dissections dissect_it0355.json; records/l2_restart_behavior.json, l2_overpress_check.json, l2_signal_check.json)
Run note: the first launch command of arm B failed at the shell; B was relaunched about 10 s after A, and both ran in parallel.
Decision rule:
- (P) fails. Accurate presses from the phase-3 start: A 0.016 (1 of 64), B 0.000; B minus A is -0.016 against the required 0.25.
- (G0) holds. Grasp: B 0.125 against A 0.156 (allowed down to 0.106). Nearest-distance median: B 0.0345 m against A 0.0313 m (allowed up to 0.0413).
- The start-state distribution is not adopted.
Reported readings:
- (R3, other measures) Top-face force in the force window from the phase-3 start fell from 0.438 to 0.297 (A) and 0.219 (B). The median relative force error
  among those contacts went from 1.291 to 1.465 (A) and 2.116 (B).
- (R2) All 64 episodes were held at the start of phase 3 in both arms. The median hold error at p* was 0.231 m (A) and 0.247 m (B); the baseline was 0.249 m.
- (RH) The mean median transfer residual over evaluations 310-355 was 1.901 N m (A) and 1.520 N m (B). B is below A, but the six values span 0.738-3.126.
- (C1) Both arms: 0 joint violations and 0 non-finite states in 55 iterations. Minimum tank-bound margin 2.6e-3 J (A) and 2.1e-3 J (B).
Descriptive dissection (added after the outcome, not registered):
- Phase-3 start, deterministic.
  - Held without top-face contact in the force window: 36 (300), 45 (A) and 50 (B) of 64 episodes.
  - Those episodes held the tip about 0.06 m below the face level and about 0.15 m to the side of p_c: beside the block, not over p_c.
  - Force in contact: median about 3.5 f*.
  - The goal-tip height had no positive slope on f/f* (-0.004, -0.008, -0.014), so the brain shows no force feedback.
- Phase-3 start, exploring (256 episodes).
  - Episodes releasing in phase 3: 0.133 (300), 0.227 (A), 0.445 (B).
  - Open probability in held phase-3 states: 0.0025, 0.0048, 0.0130.
  - The median force at release was 0 in both arms at 355, so most releases happened without contact.
- Phase-2 start, deterministic. The hold error at the end of phase 2 (0.250, 0.231, 0.247 m) stayed near the start distance 0.257 m, so carrying was not learned.
- Learning signal from the restart states (512 episodes, 4500-6200 held decisions per 128 episodes; correlation of the advantage with the goal-tip deviation toward the
  target):
  - phase 2 toward p*: -0.035 [-0.075, 0.008] (300), -0.013 [-0.064, 0.034] (A), 0.013 [-0.029, 0.054] (B);
  - phase 3 toward p_c: -0.008 [-0.041, 0.022] (300), -0.030 [-0.070, 0.009] (A), -0.003 [-0.039, 0.032] (B);
  - the sd of the deviation along the target direction was 0.012-0.016 m.
  With abundant held samples the signal is still absent, so coverage did not supply what was missing.
