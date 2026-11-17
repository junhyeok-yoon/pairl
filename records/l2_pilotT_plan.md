# Pilot T: task-space action of the brain against the joint-space action (pre-registration)

Written before either arm was run. The user approved this experiment on 2026-11-15, as the recommended direction after pilot R (records/l2_pilotR_plan.md,
outcome).

## Question
Does a task-space action with class-level kinematics give the brain a learning signal and skill in the later phases (carry, press), and precise reaching, compared
with the absolute joint-space action under the same conditions?

## Action modes (code/stageb/l2.py, goal_of)
- Joint (current): the goal posture is q_d = c + h tanh(u_1:3), inside the joint box.
- Task: q_d = proj_box(q + J_nom(q)^+ dp + n v_N), where:
  - dp = 0.3 tanh(u_1:2) m, and n = 0.3 tanh(u_3) rad;
  - J_nom is the Jacobian of the nominal link lengths, used through the damped pseudo-inverse J^T (J J^T + 0.02^2 I)^-1;
  - v_N is the unit kernel direction of J_nom;
  - q is the measured posture.
  Stiffness, gripper, observation and networks are the same in both modes.
- Checks of the task mode before the pilot:
  - Zero action gives q_d = q exactly.
  - For |dp| <= 0.05 m, the nominal tip moves by dp within 1.4 mm (median; 95th percentile 4.8 mm).
  - A null-space step of 0.1 rad moves the tip by 0.5 mm (median).
  - Rollouts at initialization (64 original starts, 32 phase-2 and 32 phase-3 restarts): 0 joint violations, 0 non-finite states, minimum tank-bound margin 0.277 J.
- Initial exploration, matched in task space: joint mode with initial goal sd 0.1 gives a goal-tip deviation of 0.067 m per axis (rms, at initialization); task mode
  with initial sd 0.22 gives 0.063 m.

## Arms (one seed each, the same seed)
Both arms:
- start from initialization (seed 0, identical network weights apart from the initial goal sd);
- use cost revision 15, role weighting, restart share 0.5 (env.restart, as arm B of pilot R) and 128 episodes per iteration from the same pool;
- run 100 iterations, evaluated every 10 iterations from iteration 0 (original start and both restart starts), in parallel on the same machine.
Arm J (records/l2_pilotT_J):
  python3 l2_train.py ../../records/l2_pilotT_J 100 128 6144 restart=0.5 eval_restart=1 act=joint sd_goal=0.1
Arm T (records/l2_pilotT_T):
  python3 l2_train.py ../../records/l2_pilotT_T 100 128 6144 restart=0.5 eval_restart=1 act=task sd_goal=0.22

## Readings at iteration 100 (deterministic, the 64 evaluation episodes)
- (T2) From the phase-3 restart: the share of accurate presses (relative force error below 0.5 over the force window, attached at the start of phase 4).
- (T1) From the phase-2 restart: the median hold error at p* among episodes attached at the start of phase 3, and the share attached.
- (T0) From the original start (dissection at 100, cost 4): grasp at the end of phase 1, median nearest distance, and the share slow enough to grasp at the nearest
  point.
- (TS) Learning signal from the restart states (l2_signal_check.py, 512 episodes, cost 4): the correlation between the advantage and the goal-tip deviation toward
  the target, for held phase-2 decisions from the phase-2 start and held phase-3 decisions from the phase-3 start; with the sd of that deviation.
- (TH) Hand: mean over evaluations 10-100 of the median transfer residual.
- (C1) on every iterate of both arms.

## Decision rule
- Adopt the task-space action (spec revision 16) if (G) holds and at least one of (P1), (P2) holds:
  - (P1) T minus J is at least 0.25 in (T2);
  - (P2) T's hold error in (T1) is at most half of J's, and T's attached share is at least J's minus 0.1;
  - (G) T's grasp is at least J's minus 0.05.
- Otherwise it is not adopted, and the dissection decides what is proposed next.
- (T0), (TS) and (TH) are reported. A failure of (C1) stops that arm.
Products:
- dissections at 100;
- restart behavior (l2_restart_behavior.py) and the signal checks;
- animations of T from all three starts and of J from the phase-3 start;
- the comparison figure (fig_l2_pilotR.py with the arm labels).

## Outcome at iteration 100 (records/l2_pilotT_J, l2_pilotT_T; dissect_it0100.json; records/l2_restart_behavior.json, l2_signal_check.json)
Decision rule:
- (P2) holds. Hold error at p* from the phase-2 start: T 0.0642 m against half of J's 0.1411 m (0.0706); both arms had all 64 episodes attached. T's error was below
  J's at every evaluation from iteration 10 on (T 0.193, 0.185, 0.148, 0.134, 0.117, 0.083, 0.085, 0.073, 0.071, 0.064; J 0.321, 0.289, 0.256, 0.233, 0.196,
  0.202, 0.165, 0.154, 0.140, 0.141).
- (P1) fails. Accurate presses from the phase-3 start: T 0.078, J 0.094.
- (G) holds. Grasp: T 0.031, J 0.016.
- The task-space action is adopted (spec revision 16).
Reported readings:
- (T0) Original start, dissection at 100.
  - Nearest-distance median: T 0.0172 m, J 0.0868 m. Share within 2 cm: T 0.547, J 0.063.
  - Share slow enough to grasp at the nearest point: T 0.016, J 0.078.
  - Goal tip to grasp point at the end of phase 1: T 0.067 m, J 0.172 m.
- (TS) Learning signal from the restart states (512 episodes, about 6,400 held decisions per 128 episodes):
  - phase 2 toward p*: T 0.096 [0.029, 0.149], J 0.021 [-0.016, 0.055]; sd of the deviation along the target T 0.0082 m, J 0.0145 m;
  - phase 3 toward p_c: T 0.004 [-0.035, 0.046], J 0.038 [0.007, 0.074]; sd T 0.0060 m, J 0.0155 m.
- (TH) Mean median transfer residual over evaluations 10-100: T 0.954 N m, J 1.475 N m. The channel test's class decision: T 0.969, J 0.891.
- (C1) Both arms: 0 joint violations and 0 non-finite states in 100 iterations. Minimum tank-bound margin: J 2.7e-7 J, T 2.6e-4 J.
Descriptive: phase-3 start, deterministic.
- T held the object above p_c without pressing in 54 of 64 episodes: tip about 0.069 m above the face, lateral offset 0.014 m. In contact, its median force was 0.80 f*.
- J was in contact in 21 episodes, median force 1.28 f*, and hovered in 43, at about 0.029 m above the face with lateral offset 0.081 m.
Correction to the outcome above: the minimum tank-bound margin of J over the 100 iterations is 2.7e-9 J, not 2.7e-7 J (2.7e-7 was its minimum over the first 50
iterations). Both arms stay above the (C1) limit of -0.01 J.
