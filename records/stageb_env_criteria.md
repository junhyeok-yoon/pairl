# Stage B environment: verification criteria (fixed before the verification runs)

Code: code/stageb/{model,env,params,refs}.py, tests code/stageb/verify_dyn.py and verify_env.py. Results: records/stageb_dyn_check.json, records/stageb_env_check.json.
All tests in float64 unless stated. "rel" = max abs difference / (1 + max abs reference).

V1 kinematics: tip position and Jacobian vs SymPy and MuJoCo, rel <= 1e-12 (position), 1e-9 (Jacobian).
V2 mass matrix: vs SymPy rel <= 1e-12, vs MuJoCo rel <= 1e-10, asymmetry <= 1e-12, positive definite on every sample.
V3 forward dynamics: joint acceleration vs SymPy rel <= 1e-10, vs MuJoCo rel <= 1e-9, also with a force at the tip; 6000 states, 240 parameter sets (half out-of-distribution bodies, half with the object).
V4 gravity: vs SymPy rel <= 1e-12, vs MuJoCo rel <= 1e-10; static hold with tau = g - d gives |qdd| <= 1e-9.
V5 integration: free motion without friction; relative energy drift over 1 s <= 1e-7 at 4 substeps (0.5 ms); observed order between 1, 2 and 4 substeps in [3.5, 4.5].
V6 power balance without contact (random torques, friction, disturbance): |E_rob(t) - E_rob(0) - (W_act - W_fric)| <= 1e-6 max(1, max|W_act|) at 4 substeps.
V7 contact: (a) nine constructed force values exact to 1e-9; (b) pointwise dS_env/dt + v.f_e <= 1e-9 on 100000 random contact states, and W_env(t) <= S_env(0) - S_env(t) + 1e-6 along simulated runs;
   (c) power balance with contact at 4 substeps <= 1e-4 max(1, max|W_act|) for stiffness <= 2e4 N/m; stiffness 2e5 N/m reported for 2, 4, 8 substeps and used to choose the substep count; (d) state difference between 4 and 16 substeps reported.
V8 gripper: (a) truth table of the event logic (12 cases) exact; (b) M+ dq+ = M- dq- to rel 1e-7 against an independent impulse computation, kinetic energy never increases; (c) energy jump at a grasp at rest equals the potential energy of the object to 1e-9;
   (d) no re-grasp after a release, joint velocity unchanged by a release; (e) jump of the gauge at admissible grasps reported.
V9 sensing: (a) one-step delay exact with noise off; (b) torque noise mean within 4 standard errors of 0, standard deviation within 2% of the truncated-normal value, |n| <= 0.15 N m; (c) quantization error <= 0.5e-4 rad;
   (d) with the object attached at rest in free space the sensed torque minus bias is 0 with noise off; (e) tip position noise standard deviation within 2%.
V10 sampler: 3000 episodes (2000 in distribution, 500 out-of-distribution bodies, 500 stiff blocks); every stated constraint satisfied on every episode; hard and budget fractions within 3 binomial standard deviations of 0.3.
V11 cost: 10 constructed states (all phases, contact and no contact, force above 2 f*) agree with an independent NumPy evaluation to 1e-9.
V12 batch and determinism: vmap batch equals single-episode runs to 1e-9; two runs with the same seed identical; float32 vs float64 difference over an episode reported.
V13 timing: phase index at the phase edges exact; gripper state changes only at decision steps; episode length equals the sum of phase lengths.
V14 end to end (privileged test controller, not the v3 structure; 64 in-distribution episodes, 4 substeps): all values finite; energy audit residual <= 1e-4 max(1, max|W_act|); grasp achieved in >= 90% of episodes; one episode plotted.
V15 trajectory vs MuJoCo RK4 (free motion with friction, constant torque per control step, 1 s): max joint difference <= 1e-6 rad.

Decision rule: every criterion must hold before any controller is built on the environment. A failed criterion is fixed in the environment (or the criterion is shown to be wrong and replaced with a stated reason), and the whole suite is rerun.

## Amendment 1 (after the first run; reasons stated, criteria unchanged unless listed)

First run (hash ccaed5dd): V5, V6, V8-V11, V13, V15 passed; V7, V12 (float32 part), V14 failed.
- V7(a),(b): the side-face damping had the wrong sign (code error). Fixed in the environment.
- V7(c): with the spring-damper (Kelvin-Voigt) law the contact force is discontinuous at touch-down; the power-balance residual at 4 substeps was 8.5e-3 at 2e4 N/m. The law is replaced by the Hunt-Crossley form
  f = k_e d max(0, 1 + delta_e d_dot), continuous at touch-down and lift-off, with dissipation delta_e ~ U[0.5, 5] s/m, and the friction regularization speed is 0.02 m/s.
  The nine constructed values of V7(a) are recomputed for the new law (k_e = 1e4, delta_e = 2): 0; 10; 12; 0; -5 tanh(10), 10; 0; (50, 0); (60, 0); (-50, 0).
- V14: the test controller drove the tip through the block side on the way to the grasp point. The sampling rule now places the grasp point and the initial tip at least 0.03 m above the block top (V10 checks the new rule).
- V12: importing the design-study module enabled float64 globally; the NumPy kinematics moved to a module without that side effect.
The whole suite is rerun.

## Amendment 2 (after the second run; no criterion changed)

Second run (hash 9a1b3517): all checks passed except V7(c): residual 2.1e-3 at 2e4 N/m and 4 substeps, with a jump to 6.7e-7 at 8 substeps.
Cause: the friction regularization tanh(v / v_t) has a slope at zero sliding speed proportional to the normal force, which leaves the stability region of RK4 at impact forces above about 100 N.
The friction is now -nu f_n tanh(c_v v / (nu f_n)) with c_v = 400 N s/m (slope bounded independently of the normal force). Constructed value 5 of V7(a) becomes (-5 tanh(16), 10).
A descriptive table of the power-balance residual by regime (operating, harsh), stiffness and substep count is added (code/stageb/contact_runs.py, records/stageb_contact_runs.json) and is used to choose the substep count.
The whole suite is rerun.

## Amendment 3 (2026-11-14, decision of the user: re-grasping must be allowed; written before the runs)

Model change. The object is a rigid planar body with a point mass at its center of mass c and a material grasp point at distance L_obj <= 0.08 m from c, pose (c, psi) with grasp point c + L_obj e(psi).
Initially the grasp point is at p_o and psi is uniform on [-pi, pi]. A detached object rests at its pose (the same support abstraction as the initial object). At a grasp, c and psi do not change;
the offset r = R(theta_3)^T (c - p(q)) and the relative orientation psi - theta_3 are fixed while attached (|r| <= L_obj + 0.02 m). At a release the object rests at its current pose. Re-grasping is allowed.
The grasp condition uses the object's current grasp point. Cost: phase 1 uses the distance to the object's grasp point; phases 2 and 3 without the object use 1 + (distance to the object's grasp point)/0.1.

Test changes.
- V1-V4: object offsets are two-dimensional with |r| <= 0.10 m in SymPy and MuJoCo (same tolerances).
- V8(a): truth table of 11 cases without the release flag, plus the sequence grasp-release-grasp at the same place (allowed) with the grasp point after the release equal to the tip to 1e-12.
- V8(c): at an exact grasp at rest, E_rob jumps by m g c_y of the resting object to 1e-9, and the center of mass is continuous to 1e-12.
- V8(d): on 2000 random states (grasp with |q_dot| <= 0.1 rad/s, release with |q_dot| <= 2 rad/s): the change of E_rob + S_obj (S_obj = m g c_y of a resting object) at every event <= 1e-9;
  at a release it equals minus the object's kinetic energy to 1e-9; the released center of mass equals the attached one to 1e-12; q_dot unchanged by a release.
- V10: L_obj in [0, 0.08] and |c0 - p_o| = L_obj to 1e-12 (replaces the check of the scalar offset).
- V11: the cost of the definition above, 12 constructed states.
- V16 (new): re-grasp end to end on 64 episodes with a test controller that releases at the target, moves to another posture with the same tip position (last-link angle changed by >= 0.4 rad when such a posture exists) and grasps again:
  >= 90% of episodes with two grasps and two releases; every event change of E_rob + S_obj <= 1e-9; center-of-mass continuity <= 1e-12; attached offset <= L_obj + 0.02 m (+1e-12); audit residual <= 1e-4; all values finite.
- The design study ((B3) sampling) is rerun with two-dimensional offsets |r| <= 0.10 m.
All of V1-V16 are rerun.

## Amendment 4 (after the fourth run; no criterion changed)

Fourth run (hash 6cc72312): V1-V15 passed; V16 failed only the audit residual (0.058 in 1 of 64 episodes, below 1.4e-8 in the others).
Cause: a test error. In that episode the initial tip was 0.0131 m from the grasp point, so the grasp happened at the first decision step (k = 0), and the audit took the energy after that event as its reference.
The audit (V14, V16) now uses E_rob minus the accumulated event jumps, whose value at k = 0 is the energy before any event. Same criterion (1e-4). V1-V16 are rerun.

## Amendment 5 (2026-11-14, decisions of the user after the L1 pilot; written before the runs)

Task change; the plant, contact, gripper and sensing models are unchanged.
- In-distribution payload: uniform on [0.2, 0.65] U [1.05, 1.5] kg (one uniform draw mapped onto the two intervals, so every other draw of the sampler is unchanged). Out of distribution unchanged ([1.5, 2.5] kg with the out-of-distribution body).
- Force targets: a pair (f_light, f_heavy), uniform on {(f1, f2) in [3, 15]^2 : |f1 - f2| >= 4} N by rejection, drawn after all earlier draws. The target is f* = f_heavy if m_obj > 0.85 kg and f_light otherwise; only the pair enters the task context.
- Budget factor: every episode draws rho uniform on [0.6, 1] after the force pair. In budget episodes E0 = Delta T + rho E_task, where E_task comes from the L1 reference brain on the same episode (set outside the environment).
- The previous task is kept as sampler version 1 for the scripts that reproduce earlier records (L0, L1 pilot, contact runs, float32 run); their records are not regenerated.
Test changes.
- V10: f_light and f_heavy in [3, 15], |f_light - f_heavy| >= 4, f* equal to the class target, class flag equal to [m_obj > 0.85], rho in [0.6, 1], in-distribution payload in the union;
  the fraction of the heavy class among the 2500 episodes with in-distribution payload within 3 binomial standard deviations of 0.5.
- V11 unchanged (the cost uses f*).
verify_dyn.py, verify_env.py and the design record are rerun on the current task version.

## Amendment 6 (2026-11-15, task cost of spec revision 14 approved by the user; written before the run)

Task cost change only; the plant, contact, gripper, sensing models and the sampler are unchanged.
- Phases 2 and 3: the rate is the progress cost of the object at its pose plus (1 - xi)(1 + |p - o|/0.1). The object's progress cost is |o - p*|/0.1 in phase 2 and
  Pi(o, f_n) + |o_x - p*_x|/0.1 in phase 3, evaluated at the tip with the top-face force while held and at the object's resting pose with zero force while not held;
  without top-face force Pi = 1 + |o - p_c|/0.1 with p_c = (p*_x, y_e). Phases 1 and 4 and the excess-force term unchanged.
Test changes.
- V11: the independent NumPy evaluation implements revision 14; the same 12 constructed states; tolerance 1e-9.
- V17 (new), cost properties on 200000 random postures of the V11 episode's plant: (a) the rate with the object dropped at the tip's pose minus the held rate, at states
  without top-face force, equals 1 within 1e-9 in phases 2 and 3; (b) along the side wall of the block (|p_x - x_e| in [w_e, w_e + 0.01], 0 < y_e - p_y < 0.3 m,
  at least 100 states) the least-squares slope of the held phase-3 rate with depth is at least 1 per m (revision 13: 0.01 per m).
verify_env.py (V5-V17) is rerun.

## Amendment 7 (2026-11-15, task cost of spec revision 15 approved by the user; written before the run)
Task cost change only. In phase 3, with top-face force f > 0, Pi = min(((f - f*)/f*)^2, 1); revision 14 used ((f - f*)/f*)^2 without the bound. Everything else is
unchanged.
Test changes:
- V11: the independent NumPy evaluation implements revision 15. Same 12 constructed states; tolerance 1e-9.
- V17: unchanged (it uses states without top-face force).
- V18 (new): cost properties with top-face force, on the V11 episode's plant, phase 3. The tip is at the center of the top face, at 200 depths whose forces span
  0.05 f* to 20 f*.
  (a) At every depth, the rate with the object dropped at the tip's pose minus the held rate is at least 1 - 1e-9.
  (b) The held rate is smallest at the depth whose force is closest to f*, and it does not decrease as |f - f*| grows on either side (tolerance 1e-9).
verify_env.py (V5-V18) is rerun.
