# Pre-registration: L0 component tests of the v3 controller on Stage B (no learning)

Fixed before the confirmatory runs. Environment: Stage B (verified; criteria hashes in records/stageb_env_criteria.hash).
Code: code/stageb/controller.py, l0.py, l0_run.py. Results: records/l0_results.json.

## Engineering pilot (not used for any claim; seeds 900-975)
- Spec law (z' = -lam G K^-1 m_hat, no normalization) with the class basis: a sustained oscillation of period about 0.5 s after a heavy grasp when updates are not rest-gated;
  with the rest gate (v_r = 0.1) slow calibration (median mismatch ratio 0.18 at 1 s, worst 0.65).
  Cause: d u_ad/dt = -lam G^T G m_hat couples the joints (G^T G is far from diagonal), so the stiff joint 1 drives the light joints 2 and 3.
- Normalized law z' = -lam G (G^T G + eps I)^-1 m_hat, which gives d u_ad/dt = -lam m_hat: with lam = 5, v_r = 0.3 the median ratio at 1 s was 0.016 (worst 0.173).
- Static error at limit-adjacent targets equals K^-1 tau_b (the memory correctly does not cancel the barrier): 38-75 mm with k_b = 20, 2.6-39 mm with k_b = 1.
- Small budgets: the tank went up to 0.64 J below the continuous bound T_min - sup U_b (E0 = 0.5 J), with T > 0. Cause: the gate tests the power at the start of a control
  period; a constant torque does positive work 0.5 T_s^2 tau^T M^-1 tau within the period.
- The candidate constants below were set after the pilot.

## Controller (minimal configuration, measured signals only)
T_s = 2 ms, T_h = 50 ms; filters Lambda_q = 20, Lambda_K = 10 s^-1; K in [k_lo, k_hi], k_lo = (50, 25, 10), k_hi = (400, 200, 80), K_n = (200, 100, 40) N m/rad;
D = diag(30, 12, 2) N m s/rad; tau_th = 0.5, tau_max = 25 N m; rest: |q_dot_meas|_inf <= v_r = 0.3 rad/s for t_r = 0.1 s;
memory: u_ad = (gravity of the nominal body) + G(q)^T z, class basis G (7 x 3: S^T e_k cos theta_k, S^T e_3 sin theta_3, e_1, e_2, e_3), |z_i| <= 50;
update law N: z' = -lam_s zeta G (G^T G + eps I)^-1 m_hat, eps = 1e-3 tr(G^T G)/3, lam_s = 5 s^-1, m_hat = u_ad - tau_prev - tau_hat_ext;
tank T(0) = T_min + E0, T_min = 1 J, Delta T = 1 J, p_m = 0.5 W, update with measured q; constraint layer k_1 = 2, p = 8, rho_on = 0.8, rho_full = 0.9, k_b chosen below.
Oracle object basis F-obj (3 x 3, uses the kinematics) only in L0-7.

## Scripted test brains (test harness only; they use nominal kinematics and the task context, never u_star)
nominal: phase 1 goal = best-margin posture of the grasp point, phase 2 = that of the target, phase 3 descends at 0.15 m/s until contact is sensed, then integrates the force error
(gain 0.3 per decision) and the x error; phase 4 releases and returns to the target posture; K = K_n; grasp command on in phases 1-3.
adversarial: as nominal, but after contact the goal moves 0.025 m deeper per decision with K = k_hi.
calib: goal at the grasp posture until 'hold' s after the grasp, then the target posture.
box: goals alternate between two opposite corners of the joint box every 0.5 s with K = k_hi, plus 0.3 u_bar sign(goal - q) added to the hand torque.
hold: goal = best-margin posture of the target (hard episodes), no object.
random: nominal plus random goal offsets (+-0.15 rad, resampled with probability 0.1 per decision), random K in range (probability 0.1) and gripper toggles (probability 0.05).

## Tests and criteria
L0-1 energy accounting (random brain; 128 episodes: 64 in distribution, 32 out-of-distribution bodies, 32 stiff blocks; seeds 1000+):
  (a) max_t |T(t) - T_true(t)| <= 0.1 J in every run (T_true: the same update with the true joint increment); (b) cumulative sum of tau_b^T dq + dU_b <= 0.05 J at every time;
  (c) zero joint-box violations; (d) all values finite.
L0-2 small budgets (nominal brain; 128 episodes, E0 in {0.5, 1, 2, 5} J, 32 each; seeds 1200+): (a) min T >= 0 in every run.
  Descriptive: min over runs of T - T_min + max_{s<=t} U_b (deficit below the continuous bound of Theorem 1(b)); runs with saturation while T < T_min.
L0-3 joint box under stress (box brain; 64 episodes per k_b in {0.25, 0.5, 1, 2}; seeds 1400+): (a) zero joint-box violations. Descriptive: max rho(w_q), max rho(w).
L0-4 static reachability (hold brain, hard targets, K = k_hi, 3 s; 64 episodes per k_b; seeds 1600+): mean tip error over the last 0.5 s (descriptive, decides k_b).
L0-5 guard (adversarial brain; 64 episodes; seeds 1800+): mean contact torque |J^T f_e| over the last 0.5 s of the press phase <= tau_max + mean |u_ad - u_star| + 1.0 N m in every run.
L0-6 calibration (law N): (a) calib brain, hold 1.5 s, 64 episodes (seeds 2000+): mismatch ratio |u_ad - u_star| at 1.0 s after the grasp / just after the grasp: median <= 0.1, 90th percentile <= 0.3;
  (b) nominal brain with the memory reset at contact onset, 64 episodes (seeds 2200+): ratio at 1.0 s after contact onset / at onset, over episodes in contact for >= 1.0 s in the press phase:
  median <= 0.1, 90th percentile <= 0.3; (c) zero memory updates at a true joint speed above 0.35 rad/s in all L0-6 runs.
L0-7 generalization across postures (descriptive; calib brain, hold 1.5 s; F-class and F-obj; 64 episodes each, seeds 2400+): ratio of the mismatch at the target posture at the first rest
  after the move to its value with the prior only. Prediction: median for F-obj <= 0.3 and below that of F-class.
L0-8 update law and stiffness (calib brain, hold 3 s; 64 episodes per cell, seeds 2600+): laws {spec (z' = -lam G K_n K^-1 m_hat, lam = 2), N (lam = 5)} x K in {k_lo, K_n, k_hi}.
  Criteria for law N: ratio at 1.0 s median <= 0.1 at K_n and k_hi, <= 0.3 at k_lo; ratio at 2.9 s <= ratio at 1.0 s + 0.05 in every episode.
  Prediction: the spec law fails the 1.0 s criterion at K_n.

## Decision rules
k_b = the largest value in {0.25, 0.5, 1, 2} with zero violations in L0-3 and median tip error <= 0.01 m in L0-4; if none meets the error bound, the smallest value with zero violations (flagged).
L0-5, L0-6 and L0-8 use the chosen k_b. L0-1, L0-2(a), L0-3(a), L0-5, L0-6 and the law-N criteria of L0-8 must pass before L1; a failure is diagnosed and fixed under a new pre-registration.
If law N passes and the spec law fails as predicted, the normalized law (lam_s = 5, v_r = 0.3) is proposed for the specification (approval needed). L0-7 informs the design of L2 only.
