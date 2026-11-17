# Pre-registration: M1 mechanism validation (Stage A, no learning)

Fixed before any M1 run. Code: code/m1_stage_a.py. Results: records/m1_stage_a.json.

## System
One joint in a vertical plane, link 1.5 kg at 0.2 m, rotor and link inertia 0.05 kg m^2, payload 0, 2/3, 4/3, 2 kg at 0.4 m.
Input limit 20 Nm. Joint box [-0.4, 1.0] rad. Viscous friction U[0.05, 0.3], constant disturbance U[-0.5, 0.5] Nm.
Wall at q_w ~ U[0.5, 0.6] rad, stiffness {500, 2000, 5000} Nm/rad, damping U[1, 5], unilateral.
Sensed contact torque: true value with one control-step delay, bias U[-0.5, 0.5] Nm, noise sd 0.05 Nm; bias tared over [1.5, 2.0) s.
24 plants (4 payloads x 3 stiffnesses x 2 draws), numpy seed 0. Control 1 ms (zero-order hold), plant RK4 4 substeps, brain 50 ms, horizon 10 s.

## Controller (minimal Stage A configuration of the spec)
Hand: impedance K e_q, fixed damping 3; memory z_a only, quadratic H with gain 2, 8 Gaussian features (width 0.2) on the joint box plus a constant, no prior, box projection 10.
Full-power tank, discrete exact update T_{k+1} = T_k - tau_k (q_{k+1} - q_k), T(0) = 20 J, T_min = 1 J, gate ramp 1 J.
Contact guard: tau_th = 0.5 Nm, tau_max = 6 Nm. Constraint layer: gauge form, k1 = 2, k_b = 20, rho_on 0.8, rho_full 0.9.
Variant E (spec): calibration drive e_q, frozen while the sensed contact torque exceeds tau_th.
Variant M (alternative): calibration drive e_q - tau_hat/K after the tare, allowed in contact.
Scripted brain: hold 0.3 rad; calibration request on [0.5, 2.0) s; approach from 2.2 s at v_a = 0.1 rad/s; on sensed contact, goal = contact position + 3/K and an outer force loop on the goal every 50 ms (gain 0.3, target 3 Nm); variant M requests calibration from 0.5 s after contact.
Adversarial brain (guard test): after contact, goal moves 0.025 rad deeper every 50 ms, stiffness 200, calibration always requested.

## Tests and criteria
T1 energy identity: max |E_rob + T + W_fric - W_env - (E_rob(0) + T(0))| / scale <= 1e-6 in every run.
T2 delivered energy: max_t of -int q_dot tau_c dt <= E_rob(0) - min E_rob + T(0) in every run (true torque); sensed version reported.
T3 joint box: zero samples outside [-0.4, 1.0] in every run, including a free-space test with goals at both limits, stiffness 200 and an added hand torque of 0.3 x input limit toward the goal.
T4 calibration: |mismatch| at the end of calibration <= 0.1 x |mismatch| at its start, every nominal run.
T5 contact accuracy: |mean contact torque on [9, 10) s - 3| <= 0.3 Nm, every nominal run.
T5b role in the mechanism: mean |mismatch| on [9, 10) s <= 0.3 Nm.
T6 guard: adversarial runs, mean contact torque on [9, 10) s <= 6.5 Nm, every run.
T7 impact (exploratory, no criterion): peak contact torque within 0.3 s after first contact vs v_a in {0.05, 0.1, 0.2, 0.4} rad/s and stiffness.
T8 gate: T(0) = 1.2 J, min T >= T_min - 0.05 J in every run.

## Predictions
P1: variants E and M both pass T1-T6 and T8.
P2: variant M meets T5b in every run; variant E fails T5b in at least half of the runs.

## Decision rule
Any failure of T1-T4, T6 or T8 by variant E: fix the mechanism before M2.
P2 holds: propose calibration in contact with the sensed signal (variant M) for the spec, subject to approval.
Variant M fails T5b: calibration in contact is not supported; keep variant E.
