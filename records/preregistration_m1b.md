# Pre-registration: M1b root cause and fixes for the failed M1 tests T6 (guard) and T8 (gate), and the windup of variant M

Fixed before any M1b run. Same system, plants, seed and scripted brains as M1 (records/preregistration_m1.md, hash 2cd695a1).
Code: code/m1b_fixes.py. Results: records/m1b_fixes.json.

## Root-cause measurements (descriptive)
D6: in the baseline guard, per adversarial run: number of steps in contact on which the guard was not blocking after the first time the sensed torque reached tau_max, and the largest one-step increase of the spring torque K(q_d - q) on such steps.
D8: in the baseline gate with T(0) = 1.2 J: tank decrease accumulated while T < T_min, split by the gate factor of the step (alpha = 1, 0 < alpha < 1, alpha = 0).

D-M: in M1 the adversarial runs of variant M reached a steady mismatch of up to 39.864 Nm. Candidate cause: the drive e_q - tau_hat/K equals the mismatch at rest only when the applied torque equals the hand torque, which fails under saturation. Measured per run: fraction of contact steps with saturated torque.

## Fix candidates
F1 yielding guard: while the sensed torque exceeds tau_max, the realized goal moves back along the push direction at rate 20 (|tau_hat| - tau_max)/K and the stiffness does not increase.
F2 spring cap: the spring torque of the hand is saturated, |K e_q| <= tau_max, with the baseline guard.
G1 power margin: a step counts as injecting when q_dot tau_h > -0.5 W.
M' corrected sensed drive: calibration drive (u_ad - tau - tau_hat)/K with the applied torque tau of the previous step, which equals the mismatch over K at rest also under saturation, gating and constraint activity; allowed in contact like M.

## Criteria
F1, F2 (each, variants E, M and M'): adversarial mean contact torque on [9, 10) s <= 6.5 Nm in every run, and nominal |mean contact torque - 3| <= 0.3 Nm in every run.
G1: min T >= T_min - 0.05 J in every gate run.
M': adversarial mean |mismatch| on [9, 10) s <= 0.3 Nm in every run, with each guard (baseline, F1, F2), and nominal T5b in every run.

## Predictions
Q1: in every baseline adversarial run above 6.5 Nm the guard unblocked after the first crossing and the largest one-step spring increase exceeds 0.5 Nm.
Q2: F1 passes for E and M. F2 passes for M and fails for E (the mismatch of E adds to the capped spring torque).
Q3: the tank decrease below T_min comes mainly from steps with alpha = 1, and G1 passes.
Q4: every adversarial M run with steady mismatch above 0.3 Nm has saturated contact steps, and M' meets its criterion with every guard.

## Decision rule
A candidate that passes all its criteria is proposed for the spec, subject to approval. If no candidate passes, the failed test stays open and M2 waits.
