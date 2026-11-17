# Pre-registration: M1c root cause of the guard failure and two further guard candidates

Fixed before any M1c run. Same system, plants, seed and brains as M1 (hash 2cd695a1) and M1b (hash def20757).
Code: code/m1c_guard.py. Results: records/m1c_guard.json.

## Facts from M1b that motivate this round
The single-step spring increase after the first crossing was at most 0.349 Nm (Q1 failed), yet the guard unblocked on up to 83 steps per run, and the yielding guard (F1) made the steady contact torque larger (up to 13.971 Nm). Candidate mechanism: the contact torque oscillates on the stiff wall, the guard unblocks in the troughs, and the goal advances in small steps until the input saturates.

## Root-cause manipulations (variant E, baseline guard, adversarial brain)
R-a: wall damping set to the critical value 2 sqrt(k_e I) (no contact oscillation).
R-b: the adversarial goal proposal stays 0.025 rad ahead of the realized goal instead of accumulating.

## Guard candidates (variants E and M')
F3 latched guard: when the sensed torque first reaches tau_max, the realized goal and stiffness at that moment become limits in the pushing direction; the latch is released after the sensed torque stays below tau_max - 1 Nm for 0.2 s.
F5 contact cap: while the sensed torque exceeds tau_th, the spring torque is saturated at tau_max; baseline guard otherwise.

## Criteria
F3, F5 (each, E and M'): adversarial mean contact torque on [9, 10) s <= 6.5 Nm in every run, and nominal |mean contact torque - 3| <= 0.3 Nm in every run.

## Predictions
Q5: R-a brings the adversarial steady contact torque to <= 6.5 Nm in every run; R-b does not.
Q6: F3 passes for E and M'. F5 passes for M' and fails for E.

## Decision rule
A candidate that passes all its criteria is proposed for the spec, subject to approval; if both pass for M', the latched guard F3 is preferred only if it also passes for E. If none passes, T6 stays open and M2 waits.
