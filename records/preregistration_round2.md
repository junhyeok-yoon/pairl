# Pre-registration, round 2 (written before any round-2 training run)

Date (UTC): 2026-11-12 19:58. Code: code/rl_compare.py, sha256 prefix eb9958f2d830b454. Launched with RL_ROUND=2 RL_UBAR=80,35.

## Why this setting (a priori)
- Round 1 showed that the proposed lower layer limits transient tracking (heavy yielding), and that the reports gave no benefit.
- The adaptive-control argument concerns steady-state accuracy under parametric uncertainty: a memoryless input policy has no integral action, so under an unknown gravity load its steady-state error is bounded below by load mismatch divided by its effective stiffness; the adaptive core adapts its feedforward online and needs no such stiffness.
- A fair baseline must be allowed integral action: architecture F is the round-1 input policy that also observes the integral of the joint error since the last target switch.
- Reports: the interface yields automatically, so a proposal is made feasible without the policy knowing the current capability. The reports can only help when the best proposal depends on that capability in a way yielding does not reproduce. No such mechanism is designed in, so no benefit is predicted.

## Setting (identical to round 1 except)
- Payload range 0 to 4 kg (round 1: 0 to 2); torque limits (80, 35) Nm (round 1: (60, 25)), chosen so that the directional authority holds over the operating region (drift-scale margin minimum 1.224 at |dq_i| <= 2.8 rad/s; 1.045 at 4.2).
- Architectures: A proposed; C proposed without reports; B input policy with funnel filter; F = B plus the error integral in the observation.
- Training: evolution strategies, 70 iterations, 48 perturbations x 3 tasks per iteration, seeds 0, 1, 2; test on 64 fixed tasks (seed 777).

## Primary metric
Steady-state joint error: mean of ||q - q*|| over the last 0.5 s of each 1.5 s target window, on the 64 test tasks, for the final policy of each run.

## Predictions
- P1 (primary): A has a lower steady-state error than F in each of the 3 seeds, and than B in each of the 3 seeds.
- P2: A is not lower than C in all 3 seeds (the reports give no consistent benefit).
- P3: A and C have zero joint-limit violations in training and test (consequence of the theorem).

## Decision rule (fixed now)
- P1 holds: continue, with the claim restricted to steady-state accuracy of the adaptive lower layer under wide parametric uncertainty at equal training budget. The claim that two-way information exchange helps learning is dropped unless P2 fails.
- P1 fails: stop this design and write it up as a negative result.
- All per-seed values are reported whatever the outcome; no other metric is used for the decision.
