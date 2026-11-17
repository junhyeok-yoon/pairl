# Pre-registration: M2 coupled learning on Stage A (first round)

Fixed before any M2 run (seeds 0, 1, 2). Code: code/m2_stage_a.py (this version). Results: records/m2_<arm>_<seed>.json.

## System and structure
Stage A as in M1 with the approved modifications: measured-mismatch calibration allowed in contact, spring torque limited to 6 Nm in sensed contact, gate power margin 0.5 W, full-power tank T(0) = 20 J, gauge constraint layer on [-0.4, 1.0] rad.
Tasks: payload U[0, 2] kg, wall stiffness log-uniform [500, 5000] Nm/rad, damping U[1, 5], friction U[0.05, 0.3], disturbance U[-0.5, 0.5] Nm, wall position U[0.5, 0.6] rad, residual sensor bias U[-0.05, 0.05] Nm, noise sd 0.05 Nm.
Episode 3 s from rest at 0.3 rad with an empty memory; control 2 ms; plant RK4 with 4 substeps; brain every 50 ms.
Target: contact torque 3 Nm. Training cost per step on [1, 3) s: ((|tau_c| - 3)/3)^2 in contact, 1 + (distance to the wall)/0.3 otherwise; plus 0.5 max(0, peak - 6)/6.
Brain: network 11-32-32-3 (goal, stiffness, calibration intensity). Hand: feature network 1-16-8 plus a constant, prior, quadratic memory, fixed damping 3.

## Learning
Coupled evolution strategies on shared rollouts: 16 antithetic pairs x 8 tasks per iteration, 300 iterations, perturbation 0.05 (brain) and 0.03 (hand), Adam step 0.02 (brain) and 0.03 (hand).
Brain updated by the task return; hand updated by its role loss mean (u_ad - u_star)^2 (privileged).
Engineering pilot (seed 100, full arm only, not used for any claim): with step 0.02 and 200 iterations, median steady error 0.282 Nm, mismatch median 0.024 Nm.

## Arms
full; no_memory (no online adaptation, learned prior only); no_ledger (brain does not see T, measured mismatch, calibration, gate factor, constraint activity; still sees the sensed torque);
no_calibration (calibration intensity fixed to 0); single_return (hand updated by the task return); torque_policy (network outputs torque every control step from t, q, q_dot, sensed torque, through tank, gate, constraint layer and saturation, no hand).

## Evaluation
64 held-out tasks (seed 999) with the final mean parameters. Steady error = |mean contact torque on [2, 3) s - 3|. Mismatch = mean |u_ad - u_star| on [2, 3) s.
RSI = |corr(payload, mean spring torque on [2, 3) s)| over the 64 tasks. Ledger reliance = median error with the ledger inputs zeroed minus median error.
PD (calibration effort while moving, |q_dot| > 0.1) and impact peak are descriptive.

## Checks and predictions
M0 manipulation check: full has median steady error <= 0.5 Nm in every seed; otherwise every comparison below is inconclusive.
V1: zero joint violations and positive minimum tank level in every evaluation rollout of every arm.
P1 (online adaptation): median error full < no_memory in every seed, and mismatch median of full <= 0.3 Nm in every seed.
P2 (calibration channel): median error full < no_calibration in every seed.
P3 (hand-internal ledger): median error full < no_ledger in at least 2 of 3 seeds.
P4 (role credit): mismatch median full < single_return in every seed.
P5 (role separation): RSI(full) <= 0.3 in every seed and RSI(no_memory) >= 0.6 in every seed.
P6 (ledger reliance): ledger reliance of full >= 0.3 Nm in every seed.
E (expressivity, descriptive with decision): median error of torque_policy versus full.

## Decision rules
M0 fails: double the iterations under a new pre-registration; no claim.
P1 or P2 fails: the adaptation or calibration claim is not supported in Stage A.
P3 or P6 fails: the claim that the brain uses hand-internal information is not supported.
P4 or P5 fails: role learning is not achieved; revise the training devices before Stage B.
E: if torque_policy has a median error lower than full by more than 0.1 Nm in at least 2 seeds, consider the learned reflex torque inside the tank.
