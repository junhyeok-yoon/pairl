# Pre-registration: L0b, fixes after the L0 run (hash b3f16fc8)

Fixed before the L0b runs. Same environment, controller constants (k_b = 2.0 as chosen by the L0 rule), brains, episodes and seeds as L0 unless stated. Code: code/stageb/l0b_run.py. Results: records/l0b_results.json.

## L0 outcome (records/l0_results.json)
Passed: L0-3(a) (no violations for every k_b), L0-5, L0-6(a)(b)(c), L0-8 law N. k_b = 2.0 chosen by the rule (median tip error 0.0062 m).
Failed: L0-1 (tank error 0.142 J > 0.1 J), L0-2(a) (min T down to -2.36 J). Prediction "spec law fails at K_n" rejected (median 0.043); the spec law failed at k_lo instead (growth 0.64).
L0-7 invalid: a test error recorded the mismatch at the grasp posture (the rest indicator was still on when the move began).

## Diagnosis (exploratory, records/l0_results.json and console)
- L0-1: the tank error is the quantization effect sum_k tau_k^T (e_{k+1} - e_k), |e_i| <= Q/2; by summation by parts it is at most (Q/2)(TV(tau) + |tau_end|_1 + |u_bar|_1).
  On the 64 in-distribution runs the error was 0.008 to 0.070 of this bound. The 0.1 J value had no derivation.
- L0-2: the gate's exception (alpha = 1 when q_dot^T tau_h <= -p_m) leaks energy in discrete time: the hand switches on against a small falling speed and does positive work within the period.
  With a gate on the tank level only, alpha = clip((T - T_min)/Delta T, 0, 1), the minimum tank level was >= 1.000 J in all 128 runs (sign exception: down to -2.359 J).

## Changes
C1 (design, candidate, approval needed): gate on the tank level only; p_m removed.
C2 (criterion of L0-1(a), with the original result kept): |T - T_true| <= (Q/2)(TV(tau) + |tau_end|_1 + |u_bar|_1) in every run.
C3 (test fix of L0-7): the mismatch at the target posture is recorded at the first step with the rest indicator on after the arm has moved faster than 0.3 rad/s following the hold;
   the number of memory updates between leaving and that step is reported.

## Tests (all with C1)
L0b-1 = L0-1 with (a) replaced by C2; (b)-(d) unchanged.
L0b-2 = L0-2 with (a) min T >= 0 in every run, and (a') min T >= T_min - max_t U_b - 0.05 J in every run.
L0b-3 = L0-3(a) for k_b = 2.0; L0b-5 = L0-5; L0b-6 = L0-6; L0b-8 = L0-8 law N criteria (the spec law cells are rerun for the record).
L0b-7 = L0-7 with C3. Prediction (unchanged): median ratio for F-obj <= 0.3 and below that of F-class.

## Decision rule
If L0b-1, L0b-2(a)(a'), L0b-3, L0b-5, L0b-6 and L0b-8 pass, propose for the specification: the normalized memory update (lam_s = 5, v_r = 0.3), the tank-level gate, and the constants of the L0 pre-registration with k_b = 2.0 (approval needed); then L1.
A failure is diagnosed and fixed under a new pre-registration.
