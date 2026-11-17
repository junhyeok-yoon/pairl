# L1 pre-registration: task validity with fixed brains (Stage B, task version 2)

Written 2026-11-14 before any confirmatory run. Specification revision 13, which applies the decisions taken after the L1 pilot:
the brain observes neither the realized goal q_d nor the realized stiffness K; budget episodes use E0 = Delta T + rho E_task with rho uniform on [0.6, 1];
the press force target is chosen by the payload class (f* = f_heavy if m_obj > 0.85 kg, else f_light; in-distribution payload uniform on [0.2, 0.65] U [1.05, 1.5] kg).

Code: code/stageb/l1.py (brains, rollout, metrics), code/stageb/l1_run.py (this protocol, including the statistics and verdicts below),
controller.py (constants fixed in L0), env.py (sampler version 2, environment re-verified under criteria amendment 5, hash cf0b8e1e).
Results: records/l1_results.json (per-episode metrics), records/l1_summary.json (statistics and verdicts).

## Engineering pilots (not used for any claim; seeds 3000-3999)
- L1 pilot on task version 1: records/l1_pilot.json (spec section L1 pilot).
- Check on task version 2 (code/stageb/l1_pilot_v2.py, records/l1_pilot_v2.log; 64 in-distribution episodes from seed 3000, 64 out-of-distribution bodies from seed 3100):
  proportional task correction k_c = 0, 1, 2 gave mean cost 2.491, 2.489, 2.784, median / 90th-percentile hold error 0.4 / 11.8, 2.0 / 7.5, 5.5 / 19.6 mm, and grasp success 0.11, 0.34, 0.83 on the out-of-distribution bodies.
  Selection rule fixed before looking at confirmatory data: the lowest in-distribution mean cost, ties within 0.01 broken by out-of-distribution grasp success. Chosen k_c = 1.
  Class accuracy with the ledger 1.000, without it 0.547; budget runs (every episode a budget episode): gate active 0.209 of the time.

## Brains
All brains are static maps of o_k = (q, q_dot, tau_ext, chi, ledger). The code passes them neither q_d nor K, and the context they receive contains neither f* nor m_obj.
Nominal postures are computed from the context (p_o, p*) before the episode.
- Reference brain B_ref: goal posture of least gauge for p_o (before the grasp) and p* (after it and in phase 4), plus the proportional correction k_c J+ sat_0.05(p_t - p_hat) with k_c = 1;
  carry right after the grasp; press: before contact is sensed (|tau_ext| <= tau_th) the goal leads the measured posture by J+ (k_x (p*_x - p_hat_x), -0.15 v) with v = 0.3 m/s above p*_y - 0.045 m and 0.15 m/s below;
  after it q + J+ (k_x (p*_x - p_hat_x), 0) - K_n^-1 J^T (0, f_hat) with k_x = 0.5; f_hat = f_heavy if the vertical tip load of the learned load in the ledger exceeds 0.85 * 9.81 N, else f_light;
  K^p = K_n = (200, 100, 40) N m/rad; psi = 1 in phases 1-3, 0 in phase 4; without the object in phases 2-3 it returns to the grasp posture with psi = 1; goals clipped to the joint box.
- B_noled: B_ref with f_hat = (1/a + 1/b) / (1/a^2 + 1/b^2), a = f_light, b = f_heavy (the constant that minimizes the expected squared relative error for equally likely classes).
Hand variants: v3 hand with the L0 constants; memory off (lambda_s = 0); memory off and K = k_hi = (400, 200, 80) N m/rad.

## Episode sets (task version 2)
- ID: 256 episodes from seed 4000 (default sampler, about 30% hard), E0 = E_nom.
- BUD: the ID episodes with E0 = Delta T + rho E_task for every episode, rho from the sampler and E_task from the ID run of B_ref with the v3 hand on the same episode and noise key.
- HEAVY: 128 episodes from seed 4300 with m_obj uniform on [1.5, 2.5] kg (seed 4301).
- BODY: 128 episodes from seed 4500 with the out-of-distribution body of Table 2 (lengths, masses and payload).
- BODY_MASS: 128 episodes from seed 4700, nominal lengths, masses times [0.7, 0.9] U [1.1, 1.3] and m_obj uniform on [1.5, 2.5] kg (seed 4701).
- BODY_LEN: the BODY episodes with nominal masses (lengths and payload out of distribution).
- STIFF: 128 episodes from seed 4900 with block stiffness up to 2e5 N/m.
One noise key per set, split over its episodes (ID and BUD share it; BODY and BODY_LEN share it), so the arms on one set are paired.

## Metrics (Definition Stage B task (l), on the true state)
Grasp success at the end of phase 1; hold error = mean of |p - p*| over the last 40% of phase 2; relative force error = mean of |f_n - f*| / f* over the last 40% of phase 3;
class decision correct if the mean f_n over that window is closer to f* than to the other target (over episodes with a grasp); first-contact peak of f_n within 0.3 s;
fractions of control steps with constraint activity (|tau_b| > 0), gate activity (alpha < 1), saturation (lambda < 1) and guard; joint violations; non-finite states; mean episode cost;
tank-bound margin min_t (T - T_min + max_{s <= t} U_b(q(s))), on episodes without saturation while T < T_min.

## Tests and criteria
L1-a (the task is solvable by the architecture), ID, B_ref with the v3 hand:
- a1 grasp success >= 0.95;
- a2 class accuracy >= 0.95;
- a3 median relative force error <= 0.05 and its 90th percentile <= 0.20;
- a4 median hold error <= 0.005 m and its 90th percentile <= 0.02 m;
- a5 no joint violation and no non-finite state in any L1 run.
L1-b (the hand memory is necessary), ID, B_ref with the memory off, and with the memory off and K = k_hi:
- b1 mean cost >= 2 times that of B_ref with the v3 hand, for both;
- b2 median relative force error >= 0.25, for both.
L1-c (the ledger is necessary for the class decision), ID, v3 hand:
- c1 median relative force error of B_noled >= 3 times that of B_ref;
- c2 press-phase cost of B_ref below that of B_noled in >= 80% of the episodes where both grasped;
- c3 class accuracy of B_noled <= 0.65.
L1-d (the guarantee devices operate), B_ref with the v3 hand:
- d1 mean fraction of time with constraint activity in the hard ID episodes >= 0.30;
- d2 in BUD, mean fraction of time with gate activity >= 0.05, and at least half of the episodes with some gate activity;
- d4 in BUD, tank-bound margin >= -0.01 J (Theorem full-power tank (b); tolerance for the tank on measured angles) on every episode without saturation below T_min (their number reported).
Reported without criteria: guard and saturation fractions; constraint activity in normal episodes; all metrics on HEAVY, BODY, BODY_MASS, BODY_LEN, STIFF
(they fix the in-distribution to out-of-distribution multiples for (P1) in L2); costs of BUD against ID.
Expectations written before the run (not criteria): grasp success on BODY and BODY_LEN well below ID (k_c = 1: 0.34 in the pilot), BODY_MASS close to HEAVY, class accuracy near 1 on the out-of-distribution sets (all heavy).

## Decision rules
- All criteria pass: L2 is pre-registered with the L1 values as references.
- a1-a4 fail: the task or the reference brain is not adequate; diagnose and propose changes before L2 (approval of the user).
- b fails: the task does not need the hand memory; propose task changes (approval).
- c fails: the class decision does not need the ledger or the estimate is not good enough; propose changes (approval).
- d1 or d2 fail: the devices are not exercised; propose range changes (approval). a5 or d4 fail: implementation or theory error; stop and fix first.
- No criterion is changed after the run. Amendments, if any, are appended with their reasons and do not replace the original verdicts.

Hash: first 8 hex digits of the SHA-256 of this file in records/preregistration_l1.hash, written before the run.
