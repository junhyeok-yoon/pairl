# Pre-registration: constants for the enlarged contact class (specification, section "접촉을 허용하는 제약")

Written 2026-11-17 (UTC), before any run of D1-D3 below. Platform: cloud container (CPU), float64. Decisions of the user that motivate the plan:
the contact force limit is raised above 60 N, the stiffness class is extended to 2e5 N/m (1e6 N/m acceptable), friction up to 0.8 stays in the class.

## D1. Authority condition (B3) of the joint box theorem at larger force limits (criterion)

`design_study.b3_samples(c=(0.9,-1.3,-0.5), h=(0.9,1.1,1.2), ub=(120,60,25), k1=2, n=40000, seed=1, ood, fbar=F, nu=0.8)` for
F in {60, 100, 150, 200, 300} N and ood in {False, True}. Margin := -max(val with contact) [rad/s^2].
- Reproduction: at F = 60 the margins equal the specification's 27.46 (in distribution) and 7.40 (out of distribution) to two decimals.
- Criterion: (B3) holds on the samples at F iff both margins are > 0. Reported: the margins, and the largest F of the list at which it holds in
  distribution and at which it holds for both.

## D2. Static force capacity of the input box (reported)

On the sampled contact postures of D3: f_stat(q) := 1 / max_i(|(J^T n)_i| / u_bar_i), the largest normal force whose joint torque J^T n f lies in
the input box. Reported: min, 5% quantile, median (in distribution, both load cases pooled).

## D3. Constants at friction 0.8 and energy constants (reported)

Sampling as in C3 of `contact_bound_check_plan.md` (200000 states per case on the gauge set, seed 1, tip in [0.20,0.90] x [-0.30,0.30] m), in
distribution, object attached or not. Reported per case: quantiles (min, 1%, 5%, 50%) of theta(q) := n^T Y n - 0.8 |t^T Y n| with Y := J M^{-1} J^T,
share with theta <= 0 and with theta < 0.05; max of the kinetic energy and of lambda_max(M); for the states with tip speed <= 0.5 m/s the max of the
kinetic energy and the max of the inward acceleration under the saturated relief torque restricted to theta >= the 5% quantile.

## D4. Formula evaluation (reported)

Speed limit of the contact point that keeps the bound of the corollary (speed limit and delayed relief) at f_lim in {60, 100, 200} N for
k_bar in {2e4, 2e5, 1e6} N/m and t_d in {0.002, 0.004} s, delta_bar = 5 s/m, with (i) the class constants at friction 0.5 of
`contact_bound_constants.json` (theta_low, a_r) and (ii) the 5% quantile of theta at friction 0.8 from D3 with the relief constant of D3.
Energy form: allowed stored energy f_lim^2 / (2 k_bar).

## If D1 fails at F = 60 to reproduce

The cause is located before any other number is used.

## Outcome of D1-D4 (2026-11-17 UTC, cloud container, CPU; records `contact_class_design.json`, `contact_class_design.log`)

- D1: reproduction holds (60 N: 27.457969792301355 in distribution, 7.397297737160855 out of distribution). (B3) fails on the samples at every
  larger force of the list: 100 N: -125.95038015700084, -176.48704101325438; 150 N: -472.47192361558996, -593.8461957734226;
  200 N: -850.3825838125434, -1024.4314104925606; 300 N: -1637.9525822528717, -1885.6018399308368. Largest force of the list at which it holds: 60 N.
- D2: f_stat min 100.00000000249689 N, 5% quantile 101.15901650218586 N, median 138.2535480082596 N.
- D3 (friction 0.8, in distribution): no object: theta min -0.0789, 1% 0.0035, 5% 0.1004, median 0.3937, share <= 0 0.0091, share < 0.05 0.0245,
  max kinetic energy 9.361 J (2.5764 J at tip speed <= 0.5 m/s), max lambda_max(M) 1.9122 kg m^2; object attached: theta min -0.0127, 1% 0.0908,
  5% 0.148, median 0.2993, share <= 0 3.4e-05, share < 0.05 0.0018, max kinetic energy 16.8518 J (2.4539 J), max lambda_max(M) 4.0173 kg m^2.
- D4: in `contact_class_design.json` (keys `speed`, `energy`).
Deviation from the plan: none.

## Addendum (written after the outcome of D1, before running E1-E3): where the authority condition stops holding

Motivation: D1 fails at 100 N, and the list has no force between 60 N and 100 N. The quantities below are reported, without criterion.
The acceleration is affine in the contact force, so with val0_i the contact-free value of sample i and s_i := max(0, g_i . M^-1 J^T (nu, 1),
g_i . M^-1 J^T (-nu, 1)) the margin is -max_i (val0_i + F s_i), and the largest force at which (B3) holds on the samples is
F* := min over the samples with s_i > 0 of (-val0_i / s_i). Same samples as D1 (n = 40000, seed 1, k1 = 2, nu = 0.8).
- E0 (check of the reimplementation): the margin at F = 60 N equals D1 to 1e-9.
- E1: F* in and out of distribution, force applied at every sampled posture (as in the specification).
- E2: F* with the force applied only at the samples whose tip lies in the contact region [0.20,0.90] x [-0.30,0.30] m.
- E3: E1 and E2 for the input boxes (150,75,40), (180,90,50), (240,120,60) N m, together with the contact-free margin.

## Outcome of the addendum (2026-11-17 UTC; records `contact_authority_limit.json`, `contact_authority_limit.log`)

- E0: holds (margin at 60 N: 27.45797 in distribution, 7.397298 out of distribution).
- E1 (input box (120,60,25) N m): F* = 79.196339 N in distribution, 71.794381 N out of distribution.
- E2: 79.196339 N and 73.425385 N.
- E3: (150,75,40): contact-free margin 40.529867 / 27.568811, F* 100.586419 / 101.036035 (region: 100.586419 / 101.036035);
  (180,90,50): 53.197782 / 37.109484, F* 120.07129 / 119.814757 (region: 121.482543 / 119.814757);
  (240,120,60): 78.533613 / 56.190832, F* 157.893102 / 157.372203 (region: 160.959119 / 157.372203).
Deviation from the addendum: none.
