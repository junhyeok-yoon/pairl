# Pre-registration: numerical checks of the contact force bound (note_contact)

Written 2026-11-17 (UTC), before any run of the checks C1, C2 and C4 below. Platform: cloud container (CPU), JAX 0.12.0.1, numpy 2.5.3, float64.

## What is checked

The note `tex/note_contact.tex` states, for a flat compliant surface with normal force f_n = k l max(0, 1 + delta l_dot) (l: penetration depth) and
friction |f_t| <= nu f_n:

- Lemma (scalar comparison). If l_ddot <= a - c l at the times with l_dot > 0, then V := (l_dot^+)^2/2 + c l^2/2 - a l satisfies
  V(t) <= max{V(t0), 0}, hence l(t) <= (a + sqrt(a^2 + 2 c V0^+)) / c.
- Theorem (contact force bound). On a contact interval of the robot, with mu(q) := n^T Lam^{-1} n - nu |P_t Lam^{-1} n| >= mu_low > 0,
  Lam^{-1} := J_c M^{-1} J_c^T, and the inward drive acceleration a_in <= a_bar at the times with l_dot > 0:
  l(t) <= l_bar := (a_bar + sqrt(a_bar^2 + 2 c V0^+)) / c with c := mu_low k, and f_n(t) <= k l_bar (1 + delta sup l_dot^+).

## Exploration done before this plan (disclosed)

Sampling of the Stage B constants (not a test of the statements): on the contact region (tip in [0.20, 0.90] x [-0.30, 0.30] m) of the joint box,
mu(q) at nu = 0.8 was nonpositive at about 1% of the sampled postures without the object; the sampled maxima of the tip speed on the gauge set and of
a_in under the input box were about 3.65 m/s and 160-320 m/s^2. The checks C1, C2 and C4 have not been run.

## C1. Scalar lemma, one degree of freedom (numpy, RK4, dt = 1e-5 s, horizon 0.5 s or until the contact ends)

Mass m ~ U[0.5, 10] kg, k log-uniform on [2e3, 2e5] N/m, delta ~ U[0, 5] s/m, drive bound F ~ U[0, 100] N, onset speed v0 ~ U[0.01, 1] m/s,
2000 cases per family, seed 0. Dynamics in contact: m l_ddot = u - k l max(0, 1 + delta l_dot), l(0) = 0, l_dot(0) = v0. Drive families:
(i) constant u = F with delta = 0; (ii) constant u = F; (iii) bang-bang u = F sign(l_dot); (iv) piecewise constant, uniform on [-F, F], held 2 ms.
Bound: a = F/m, c = k/m, V0 = v0^2/2. Ratios r_l := max l / l_bar and r_f := max f_n / (k l_bar (1 + delta max l_dot^+)).

- Pass: max r_l <= 1.001 and max r_f <= 1.001 over all families.
- Tightness (reported, with criterion): in family (i) the median of r_l is >= 0.999.

## C2. Theorem on the Stage B plant (JAX, RK4, dt = 1e-4 s, horizon 1.2 s)

3000 trajectories, seed 0. Plant of `model.py`: in-distribution body for 2000, out-of-distribution body for 1000; object attached (xi = 1) for half.
Surface: the block of `model.contact` with half-width 10 m (only the top face acts), top height ye ~ U[-0.30, 0.30] m, k log-uniform on
[2e3, 2e5] N/m, delta ~ U[0.5, 5] s/m, nu ~ U[0.2, 0.8]. Initial posture: uniform on the joint box scaled by 0.9 about its center, kept if the tip is
0.03-0.25 m above the surface with x in [0.2, 0.9] m; at rest. Torque (held over each step, clipped to the input box):
- family A (1500): tau = Kp (q_t - q) - Kd q_dot + g(q) + dither, q_t := q0 + J^+ (p_t - p0), p_t = (p0_x + U[-0.1, 0.1], ye - U[0, 0.05]),
  Kp = U[0.3, 1] x (400, 200, 80), Kd = U[0.3, 1.5] x (30, 12, 2), dither uniform on +-U[0, 10] N m per joint, held 30 ms;
- family B (1500): tau = g(q) - Kd q_dot + amp x dither x u_bar + J^T (0, -F_b), amp ~ U[0.05, 0.3], F_b ~ U[0, 80] N, Kd as in A.
At every node: signed depth l_s = ye - p_y, l_dot = -v_y, f_n, mu(q) with the trajectory's nu, and a_in := -n^T (J q_ddot0 + J_dot q_dot) with q_ddot0
the acceleration without the contact force, evaluated with the torque of the previous and of the next step (maximum of the two).
A contact interval is a maximal run of nodes with l_s > 0 (it also ends at the horizon). Per interval: mu_low := min mu, a_bar := max{0, max a_in over
the nodes with l_dot > 0}, V0 from the first node of the interval, l_bar and the force bound as in the theorem with the realized sup of l_dot^+.
An interval is valid if mu_low > 0.

- Pass: over the valid intervals, max r_l <= 1.001 and max r_f <= 1.001; and the number of valid intervals is >= 1000.
- Reported: number of intervals, share with mu_low <= 0, quantiles of r_l and r_f, and for the intervals with mu_low <= 0 the largest observed f_n.

## C3. Constants of the Stage B structure (sampling, reported, no criterion)

200000 samples per case (in/out of distribution body x object attached or not), seed 1, states on the gauge set {rho(w) <= 1, rho(w_q) <= 1}
(half of them with rho(w) = 1), restricted to the contact region above. Reported: min of n^T Lam^{-1} n; min of nu_crit := n^T Lam^{-1} n / |t^T Lam^{-1} n|;
mu at nu = 0.5 and 0.8 (min, share <= 0, share < 0.05); sampled max of the tip speed |J q_dot|; sampled max of a_in over the input box;
the inward acceleration under a saturated torque along J^T n (relief), its max over all states and over the states with |J q_dot| <= 0.5 m/s;
max of |J|. From these: the bound of the theorem for the present structure and the onset speed that keeps the bound at 60 N and 200 N for
k in {2e3, 2e4, 2e5} N/m (formula evaluation).

## C4. First-contact peak force of the learned policy (descriptive, no criterion)

Checkpoint `records/l2_attempt3/params_it0300.pkl` (cloud records), deterministic policy, the 64 evaluation episodes from the three starts, as in
`port_check.py`. Reported per start: number of episodes with contact, median and max of `peak_first` (largest normal force within 0.3 s of the first
contact), share above 60 N.

## If a check fails

A failure of C1 or C2 means that the statement or the implementation is wrong. The cause is located by a minimal experiment before any change, the
outcome is appended here, and the note does not state the result as proven until the cause is resolved.

## Outcome (2026-11-17 UTC, cloud container, CPU)

Records: `contact_bound_check.json`, `contact_bound_check_c2.npz`, `contact_bound_check_c2.log`, `contact_bound_constants.json`, `contact_peak_attempt3.json`.

- C1: pass. Largest ratio over the four families 0.9999999999999996 (max r_l; max r_f 0.9999999999999994). Family (i): median r_l 0.9999999924671727
  (criterion >= 0.999). Families (ii) and (iii): max r_l 0.9996344576140304, max r_f 0.998699801700337. Family (iv): max r_l 0.9676725027319578,
  max r_f 0.9438856945803449.
- C2: pass. 3000 trajectories, all finite; 13181 valid contact intervals on 2823 trajectories (family A 8901, family B 4280); 0 intervals with
  mu_low <= 0. r_l: median 0.5400260323393121, 90% 0.824825300991253, 99% 0.9270670323425527, max 0.9623957386880435. r_f: median 0.3928401554156043,
  90% 0.7278540362767302, 99% 0.8800194121475584, max 0.9367375951962514. Largest observed f_n 3822.2158366348735 N. Run time 193 s.
- C3 (reported): in distribution, nu = 0.5: min mu 0.05867825276777758 (no object) and 0.08124087819511916 (object attached); nu = 0.8: min mu
  -0.07891105383679725 and -0.012712657920394332, share of nonpositive mu 0.009140180047767775 and 3.445345338447757e-05. Sampled maxima:
  tip speed 3.7491788729960027 m/s, a_in over the input box 243.19822903873668 m/s^2. Bound of the theorem for the present structure at
  k = 2e4 N/m (nu <= 0.5): 174389.5931615719 N. Inward acceleration under the saturated relief torque, states with tip speed <= 0.5 m/s: at most
  -22.135659114587845 m/s^2 (in distribution, worst of the two load cases).
- C4 (reported): mean costs 17.37026293589561, 6.4211448063727, 4.681266902310917 (equal to the cloud records). peak_first, starts 0, 2, 3:
  contact in 43, 60, 63 of 64 episodes; median 14.659292772592734, 14.236578460639425, 16.187837413759947 N; max 57.0721659783917,
  39.875976206695704, 106.92021057533775 N; share above 60 N: 0, 0, 0.07936507936507936 (5 of 63).

Deviation from the plan: none. Note on C2: no sampled interval had mu_low <= 0, so the behavior outside (F1) was not exercised.
