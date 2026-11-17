# Verified values (single record)

## Divergence (open)

(none)

## Platform reproduction

Port check (attempt 3, iteration 300; 64 evaluation episodes from starts 0, 2, 3; deterministic, key 555; float64 on CPU), same code, checkpoint and package versions
(jax/jaxlib 0.12.0.1, numpy 2.5.3, Python 3.13.18) on two machines. Records: records/port_check_cloud.log (cloud), records/port_check_pc_baseline.json (this PC,
per-episode costs with machine, CPU flags and package versions; computed with the unmodified bundle code before the repository migration).

- Cloud (Xeon 2.1 GHz, 2 cores): mean cost 17.370263, 6.421145, 4.681267; identical to the evaluation recorded by l2_train.py (records/l2_attempt3/evals.json).
- This PC (Intel Core Ultra 9 285K; AVX2, FMA, AVX-VNNI, no AVX-512): mean cost 17.373252, 6.421073, 4.681267.
- Agreement per start (this PC vs cloud, relative difference per episode):
  - start 3: all 64 within 3.45e-12 (max).
  - start 2: 54 of 64 within 1e-12, 6 above 1e-6, max 4.21e-04.
  - start 0: 16 of 64 within 1e-12, 39 above 1e-6, 3 above 1e-2, max 1.16e-01 (episode 7: 10.276032 cloud, 11.465325 this PC).
- Diagnosis: machine-code-dependent floating-point rounding, amplified by discrete grasp and contact events (start 3 has the fewest such events and agrees to
  rounding level; start 0 includes the grasp). Deterministic within one machine: repeated runs, a single Eigen thread (--xla_cpu_multi_thread_eigen=false)
  and two pinned cores give bit-identical costs. The result moves with --xla_cpu_max_isa: SSE4_2 gives 17.374157, 6.421471, 4.681267 (AVX2 equals the default).
- Rule: the cloud records stay valid and are never overwritten by recomputation. New runs and diagnostics are produced on this PC, and comparisons are made
  only within one platform. Reproduction of the repository code is checked by `python script/port_check.py ref=pc` (EXACT only if all 192 episode costs are
  bit-identical to records/port_check_pc_baseline.json).
- Pool cache: data/pools/l2_pool_E0_6144.json is the pool built on the cloud (identical to the bundle's records/l2_pool_E0_6144.json). Rebuilt on this PC by
  the bundle code and by the repository code (identical to each other), 1486 of the 6144 entries differ from it, all of them budget episodes (1847 budget
  episodes; median relative difference 8.2e-12, maximum 4.02e-02), because E_task comes from L1 reference-brain rollouts (records/train_gate_pc.log). The
  cloud pool is kept and never regenerated on this PC; script/l2_train.py refuses to overwrite an existing pool file.

## Relative degree one: decrease inequality, point evaluation

- Script: `code/check_decrease.py` (seed 0). Log: `records/log_check_decrease.txt`.
- What: exact value of psi_g * d||w||/dt at random points with normalized error in [1 - m_star, 1), adversarial drift, feedforward and proposal; 2,000,000 samples per row.
- Level: point evaluation of a closed-form inequality (not a simulation).
- Common constants: gamma0 0.5, Fbar 0.6, ubar_ff 1, kappa0 1 (kappa in [1,10] for half of the samples), alpha 2, psi_low 0.05, psi_max 0.4, vbar 3, gamma_r(m) = gamma_psi(m) = min(1, 2m); rho0 = 0.4.

| m | ubar | gamma1 | m_star | max psi_g d||w||/dt | min(restoring - Psi(s)) |
|---|---|---|---|---|---|
| 1 | (2) | 2.0 | 0.03521 | -0.1412 | 5.276e-11 |
| 1 | (2) | 9.5 | 0.01715 | -0.2735 | 2.209e-10 |
| 2 | (2,3) | 2.0 | 0.03521 | -0.1390 | 6.226e-08 |
| 2 | (2,3) | 9.5 | 0.01715 | -0.2533 | 2.335e-07 |
| 3 | (2,3,8) | 2.0 | 0.03521 | -0.1417 | 4.536e-05 |
| 3 | (2,3,8) | 9.5 | 0.01715 | -0.2650 | 5.850e-05 |

- Componentwise clipping, Gamma = 5*[[1,0.9],[0.9,1]], ubar = (2,8), w ~ (1,-0.3): w^T Gamma clip(-s w) -> +15.9957; direction-preserving: -5.268 (exact arithmetic on one instance).

## Relative degree one: closed loop

- Script: `code/check_closed_loop.py` (seed 0). Record: `records/closed_loop.jsonl`; log of the figure case: `records/log_fig_yield_core.txt`.
- What: 64 plants, 2 inputs, C = [-1,1]^2, c0 = 0.2, 20 s each, RK4 with step 2e-4; proposals change every 0.25 s (random direction at full speed, tolerance from {0.05, 0.1, 0.2, 0.4}); drift adversarial in half of the cases, feedforward adversarial in half of the cases.
- Level: from-scratch simulation, used as a sanity check only.

| run | law | gamma1, ubar | left tolerance | max||w|| median | max||w|| worst | 1 - m_star | realized/proposed path |
|---|---|---|---|---|---|---|---|
| 0 | proposed | 2.0, (2,3) | 0/64 | 0.889278 | 0.93901 | 0.964787 | 0.344931 |
| 1 | no yielding to the error margin | 2.0, (2,3) | 64/64 | - | - | 0.964787 | - |
| 2 | proposed | 9.5, (2,8) | 0/64 | 0.839218 | 0.878179 | 0.982847 | 0.390263 |
| 3 | componentwise clipping | 9.5, (2,8) | 3/64 | 0.839331 | 1.01125 | 0.982847 | - |
| 4 | proposed, proposal speed 0.3, no adversarial signals | 2.0, (2,3) | 0/64 | 0.379195 | 0.815889 | 0.964787 | 0.917532 |

- Other values of runs 0 and 2: max gauge of u = 1.0; min output clearance 0.0504426 and 0.0535253; min tube clearance 9.25648e-15 and 5.12784e-15; time fraction with active saturation 0.507321 and 0.332276; mean error-margin yield factor 0.76045 and 0.859161.
- Run 4: largest initial normalized error 0.816833; mean error-margin yield factor 0.999782; time fraction with active saturation 0.000331875.
- Figure case (run 0, case 42): max ||w|| 0.93901, a priori bound 0.96479.

## Relative degree one, isotropic exact case: storage balance and yielding

- Script: `code/check_adapt_yield.py` (seeds 0 and 1). Records: `records/adapt_yield.jsonl`, `records/log_identity.txt`.
- Identity of the storage balance: chain-rule derivative along the vector field vs closed form at 200,000 random points; max relative difference 2.117e-14 (input saturated at 15% of the points, feedforward saturated at 78%). Level: point evaluation.
- Closed loop: 32 plants, 2 inputs, gamma in [0.5, 2], 7 features (exact), kappa = 4 psi_g^2, g_a = 0.5, ubar_ff = 1, ubar = (2,3), requested tolerance 0.1, periodic square proposal, no projection, 40 s, RK4 step 4e-4. Level: from-scratch simulation, sanity check only. Values are means over the first and the last 8 s.

| run | law | speed | error-margin yield factor | mean ||w|| | mean kappa||xi|| | time with any saturation | V start -> end |
|---|---|---|---|---|---|---|---|
| 0 | feedback only | 0.3 | 0.951 -> 0.944 | 0.418 -> 0.444 | 0.245 -> 0.241 | 0 -> 0 | - |
| 1 | adaptation with the factor lambda*lambda_ff | 0.3 | 1.000 -> 1.000 | 0.055 -> 0.036 | 0.031 -> 0.014 | 0.001 -> 0 | 1.886 -> 1.171 |
| 2 | adaptation without the factor | 0.3 | same as run 1 to the digits shown | | | | 1.886 -> 1.171 |
| 3 | feedback only | 1.5 | 0.616 -> 0.652 | 0.687 -> 0.635 | 0.656 -> 0.529 | 0 -> 0 | - |
| 4 | adaptation with the factor | 1.5 | 0.913 -> 0.926 | 0.316 -> 0.251 | 0.243 -> 0.157 | 0.648 -> 0.620 | 1.886 -> 74.9 |
| 5 | adaptation without the factor | 1.5 | 0.911 -> 0.912 | 0.321 -> 0.265 | 0.247 -> 0.177 | 0.658 -> 0.655 | 1.886 -> 4612 |

- Steps without saturation on which V minus the supply from tolerance contraction increased: 0 of 3,198,910 (run 1), 0 of 1,123,236 (run 4), 0 of 1,063,507 (run 5).
- Ratio of mean kappa||xi|| to mean mismatch ||u_ff - u_star|| in the feedback-only runs: 0.948 to 0.950 (run 0), 0.828 to 0.840 (run 3).
- max ||w||: 0.678 (run 0), 0.445 (runs 1, 2), 0.833 (run 3), 0.822 (run 4), 0.834 (run 5). B_r = 0.1438 for m_r = 0.5; no case of run 1 was below B_r at 40 s.

## 7-DOF arm: authority of the saturated input

- Script: `code/check_authority_panda.py` (JAX key 0; model `code/panda.py` from the earlier bundle). Record: `records/authority_panda.json`.
- What: 200 samples per setting, posture q0 +- 0.5 rad, payload up to 3 kg, limits (87,87,87,87,12,12,12) Nm, output eta = dq + k1 q, drift torque tau_d = C dq + g + friction - k1 M dq. Level: point evaluation; mu by convex QP on each face of the input box (Dinkelbach), rho_dir lower bound certified per sample, finite-gain rates are sampled maxima (not certified).

| k1 | max abs dq_i | (C2) pointwise > 0 | closed-form sufficient > 0 | mu > 1 | mu min | mu median | rho_dir certified min | rho_dir sampled min |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.000 | 0.985 | 1.000 | 1.3419 | 2.0550 | 1.2837 | 5.8618 |
| 2 | 0.5 | 0.000 | 0.965 | 1.000 | 1.2641 | 2.0810 | 0.9942 | 3.5638 |
| 5 | 0.5 | 0.000 | 0.940 | 1.000 | 1.3018 | 2.0784 | 1.0755 | 4.2122 |
| 2 | 1 | 0.000 | 0.935 | 1.000 | 1.2166 | 2.1157 | 0.7747 | 2.8485 |
| 5 | 1 | 0.000 | 0.900 | 1.000 | 1.0863 | 2.1892 | 0.4282 | 1.4594 |

- Eigenvalues of M over all samples: 0.1013 to 5.5499. Median of the pointwise (C2) value: -17.72 to -14.68.
- Worst sampled rate w'(F + Gamma sigma(u_ff - s w)) with ||u_ff|| = 50 Nm, per setting in the order above:
  - s = 100 Nm: 34.550, 39.276, 36.974, 39.895, 41.546
  - s = 300 Nm: 0.091, 0.470, -0.995, -0.072, 1.605
  - s = 1000 Nm: -4.798, -2.450, -3.280, -2.689, -0.341
  - s = 10000 Nm: -5.751, -3.455, -4.114, -2.832, -1.337

## Two-link arm with unknown inertia: storage balance

- Script: `code/check_mech_arm.py` (seeds 0, 1, 2). Records: `records/mech_arm.jsonl`, `records/mech_arm_points.json`.
- Identity of the storage balance at 200,000 random points: max relative difference 1.954e-14 (input saturated at 0.23 of the points, feedforward saturated at 0.21). Level: point evaluation.
- Authority over the region (20,000 samples): mu > 1 in 1.000, mu min 1.4242, mu median 2.8918; pointwise (C2) > 0 in 0.0015; eigenvalues of M 0.0316 to 2.5870.
- Closed loop: 32 arms, 40 s, RK4 step 5e-4, limits (60,25) Nm, k1 = 4, C_q = [-0.2,1.2] x [0.2,1.8], kappa = 15 psi_g^2, g_a = 40, ubar_ff = 50, requested tolerance 0.2, exact regressor features (7 parameters), no projection. Level: from-scratch simulation, sanity check only. Means over the first and last 8 s.

| law | error-margin yield factor | mean ||w|| | mean kappa||xi|| [Nm] | mean ||e_q|| [rad] | V_M start -> end | max ||w|| | min distance of q to the boundary of C_q | max gauge of u |
|---|---|---|---|---|---|---|---|---|
| none | 0.1876 -> 0.1284 | 0.9059 -> 0.9358 | 23.584 -> 23.255 | 0.0667 -> 0.0468 | 5.9448 -> 5.9720 | 0.9511 | 0.3710 | 0.5521 |
| adapt | 0.8912 -> 1.0000 | 0.3942 -> 0.0459 | 2.894 -> 0.139 | 0.0260 -> 0.0020 | 5.9448 -> 0.0463 | 0.8406 | 0.2151 | 0.8029 |

- Adaptive law: steps without saturation on which V_M increased: 0 of 2559958.

## Learning comparison (two-link arm; evolution strategies, 70 iterations, 3 seeds, 64 test tasks)

- Scripts: `code/rl_compare.py` (runs `run_rl.sh`, `run_rl2.sh`, `run_rl3.sh`), `code/rl_scripted.py`, `code/fig_rl_comparison.py`. Records: `records/rl_*.jsonl`, `records/rl2_*.jsonl`, `records/rl_summary.json`, `records/rl2_summary.json`, `records/rl_scripted.json`, `records/rl_dt_check.json`, `records/rl_reflex_dt.json`, `records/rl_authority.json`, `records/rl2_authority.json`.
- Level: from-scratch simulation (RK4, step 1e-3); learning results at one training budget. The RL runs were not repeated in a clean room (deterministic seeds; each run about 15 to 80 s).
- Pre-registration of round 2: `records/preregistration_round2.md`, sha256 748f22277a1e327a20ffdff69ab479a67b486705178085ed35fee41700f8f3f7.

| round | architecture | per-seed values | violations (train) |
|---|---|---|---|
| 1 | input policy with funnel filter | joint error 0.2977, 0.2889, 0.3074; return -0.1550, -0.1558, -0.1555; max normalized error 0.9077 | 0 / 30240 |
| 1 | proposed | joint error 0.5516, 0.5631, 0.5651; return -0.4015, -0.4088, -0.4152; max normalized error 0.9414 | 0 / 30240 |
| 1 | proposed without reports | joint error 0.5399, 0.5398, 0.5547; return -0.3966, -0.3972, -0.3971; max normalized error 0.9412 | 0 / 30240 |
| 1 | proposed with learned reflex | joint error 0.5727, 0.5967, 0.5745; return -0.4134, -0.4363, -0.4197; max normalized error 1.2677 | 0 / 30240 |
| 1 | proposed with learned reflex, without reports | joint error 0.5425, 0.5778, 0.5169; return -0.3733, -0.4166, -0.3763; max normalized error 1.3802 | 0 / 30240 |
| 2 | B | steady-state error 0.1358, 0.1390, 0.1752; interior 0.1112, 0.1024, 0.1547; near limit 0.1516, 0.1625, 0.1883; mean yield factor 0.302, 0.313, 0.314 | 0 / 30240 |
| 2 | F | steady-state error 0.1436, 0.1556, 0.1192; interior 0.1296, 0.1208, 0.0953; near limit 0.1526, 0.1779, 0.1344; mean yield factor 0.511, 0.291, 0.442 | 0 / 30240 |
| 2 | A | steady-state error 0.4481, 0.4222, 0.4229; interior 0.4028, 0.3844, 0.3842; near limit 0.4771, 0.4464, 0.4477; mean yield factor 0.515, 0.538, 0.523 | 0 / 30240 |
| 2 | C | steady-state error 0.4016, 0.4076, 0.4304; interior 0.3692, 0.3807, 0.3956; near limit 0.4224, 0.4249, 0.4528; mean yield factor 0.497, 0.436, 0.499 | 0 / 30240 |

- Round-2 predictions: {'P1': False, 'P2': True, 'P3': True} (P1 rejected). Test tasks: 25 with all targets away from the limits, 39 with a target next to a limit.
- Fixed proportional command (lower layer only): vbar 4, kc 15, ga 10, psip 0.05: error 0.6270, yield 0.2246, max w 0.9656; vbar 4, kc 15, ga 10, psip 0.4: error 0.5850, yield 0.3423, max w 0.9296; vbar 4, kc 60, ga 10, psip 0.05: error 0.5564, yield 0.3647, max w 0.9313; vbar 4, kc 60, ga 40, psip 0.05: error 0.4731, yield 0.5915, max w 0.9240; vbar 6, kc 15, ga 10, psip 0.05: error 0.5808, yield 0.2181, max w 0.9811; vbar 6, kc 60, ga 40, psip 0.05: error 0.4207, yield 0.5622, max w 0.9348; vbar 6, kc 60, ga 40, psip 0.2: error 0.4173, yield 0.5369, max w 0.7781; vbar 6, kc 60, ga 0, psip 0.05: error 0.5290, yield 0.3232, max w 1.4380.
- Coupled-input point check of the closest prior work: M^{-1} = 5[[1,0.9],[0.9,1]], taubar = (2,8): M^{-1} taubar = (46, 49), m = 6.125; tau = (-2, 8) gives (M^{-1} tau)_1 = +26 against the bound -12.25 used in the proof (exact arithmetic).

## Cause check (pre-registered; records/preregistration_causes.md, sha256 d158152c0828de51ddc07b9d9597f7b1b3f40d634f06d62b71807b2a7a190611)

- Scripts: `code/meta_features.py`, `code/lib_meta.py`, `code/run_fix.sh`, `code/fig_cause_check.py`, `code/bound_reference.py`. Records: `records/meta_features.json`, `records/meta_features.npz`, `records/rlfix_*.jsonl`, `records/rlfix_summary.json`, `records/bound_reference.json`. Level: from-scratch simulation (sanity level); the bound is a feasible point of a convex relaxation after 3,000 projected-gradient steps (an upper bound of the relaxed optimum).
- Feature fit (64 held-out tasks): relative residual 1.123e-04 (20 meta-learned features), 1.202e-02 (29 random features).

| repair | joint error per seed | change vs round 1 | supported | mean yield factor | violations |
|---|---|---|---|---|---|
| D1 looser tolerance | 0.5418, 0.6158, 0.5323 | -1.8%, +9.4%, -5.8% | False | 0.647 | 0 |
| D2 learned gain | 0.5306, 0.4882, 0.5373 | -3.8%, -13.3%, -4.9% | False | 0.649 | 0 |
| D3 target action | 0.5651, 0.5495, 0.5659 | +2.4%, -2.4%, +0.1% | False | 0.595 | 0 |
| D4 meta-learned features | 0.5154, 0.4786, 0.5385 | -6.6%, -15.0%, -4.7% | False | 0.889 | 0 |
| all four | 0.4566, 0.4547, 0.4707 | -17.2%, -19.2%, -16.7% | False | 0.844 | 0 |

- Combined repair closes half of the gap (<= 0.429 in each seed): False. Order of mean reductions: ['feat', 'gain', 'target', 'psi'].
- Kinematic relaxation, mean over test tasks: vbar4_psi0.4: 0.1904, vbar4_psi1: 0.1496, vbar6_psi0.4: 0.1575.

## R7 check (code/check_r7.py, records/r7_check.json)

- One-joint arm, RK4 step 1e-4, 5 s, 29 cases. Identity error of Theorem (R7)(a): max 1.78e-07.
- Torque limit above the gravity load (28 cases): positive debt 0, constraint layer or saturation active 0.000-0.498 of the time, no joint violation, min T 0.5000 (= T_min).
- Torque limit 15 Nm below the gravity load 17.27 Nm: cumulative debt 52.1, min T -16.31, joint violation (q_min -0.984).
- Pressing cases: 8 of 12 stayed at T_min without reaching the wall (gate deadlock, R9).

## R8 and R9 checks (code/check_r8r9.py, code/check_r8b.py; records/r8r9_check.json, records/r8b_check.json)

- A_none: 5.563--10.224
- A_cal: 4.800--10.218
- A_reach: 0/16
- A_minT: 0.500
- B_id: 6.5\times10^{-11}
- B_none: 5.902--11.454
- B_cal: 0.579--0.865
- B_reach_c: 3/4
- B_reach_n: 0/4
- B_err: 1.091--1.150
- B_wind: 12.314--12.449
- B_nwind: 3
- B_minT: 4.777

## Sensed freeze check (code/check_r8b.py, mode calib_in_contact_sensed)

- steady contact torque 0.605--4.150 Nm (target 3), peak 11.04 Nm; correct use peak 7.02 Nm.

## R1 check (code/check_r1.py, records/r1_check.json)

- two-link arm: (B3) sampled margin min 11.5030; 0 joint violations; max rho 0.9440 (gauge), max |w|_inf 0.9392 (componentwise).
- strongly coupled inertia: margin min -6.1243 (fraction positive 0.9745); weak limits: margin min -12.4058 (fraction positive 0.8276); both layers violated in both.

## M1 to M1d (code/m1_stage_a.py, m1b_fixes.py, m1c_guard.py, m1d_confirm.py; pre-registration hashes 2cd695a1, def20757, aa45e205, b492b6ae)

- T1a: 5.57e-08
- T6E: 6.238--10.448
- T6M: 7.040--11.620
- T8a: 0.3125
- T5bM: 0.0013--0.0064
- T5bE: 1.416--4.117
- windup: 39.864
- drain: 8.4084
- q1: 0.349
- Ra: 6.909
- Ran: 3
- T1d: 1.46e-09
- T4d: 0.0124
- T5d: 0.0206
- T5bd: 0.0090
- T6d: 5.971--6.002
- T8d: 1.0467
- rho: 1.0896
- pk1: 0.666--6.376
- pk4: 7.612--15.450

## M2 (code/m2_stage_a.py, analyze_m2.py, m2_diag.py; pre-registration hash 2cfbe45e)

- full_err: 0.342, 0.431, 0.397
- tp_err: 0.046, 0.055, 0.061
- mis2: 2.943
- rel1: -0.035
- nexp: 4
- d2e: 0.397
- d2g: 1.336
- d2m: 2.392
- checks: {"M0": true, "V1": true, "P1": false, "P2": true, "P3": true, "P4": true, "P5": false, "P6": false, "E_gap_seeds": 3}

## M2 audit (code/m2_trace.py, m2_trace_vel.py; records/m2_trace_audit.json; full-arm seeds 1 and 2, 64 evaluation tasks, reproduction of the recorded evaluation checked)

- prior-only mismatch at the wall: seed 1 [-1.546, 5.098], seed 2 [-2.768, 3.845] Nm; fraction outside [-3, 9]: 0 and 0.
- time input frozen at 1.5 s: seed 1 error 0.431 -> 0.392, seed 2 0.397 -> 0.862 Nm.
- gate factor always 1; minimum tank 17.23 J of 20 J; constraint activity 1.9% (seed 1), 0.6% (seed 2) of samples; T/T0 in 0.86-1.01.
- first contact median 0.485 s (seed 1), 0.774 s (seed 2).
- seed 2: spring at the 6 Nm cap 95.8% of contact time; mean calibration in contact 0.0017; mismatch at 1 s 3.087 Nm.
- seed 2 signed steady mismatch: -2.943 (as trained), -1.347 (velocity term off), -2.392 (calibration at rest only), -0.055 (both) Nm.
- out-of-distribution payload 2-4 kg: seed 1 error 0.381, mismatch 0.041; seed 2 error 1.963, mismatch 5.760 Nm.

## Stage B environment (code/stageb; records/stageb_dyn_check.json, stageb_env_check.json, stageb_contact_runs.json, stageb_design.json, barrier_storage_check.json; criteria hashes ccaed5dd, 9a1b3517, e1e222a1)

- b3_in: 22.52
- b3_ood: 12.69
- b3_in0: 35.82
- b3_ood0: 21.83
- r_in: -71.24
- r_ood: -107.14
- r_in0: 16.48
- r_ood0: -6.21
- ws_n: 85.7
- ws_h: 7.3
- ws_u: 3.2
- gr_in: 0.35, 0.30, 0.22
- gr_ood: 0.51, 0.47, 0.37
- gj: 0.138
- thr: 73{,}000
- f32q: $3.3\cdot10^{-5}$
- f32f: 0.0031
- hard: 0.296
- budg: 0.293
- ferr: 0.0052
- op2e5: $1.6\cdot10^{-6}$
- h2e4: $9.2\cdot10^{-7}$
- h2e5_4: $1.0\cdot10^{-4}$
- h2e5_8: $7.0\cdot10^{-6}$
- keig: 0.024--5.915
- ker: 0.217

## Stage B environment, first and second verification runs (console output; the result files were overwritten by the reruns; see the amendments in records/stageb_env_criteria.md)

- first run: V7 contact power-balance residual 8.5e-3 at 2e4 N/m with 4 substeps (spring-damper law); V14 peak normal force 522 N and audit residual 1.5e-2 (tip driven through the block side).
- second run: V7 residual 2.1e-3 at 2e4 N/m with 4 substeps and 6.7e-7 with 8 substeps (friction slope proportional to the normal force).

## Stage B environment after the re-grasp change (fifth run; criteria hashes ccaed5dd, 9a1b3517, e1e222a1, 6cc72312, 7876e1ff; values generated by code/stageb/write_envcheck_tex.py into records/stageb_spec_values.json)

- b3_in: 27.46
- b3_ood: 7.40
- b3_in0: 29.27
- b3_ood0: 18.03
- r_in: -70.39
- r_ood: -95.57
- r_in0: 12.02
- r_ood0: -4.68
- ws_n: 85.7
- ws_h: 7.3
- ws_u: 3.2
- gr_in: 0.37, 0.32, 0.28
- gr_ood: 0.50, 0.51, 0.46
- gj: 0.145
- thr: 60{,}000
- f32q: $5.7\cdot10^{-5}$
- f32f: 0.0023
- hard: 0.297
- budg: 0.311
- op2e5: $1.7\cdot10^{-6}$
- h2e4: $9.3\cdot10^{-7}$
- h2e5_4: $8.6\cdot10^{-5}$
- h2e5_8: $5.6\cdot10^{-6}$
- ker: 0.185
- v16f: 100
- dth: 0.4

## L0 and L0b (records/l0_results.json, l0b_results.json, l0_fig_data.npz; pre-registration hashes b3f16fc8, 384c4231)

- kb_err: {'0.25': '0.001', '0.5': '0.002', '1.0': '0.004', '2.0': '0.006'}
- kb_p90: {'0.25': '0.026', '0.5': '0.027', '1.0': '0.037', '2.0': '0.039'}
- tq_old: 0.142
- tq_ratio: 0.089
- minT_sign: -2.359
- minT_level: 1.0002
- rho_w: 2.46
- rho_wq: 0.825
- g_tc: 24.77
- g_ex: 1.30
- g_pk: 30.93
- c6a: 0.018
- c6a90: 0.069
- c6b: 0.019
- c6b90: 0.044
- n6b: 62
- gcls: 0.377
- gcls90: 0.599
- gobj: 0.111
- gobj90: 0.247
- s_lo: 0.183
- s_lo_g: 0.64
- s_lo_v: 1.20
- s_n: 0.043
- s_hi: 0.082
- n_lo: 0.051
- n_n: 0.018
- n_hi: 0.024
- n_g: 0.011

## Earlier values cited in the spec status table

- Produced in the previous session by `code/funnel_core_test.py` and `code/eism_test.py` (scalar plant); not recomputed here.

## Contact-permitting constraint (specification revision 17; merged 2026-11-17 from the cloud patch)

Sections "접촉을 허용하는 제약" and "접촉력 상한의 상수와 수치 확인" of docs/notes/spec.tex. Platform of all values in the four sections below: cloud container, CPU, JAX 0.12.0.1, numpy 2.5.3, float64; they are not recomputed on this PC (platform rule). Gate: the contact block of script/check_sync.py.

## Contact-permitting constraint: constants of the Stage B structure, by sampling, by sampling

- Script: `eval/verify/contact_bound_constants.py` (seed 1, 200000 samples per case). Record: `records/contact_bound_constants.json`. Level: sampled minima and maxima (estimates, not proofs).
- States on the gauge set {rho(w) <= 1, rho(w_q) <= 1}, half with rho(w) = 1, k_1 = 2, kept if the tip lies in [0.20, 0.90] x [-0.30, 0.30] m; contact normal n = (0, 1), contact point = tip.
- in_xi0: states 87088; min n^T Y n 0.1541; min mu at nu 0.5 0.0587; min mu at nu 0.8 -0.0789 (share nonpositive 0.00914); max tip speed 3.749 m/s; max a_in over the input box 243.2 m/s^2; max a_in under the saturated torque along J^T n at tip speed <= 0.5 m/s -60.5 m/s^2.
- in_xi1: states 87074; min n^T Y n 0.1208; min mu at nu 0.5 0.0812; min mu at nu 0.8 -0.0127 (share nonpositive 0.00003); max tip speed 3.548 m/s; max a_in over the input box 195.3 m/s^2; max a_in under the saturated torque along J^T n at tip speed <= 0.5 m/s -22.1 m/s^2.
- ood_xi0: states 87147; min n^T Y n 0.1332; min mu at nu 0.5 0.0240; min mu at nu 0.8 -0.1588 (share nonpositive 0.01044); max tip speed 3.803 m/s; max a_in over the input box 317.6 m/s^2; max a_in under the saturated torque along J^T n at tip speed <= 0.5 m/s -44.9 m/s^2.
- ood_xi1: states 86588; min n^T Y n 0.1061; min mu at nu 0.5 0.0824; min mu at nu 0.8 0.0379 (share nonpositive 0.00000); max tip speed 3.728 m/s; max a_in over the input box 161.6 m/s^2; max a_in under the saturated torque along J^T n at tip speed <= 0.5 m/s -11.1 m/s^2.
- Class constants in distribution (worse of the two load cases, nu <= 0.5): mu_low 0.0587 1/kg; tip speed 3.749 m/s; a_in 243.2 m/s^2; a_r 22.14 m/s^2. Median posture: mu 0.3469, a_r 57.57.
- Bound of the closure theorem for the present structure, k 2e4 N/m, delta 5 s/m: a/mu 4144.6 N, impact term 2188.8 N, bound 174389.6 N. Certificate of (B3) in the specification: 60 N.
- Speed limit [m/s] that keeps the bound at a force limit (formula evaluation; relief: delay 4 ms; zero inward drive: theorem (c) with a = 0; under the input-box drive: none in every case):
  - mu_low, k 2000: 60 N: zero drive 0.174, relief 0.528; 200 N: zero drive 0.376, relief 0.854.
  - mu_low, k 20000: 60 N: zero drive 0.075, relief 0.191; 200 N: zero drive 0.180, relief 0.342.
  - mu_low, k 200000: 60 N: zero drive 0.028, relief 0.048; 200 N: zero drive 0.078, relief 0.109.
  - mu_median, k 2000: 60 N: zero drive 0.310, relief 0.693; 200 N: zero drive 0.633, relief 1.148.
  - mu_median, k 20000: 60 N: zero drive 0.145, relief 0.237; 200 N: zero drive 0.320, relief 0.444.
  - mu_median, k 200000: 60 N: zero drive 0.061, relief 0.054; 200 N: zero drive 0.150, relief 0.130.

## Contact-permitting constraint: numerical checks of the lemma and the theorem

- Plan: `records/contact_bound_check_plan.md` (hash 72d4e1cd before the runs, a2693450 with the outcome). Script: `eval/verify/contact_bound_check.py` (seed 0). Record: `records/contact_bound_check.json`. Level: inequality checked on simulated trajectories (corroboration, not proof).
- C1 (one degree of freedom, 2000 cases per family): constant drive with delta = 0: depth ratio median 1.000000, max 1.000000; constant and bang-bang drive: max depth ratio 0.999634, max force ratio 0.998700; random drive: 0.967673, 0.943886.
- C2 (Stage B plant, 3000 trajectories): 13181 valid contact intervals, 0 with mu <= 0; max depth ratio 0.962396, max force ratio 0.936738; largest observed normal force 3822.2 N.

## Contact-permitting constraint: first-contact peak force of the learned policy

- Script: `eval/verify/contact_peak_record.py` (checkpoint `l2_attempt3/params_it0300.pkl` (cloud run), deterministic policy, 64 evaluation episodes per start, key 555). Record: `records/contact_peak_attempt3.json`. Mean costs equal the cloud records (17.370263, 6.421145, 4.681267).
- original start: contact in 43 of 64; median 14.66 N, 90% 32.11 N, max 57.07 N; share above 60 N 0.000.
- phase-2 start: contact in 60 of 64; median 14.24 N, 90% 30.35 N, max 39.88 N; share above 60 N 0.000.
- phase-3 start: contact in 63 of 64; median 16.19 N, 90% 54.95 N, max 106.92 N; share above 60 N 0.079.

## Contact-permitting constraint: enlarged contact class (plan `records/contact_class_plan.md`, first hash a7186461)

- Scripts: `eval/verify/contact_class_design.py` (D1-D4), `eval/verify/contact_authority_limit.py` (E0-E3). Records: `records/contact_class_design.json`, `records/contact_authority_limit.json`. Level: sampled (estimates, not proofs); D4 is formula evaluation.
- Bound of the closure theorem for the present structure: 174389.6 N at k_bar = 2e4 N/m, 241142.7 N at 2e5 N/m.
- D1, margin of (B3) with contact [rad/s^2], in / out of distribution: 60 N: 27.46 / 7.40; 100 N: -125.95 / -176.49.
- Input box (120, 60, 25) N m: margin without contact 27.86 / 18.03 rad/s^2; F* 79.20 / 71.79 N; F* with the force only in the contact region 79.20 / 73.43 N.
- Input box (150, 75, 40) N m: margin without contact 40.53 / 27.57 rad/s^2; F* 100.59 / 101.04 N; F* with the force only in the contact region 100.59 / 101.04 N.
- Input box (180, 90, 50) N m: margin without contact 53.20 / 37.11 rad/s^2; F* 120.07 / 119.81 N; F* with the force only in the contact region 121.48 / 119.81 N.
- Input box (240, 120, 60) N m: margin without contact 78.53 / 56.19 rad/s^2; F* 157.89 / 157.37 N; F* with the force only in the contact region 160.96 / 157.37 N.
- D2, static force capacity of the input box at the contact postures: min 100.0 N, median 138.3 N.
- D3, friction 0.8, in distribution, no object: theta 1% 0.0035, 5% 0.1004, median 0.3937 1/kg; share with theta <= 0: 0.914%; max kinetic energy 9.361 J, with tip speed <= 0.5 m/s 2.576 J.
- D3, friction 0.8, in distribution, object attached: theta 1% 0.0908, 5% 0.1480, median 0.2993 1/kg; share with theta <= 0: 0.003%; max kinetic energy 16.852 J, with tip speed <= 0.5 m/s 2.454 J.
- D4, speed limit [m/s] at k_bar = 20000 N/m, class constants at friction 0.5 (force limit 60, 100, 200 N; t_d = 4 ms, 2 ms each): 0.191, 0.216, 0.246, 0.273, 0.342, 0.370; with the 5% quantile of theta at friction 0.8, 60 N, 4 ms: 0.193.
- D4, speed limit [m/s] at k_bar = 200000 N/m, class constants at friction 0.5 (force limit 60, 100, 200 N; t_d = 4 ms, 2 ms each): 0.048, 0.067, 0.069, 0.091, 0.109, 0.134; with the 5% quantile of theta at friction 0.8, 60 N, 4 ms: 0.049.
- D4, speed limit [m/s] at k_bar = 1e+06 N/m, class constants at friction 0.5 (force limit 60, 100, 200 N; t_d = 4 ms, 2 ms each): 0.013, 0.022, 0.021, 0.033, 0.037, 0.054; with the 5% quantile of theta at friction 0.8, 60 N, 4 ms: 0.013.
- D4, stored energy f^2 / (2 k_bar) at 60 N: 0.09, 0.009, 0.0018 J at 2e4, 2e5, 1e6 N/m.
