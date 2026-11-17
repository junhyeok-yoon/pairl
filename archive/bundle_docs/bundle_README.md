# Adaptive-philosophy RL policy: specification v3 (revision 16), Stage B learning code and records

Start with HANDOFF.md (state of the work, rules, environment, next steps).

## Structure

```
tex/      spec.tex        specification v3 (revision 16): hand, full-power tank and contact guard, constraint layer, brain, coupled learning, role verification, implementation-learning-verification plan, Stage B experiments L0-L2 (kotex)
          spec_v2.tex     specification v2 (revision 3), stopped design, kept as the record (kotex)
          fig_architecture.tex, fig_architecture_v2.tex   TikZ sources of the two architecture diagrams
          note_learning.tex learning comparison and pre-registered test: negative result, design stopped (kotex)
          note_core.tex   derivation note: constraint guarantee for relative degree one (kotex)
docs/     Research_Principles_and_Guidelines.md (the user's writing and verification rules; see HANDOFF.md for the sections that apply),
          Research_Note_Example.tex (note preamble)
archive/  earlier_bundle/ files of the earlier directions, not used by the current work
code/     lib_yield_core.py     closed-form pieces shared by the checks and the figure
          m2_trace.py, m2_trace_vel.py   M2 audit: ledger content, guarantee activity, memory-bias paths, out-of-distribution preview
          stageb/         Stage B environment (planar 3R arm, object, block): model.py, env.py, params.py, kin_np.py;
                          references refs.py (SymPy, MuJoCo); tests verify_dyn.py, verify_env.py, contact_runs.py, f32_run.py;
                          design_study.py ((B3) sampling, workspace margins), design_record.py, check_barrier_storage.py, harness.py (privileged test controllers, incl. re-grasp),
                          fig_stageb_env.py, write_envcheck_tex.py (regenerates spec section 3.2 from the records)
                          controller.py (v3 controller for Stage B), l0.py (rollouts with scripted test brains), l0_run.py, l0b_run.py, fig_l0.py
                          (pre-registrations records/preregistration_l0*.md; results records/l0_results.json, l0b_results.json)
                          l1.py (memoryless test brains, task metrics, budgets), l1_pilot.py (engineering pilot, not pre-registered; records/l1_pilot.json), fig_l1_pilot.py
                          l1_pilot_v2.py (pilot on the revised task; records/l1_pilot_v2.log), l1_run.py (pre-registered L1: records/preregistration_l1.md, .hash;
                          results records/l1_results.json, l1_summary.json), fig_l1.py; env.py sampler version 2 = revised task (version 1 kept for earlier records)
                          l2.py (coupled learning: PPO brain, learned hand features with the replayed role loss, diagnostics), l2_train.py (one-seed run),
                          l2_dissect.py (component dissection of a checkpoint, incl. exploring-policy statistics and the channel test: L1 reference brain driven with the
                          learned hand), l2_release_mech.py (held vs dropped cost rate, critic value with xi flipped, open probability -> records/l2_release_mechanism.json),
                          fig_l2_attempt1.py (-> figures/l2_attempt1.png), fig_l2_pilots.py (-> figures/l2_pilots.png); attempt plan records/l2_attempt1_plan.md (+ .hash: diagnosis,
                          pilots P1-P3), runs records/l2_attempt1/ (it 0-70), l2_attempt1b/ (71-100), pilots l2_pilot_cost2/, l2_pilot_cost2_grip/, l2_pilot_v3/ (101-130).
                          Options of l2_train.py: cost=1|2|3 (env.COST_V; 3 = spec revision 14, the default; attempt 1 used 1), grip_reset=1 (pilots only),
                          role=0|1 (role-loss weighting of revision 14, default 1). Attempt 2: records/l2_attempt2_plan.md (+ .hash), run records/l2_attempt2/,
                          fig_l2_attempt2.py (-> figures/l2_attempt2.png), anim_rollout.py (-> figures/l2_attempt2_rollout_itXXXX.gif: four evaluation episodes,
                          arm, commanded goal, object, block, contact force, rest gate, ledger load). Checks of attempt 1 and the pilots: cost_audit.py
                          (-> records/cost_audit.json), l2_hand_share.py, l2_face_check.py, l2_carry_check.py (-> records/l2_hand_share.json, l2_face_check.json,
                          l2_carry_check.json). L1 under revision 15: l1_run.py (rev-13 and rev-14 records kept as records/l1_*_rev13.json, l1_*_rev14.json); environment check V5-V18: verify_env.py
                          Attempt 2 history: records/l2_attempt2a (initial goal sd 0.04, stopped at 60), l2_attempt2b (sd 0.1, stopped at 50), l2_attempt2 (resumed
                          from l2_attempt2b at 50); decisive experiments records/l2_diag_relzero (E1), l2_diag_cost13 (E2), l2_diag_init1_rev14 (E3),
                          l2_diag_init1_rev13 (E4), l2_diag_init1_rev14_vnorm (E5); checks l2_goal_check.py, l2_adv_check.py, l2_value_check.py
                          (-> records/l2_goal_check.json, l2_adv_check.json, l2_value_check.json). Options of l2_train.py added: sd_goal=, rel_zero=1, sd_reset=0, vnorm=1.
                          fig_l2_attempt2.py RUN OUT L1_SUMMARY pre=EARLIER_RUN prepends the run a resumed run started from.
                          Dissection of attempt 2 at iteration 300 (plan, amendments 6 and 7): l2_signal_check.py (learning signal by phase on exploring rollouts of
                          2048 pool episodes, held coverage near the start grasp point and p*, exploration scale -> records/l2_signal_check.json; the first computation on
                          256 episodes kept as records/l2_signal_check_n256.json), l2_hand_check.py (transfer residual of each checkpoint and of the weight-averaged hand
                          -> records/l2_hand_check.json), fig_l2_signal.py (-> figures/l2_signal.png).
                          Revision 15 (env.COST_V 4, default; 3 = revision 14): press term bounded by 1. Environment check V5-V18 (run 7), cost_audit.py with the
                          top-face check, L1 rerun (rev-14 records kept as records/l1_*_rev14.json, stageb_env_check_rev14.json, cost_audit_rev14.json).
                          Pilot R (records/l2_pilotR_plan.md, .hash; runs records/l2_pilotR_A, l2_pilotR_B): env.restart builds the restart states; options of
                          l2_train.py: restart=BETA (share of each batch from the restart states), eval_restart=1 (evaluation also from both restart states),
                          vwarm=K (critic-only warm-up). l2_overpress_check.py (pressing and releasing from the phase-3 restart -> records/l2_overpress_check.json),
                          fig_l2_pilotR.py (-> figures/l2_pilotR.png); anim_rollout.py start=2|3. Smoke tests and checks of the preparation:
                          records/l2_pilotR_smoke_B_nowarm.log, l2_pilotR_smoke_B_warm.log, l2_role_defect_check.log.
                          Pilot T (records/l2_pilotT_plan.md, .hash; runs records/l2_pilotT_J, l2_pilotT_T): l2.goal_of (action mode 'joint' or 'task': box
                          projection of q + J_nom(q)^+ dp + n v_N), l2.set_mode (mode of a checkpoint, key 'act'); l2_train.py option act=joint|task; the diagnostics
                          (l2_dissect.py, anim_rollout.py, l2_signal_check.py, l2_restart_behavior.py) take the mode from the checkpoint. l2_restart_behavior.py
                          (behavior from the restart states -> records/l2_restart_behavior.json); fig_l2_pilotR.py labelA= labelB=.
                          Attempt 3 (records/l2_attempt3_plan.md, .hash; run records/l2_attempt3, continuing l2_pilotT_T from iteration 100): checkpoints
                          store the optimizer states (key 'opt', restored on a warm start unless opt_reset=1); fig_l2_attempt3.py (-> figures/l2_attempt3.png);
                          l2_dissect.py adds the exploration scale by action dimension. Post-hoc hand and ledger analyses: l2_hand_check.py, l2_hand_replay.py
                          (role loss of several hands on one brain's rollouts), l2_hand_contact.py and l2_hand_fresh.py (offline hand training with role-loss
                          variants from l2_hand_contact_core.py; reused or fresh batches), l2_transfer_exposure.py (role weight by distance from the last rest
                          posture), l2_ledger_check.py (ledger load as a tip mass and the L1 class rule) -> records/l2_hand_*.json, l2_transfer_exposure.json,
                          l2_ledger_check.json.
                          Status at the end of attempt 3 (rollouts only): l2_traj_check.py (reaching, grasp outcome, pressing by class, release with the
                          deterministic and the sampled gripper -> records/l2_traj_check.json), l2_never_rest.py (episodes by grasp outcome, rest share by
                          phase -> records/l2_never_rest.json), fig_l2_status.py (-> figures/l2_status.png).
                          port_check.py (reproduction of the evaluation at iteration 300 and rollout throughput; first check on a new machine),
                          report_checkpoint.sh (dissection, curves, animations and signal checks of a checkpoint of a run).
                          run from code/stageb: python3 verify_dyn.py && python3 verify_env.py   (criteria: records/stageb_env_criteria.md)
          check_decrease.py     point evaluation of the decrease inequality (Table 1 of the note)
          check_closed_loop.py  closed-loop sanity check, five runs (Table 2 of the note)
          check_adapt_yield.py  storage identity (point evaluation) and yielding with and without adaptation (Table 3 of the note)
          check_authority_panda.py  authority of the saturated input for the 7-DOF arm (Table 4 of the note; needs JAX)
          panda.py              7-DOF model of the earlier bundle (used by check_authority_panda.py)
          check_mech_arm.py     two-link arm with unknown inertia: storage identity, authority, closed loop (Table 5 of the note)
          lib_tables.py         table rows of the notes generated from the records
          rl_compare.py         learning comparison (JAX): architectures, evolution strategies, metrics, authority check
          run_rl.sh, run_rl2.sh, run_rl3.sh   round 1, round 1 with learned reflex, round 2 (RL_ROUND=2 RL_UBAR=80,35)
          rl_scripted.py        lower layer under a fixed proportional command (Table 2 of the learning note)
          fig_rl_comparison.py  aggregation of the RL records -> figures/rl_comparison.png, records/rl_summary.json, rl2_summary.json
          meta_features.py, lib_meta.py  offline meta-learning of the core's feature map -> records/meta_features.npz
          run_fix.sh            pre-registered cause check (RL_FIX=psi|gain|feat|target and all four)
          fig_cause_check.py    cause-check summary -> figures/cause_check.png, records/rlfix_summary.json
          bound_reference.py    convex kinematic relaxation of the motions the tolerance hierarchy permits
          fig_yield_core.py     figure script -> figures/yield_core.png
          check_sync.py         consistency gate: cited numbers vs records, retired strings, open divergences
          build_all.py          runs all of the above
          funnel_core_test.py, eism_test.py, ledger.py, check_online.py
                                earlier checks cited in the status table of the spec
                                (ledger.py and check_online.py need the code and weights of the earlier bundle)
figures/  yield_core.png
records/  verified_values.md    single record of verified values and their conditions
          closed_loop.jsonl, log_*.txt
spec.pdf, spec_v2.pdf, note_core.pdf, note_learning.pdf   rendered with XeLaTeX and Noto Serif CJK KR (./build_tex.sh <name>, output in build/)
build_tex.sh                    render script (swaps kotex for fontspec, three passes, gate counts)
```

## Figure, script and place of use

| Figure | Script | Used in |
|---|---|---|
| `figures/rl_comparison.png` | `code/fig_rl_comparison.py` (reads `records/rl_*.jsonl`, `records/rl2_*.jsonl`) | `tex/note_learning.tex`, Section 4 |
| `figures/cause_check.png` | `code/fig_cause_check.py` (reads `records/rlfix_*.jsonl`) | `tex/note_learning.tex`, Section 5 |
| `figures/architecture.pdf`, `figures/architecture.png` | `tex/fig_architecture.tex` (TikZ, `pdflatex`; PNG by `pdftoppm -r 200`) | `tex/spec.tex`, Section 1 |
| `figures/architecture_v2.pdf`, `figures/architecture_v2.png` | `tex/fig_architecture_v2.tex` | `tex/spec_v2.tex`, Section 5 |
| `figures/yield_core.png` | `code/fig_yield_core.py` (uses `check_closed_loop.py`, `lib_yield_core.py`) | `tex/note_core.tex`, Section 4 |
| Overview and architecture diagrams | TikZ inside the `.tex` sources | `tex/note_core.tex` Section 1; `tex/spec.tex` Section 1 |

## Reproduction

```
cd code && python3 build_all.py        # about 17 minutes; or run the scripts one by one:
python3 check_decrease.py
python3 check_closed_loop.py 0 1 2 && python3 check_closed_loop.py 3 4
python3 check_adapt_yield.py 0 1 2 && python3 check_adapt_yield.py 3 4 5
python3 check_authority_panda.py && python3 check_mech_arm.py
python3 fig_yield_core.py && python3 check_sync.py
cd .. && ./build_tex.sh note_core && ./build_tex.sh note_learning && ./build_tex.sh spec && ./build_tex.sh spec_v2     # PDFs in build/
```

Dependencies: Python 3 with numpy, matplotlib, jax (for the 7-DOF model only); XeLaTeX (texlive-xetex, texlive-latex-extra), Noto Serif CJK KR and DejaVu Serif fonts, poppler-utils. The delivered `.tex` files use `kotex` and compile as they are.

Gates before delivery: `build_tex.sh` reports errors 0, undefined 0, overfull 0, missing characters 0; `check_sync.py` reports 0 findings.

## Status

- Learning comparison (`note_learning.tex`): the proposed architecture was safe but worse than an input policy with a funnel filter at equal budget; the pre-registered round-2 prediction was rejected; by the pre-registered rule the design is stopped.
- RL runs: `cd code && ./run_rl.sh && ./run_rl2.sh && ./run_rl3.sh` (about 25 minutes on one core), then `python3 rl_scripted.py && python3 fig_rl_comparison.py`. The pre-registration in `records/` must not be edited.

- Proved in `note_core.tex`: tolerance, state constraint, input constraint, global solution and a priori bounds for relative degree one, for every admissible learned content. The condition is sufficient and conservative.
- Proved in `note_core.tex` for the isotropic exact case and for mechanical systems with exact representation: storage balance, decrease without saturation, convergence of the error and vanishing of the error-margin yielding.
- Proved in `note_core.tex`: the constraint guarantee under directional authority, and its application to mechanical systems through the output dq + k1 q with convex position constraints. The directional authority holds on all samples of the 7-DOF model; the condition of the first theorem holds on none.
- Checked numerically only as a sanity check: see `records/verified_values.md`.
- Not covered: relative degree two, exact convergence, progress of the realized command, sampled data, reinforcement-learning experiments.
