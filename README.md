# pairl

**pairl** (Passive Adaptive Impedance Reinforcement Learning): an RL policy with an adaptive lower layer inside it; constraint satisfaction and passivity hold
by structure, independently of learning.

The policy has two coupled layers: a learned RL policy (task-level intent: goal and stiffness) and an adaptive compensation layer (virtual impedance and an
adaptive memory updated from the mismatch measured at rest). The guarantees come from parts that are not learned: an energy tank with its gate, a constraint
layer, direction-preserving saturation and a contact guard. The current stage (Stage B) is a planar 3-joint arm with a gripper, an object and a block.

State of the work, rules and next steps: [docs/20261115_handoff.md](docs/20261115_handoff.md). Working rules: [CLAUDE.md](CLAUDE.md).
Specification (single source of the research): `docs/notes/spec.tex` (rendered `docs/notes/spec.pdf`).

## Installation

```
conda create -p _local/envs/py313 python=3.13.18     # base interpreter (this PC has no python3-venv); git-ignored
_local/envs/py313/bin/python -m venv .venv && . .venv/bin/activate
pip install -e ".[gpu]"                               # or: pip install -e .   (CPU only)
```

Pinned versions (as in the cloud records): jax/jaxlib 0.12.0.1, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, Pillow 12.3.0, sympy 1.14.0, mujoco 4.0.1
(mujoco only for the reference checks in `refs.py`). The `gpu` extra installs `jax[cuda13]` (NVIDIA driver 580 or later).
Documents: XeLaTeX (`texlive-xetex texlive-latex-extra texlive-pictures`), `fonts-noto-cjk`, `fonts-dejavu`, `poppler-utils`.

All code runs in float64 (`jax_enable_x64` is the first line of every entry point).

## Structure

```
src/pairl/            importable package
  paths.py            single source of repository paths (root: $PAIRL_ROOT or the package location)
  stageb/             plant, environment, control structure: params, model, kin_np, env, controller, harness, refs
  learn/              learning and evaluation core: l0, l1, l2, l2_hand_contact_core
  tools/              record and document helpers: lib_tables, lib_meta, write_envcheck_tex, design_study
script/               entry points: l2_train.py, port_check.py, report_checkpoint.sh, build_docs.sh, check_sync.py
eval/                 programs that measure policies and the environment without training
  diagnostics/        l2_dissect, l2_signal_check, l2_restart_behavior, l2_hand_*, l2_ledger_check, l2_traj_check, ..., L0/L1 runs
  verify/             verify_dyn, verify_env, f32_run, cost_audit, check_barrier_storage, contact_runs, design_record
  figures/            fig_*.py, anim_rollout.py
data/runs/<timestamp>/<run>/   training runs grouped by experiment (one group per plan and code state; manifest.json per group;
                      data/runs/INDEX.md lists the groups); train.log, hist.json, evals.json, dissect_*.json; checkpoints only on disk, three tracked
data/pools/           episode pool (l2_pool_E0_6144.json)
records/              research records (pre-registrations, verified_values.md, diagnostics); see records/README.md
rsc/                  static resources (robot description files of later stages)
docs/notes/           research notes (tex sources and rendered PDFs)
docs/figures/         figures used by the notes; rollouts/ holds the GIFs (not tracked)
docs/guidelines/      writing and verification rules
archive/              legacy_code/ (M1, M2, F1 stages, not rerun), earlier_bundle/, bundle_docs/
_local/               git-ignored local material: bundle/ (bundle extract, read-only), baseline/, envs/py313, tmp/ (temp files)
```

## Main commands (from the repository root)

```
python script/l2_train.py group=new <run_name> N_IT [N_EP] [POOL] [WARM_CKPT] [cost=4] [restart=0.5] [eval_restart=1] [act=task] plan=records/<plan>.md purpose="..."   # training (needs approval)
JAX_PLATFORMS=cpu python script/port_check.py ref=pc                     # reproduction on this PC: bit-identical to records/port_check_pc_baseline.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python script/port_check.py ref=pc bench=1 sizes=64,256,1024,4096   # GPU: difference to the CPU baseline, throughput
python eval/diagnostics/l2_dissect.py data/runs/20261115134229/l2_attempt3 300 cost=4   # component dissection of a checkpoint
python eval/diagnostics/l2_signal_check.py records/l2_signal_check.json data/runs/20261115134229/l2_attempt3:300 n=512 start=2 cost=4
sh script/report_checkpoint.sh l2_attempt3 300 l2_pilotT_T fig_l2_attempt3.py signal=1   # checkpoint report (dissection, curves, GIFs)
bash script/build_docs.sh spec                                           # render docs/notes/spec.tex (gates must all be 0)
python script/check_sync.py                                              # consistency gate: must print "sync findings: 0"
sh script/safe_run.sh mem=20G python script/parallel_run.py eval/diagnostics/l2_dissect.py data/runs/20261115134229/l2_attempt3 300 cost=4   # parallel CPU rollouts
```

Every job that can take much memory runs under `script/safe_run.sh` (its own systemd unit with a hard memory cap and no swap, lower CPU and IO weight,
GPU without preallocation), so an oversized job is killed alone. `script/parallel_run.py` runs a non-training diagnostic or evaluation with its rollouts in
worker processes (one per core, single-threaded, chunks of 2-64 episodes; bit-identical to one process, `records/parallel_check.log`); the number of workers
is limited to the memory budget.

On the GPU, `XLA_PYTHON_CLIENT_PREALLOCATE=false` is needed while the desktop holds part of the VRAM (otherwise cuBLAS fails to allocate).
Without `ref=pc`, `port_check.py` compares with the cloud records; on this PC that comparison reports DIFFERENT for a known platform reason
(`records/verified_values.md`, `## Platform reproduction`).
