# records

Research records: pre-registrations (`*_plan.md`, `preregistration_*.md` with `*.hash` / `*.sha256`), `verified_values.md` (single record of verified values,
divergence ledger, platform reproduction), criteria documents, diagnostic results (JSON, logs), records of the stages M1 to L1, and the port checks
(`port_check_cloud.log`, `port_check_pc_baseline.json`).

## Origin and layout

The files came from `20261115_v3_brain_hand_full_bundle.zip` and are unchanged, except `verified_values.md` (section `## Platform reproduction` added) and the
new files `README.md`, `port_check_pc_baseline.json`, `migration_map.tsv` (bundle path -> repository path of every migrated file) and `runs_grouping_map.tsv`
(data/runs/<run> -> data/runs/<timestamp>/<run>, 2026-11-16).

| bundle | repository |
|---|---|
| `code/stageb/` modules imported by other files | `src/pairl/stageb/`, `src/pairl/learn/`, `src/pairl/tools/` |
| `code/lib_tables.py`, `code/lib_meta.py` | `src/pairl/tools/` |
| `code/stageb/l2_train.py`, `port_check.py`, `report_checkpoint.sh`, `code/check_sync.py`, `build_tex.sh` | `script/` (`build_tex.sh` -> `build_docs.sh`) |
| other scripts of `code/stageb/` | `eval/diagnostics/`, `eval/verify/`, `eval/figures/` |
| rest of `code/` (M1, M2, F1 stages) | `archive/legacy_code/` |
| `records/<run>/` | `data/runs/<run>/`, then grouped as `data/runs/<timestamp>/<run>/` (2026-11-16; `runs_grouping_map.tsv`, `data/runs/INDEX.md`) |
| `records/l2_pool_E0_6144.json` | `data/pools/` |
| rest of `records/` | `records/` |
| `tex/*.tex`, root `*.pdf` | `docs/notes/` |
| `figures/*.png`, `figures/architecture*.pdf` | `docs/figures/` |
| `figures/*.gif` | `docs/figures/rollouts/` (not tracked) |
| `docs/*` | `docs/guidelines/` |
| `README.md`, `HANDOFF.md`, `CLAUDE.md` of the bundle | `archive/bundle_docs/bundle_README.md`, `bundle_HANDOFF.md`, `bundle_CLAUDE.md` (provenance only; their instructions are not current) |

## Commands inside the bundle-era records

The pre-registrations must not be edited (the hash chain in `*.hash` would break). Their commands and paths refer to the bundle layout: they were run from
`code/stageb`, run directories were `../../records/<run>`, code paths were `code/stageb/<file>`. In this repository the same command runs from the repository
root with the new locations, for example

```
python3 l2_train.py ../../records/l2_attempt3 200 128 6144 ../../records/l2_pilotT_T/params_it0100.pkl restart=0.5 eval_restart=1    (bundle)
python script/l2_train.py data/runs/20261115134229/l2_attempt3 200 128 6144 data/runs/20261115113808/l2_pilotT_T/params_it0100.pkl restart=0.5 eval_restart=1       (repository)
```

Keys of the diagnostic JSON files are the basenames of the run directories (for example `l2_attempt3:300`) and did not change.

## Platform

The cloud records were produced on a 2-core Xeon. This PC reproduces them to rounding level only where no discrete grasp or contact event differs; see
`verified_values.md`, `## Platform reproduction`. Cloud records stay valid and are never overwritten by recomputation; new results are compared within one platform.
