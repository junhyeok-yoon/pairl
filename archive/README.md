# archive

Kept for provenance only; nothing here is part of the current work, and none of it is rerun.

- `legacy_code/`: the scripts of the earlier stages (M1, M2, F1: `m1_*`, `m2_*`, `rl_*`, `check_*`, `panda.py`, `ledger.py`, `run_*.sh`, `fig_*`, `build_all.py`, ...),
  copied unchanged from the bundle's `code/`. They keep the bundle's flat imports and relative paths (for example `from lib_meta import ...`, now
  `src/pairl/tools/lib_meta.py`; `lib_yield_core.py` and `panda.py` sit next to them; `../records/`, `../tex/`, `../figures/`) and are not runnable as is.
  Their results are in `records/` and are cited by the notes.
- `earlier_bundle/`: files of the directions before v3, as they were in the bundle (`archive/earlier_bundle/`); its `.pkl` weights are not tracked.
- `bundle_docs/`: `bundle_README.md`, `bundle_HANDOFF.md`, `bundle_CLAUDE.md`, the bundle's own README, handoff and rules. They describe the bundle layout and an
  earlier state; their instructions are not current. The current documents are `README.md`, `CLAUDE.md` and `docs/20261115_handoff.md`.
