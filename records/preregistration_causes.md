# Pre-registration, cause check (written before any repaired run)

Date (UTC): 2026-11-12 22:23. Code: code/rl_compare.py and code/lib_meta.py, combined sha256 prefix 368ef17e376ea31d. Feature map: records/meta_features.npz (offline meta-learning on class samples; relative residual of per-task least squares 1.12e-4 for 20 learned features vs 1.20e-2 for the 29 random features).

## Diagnoses and the repair that tests each (proposed architecture, round-1 setting, everything else unchanged)
- D1 tolerance tube ties the motion to a slow reference: repair psi, upper tolerance 0.4 -> 1.0 (directional authority still holds: minimum drift-scale margin 1.210 at the new velocity bound 4.0 rad/s).
- D2 the feedback gain is outside learning: repair gain, the task policy chooses the gain scale in [15, 60].
- D3 the rate action is hard to learn and its effect is erased by yielding: repair target, the task policy outputs a target position; the reference is steered proportionally toward it.
- D5 the adaptive representation is weak: repair feat, meta-learned feature map with its mean task weights as the initial fast weights.
- All four repairs together.

## Reference values (round 1, same seeds, tasks and budget)
Proposed: mean joint error 0.5516, 0.5631, 0.5651. Input policy with funnel filter: 0.2977, 0.2889, 0.3074.

## Predictions
- A diagnosis is supported if its repair alone lowers the mean joint error by at least 20 percent relative to the proposed architecture in each of the 3 seeds; otherwise it is not supported.
- D1 and D5 give the two largest reductions (seed means).
- The combined repair closes at least half of the gap: error <= 0.429 in each seed.
- No joint-limit violation for any repaired variant.

## Use of the result (fixed now)
- Combined repair closes at least half of the gap: the hierarchy with a loose tolerance stays a candidate for spec v3 (tolerance as a delegation level).
- Otherwise: spec v3 drops the tolerance-tube hierarchy (adaptive hand plus a boundary-local constraint layer).
- All per-seed values are reported whatever the outcome.
