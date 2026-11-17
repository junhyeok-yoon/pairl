# Pre-registration: M1d confirmatory run of the configuration proposed by M1b and M1c

Fixed before any M1d run. Configuration: corrected sensed calibration (M'), contact cap (F5), gate with power margin (G1).
Same system, plants, seed, brains, groups and tests as M1 (hash 2cd695a1). Code: code/m1d_confirm.py. Results: records/m1d_confirm.json.

## Criteria (all from M1, now including T5b as a criterion)
T1 <= 1e-6; T2 true torque within the bound; T3 zero joint-box samples outside; T4 ratio <= 0.1; T5 <= 0.3 Nm; T5b steady |mismatch| <= 0.3 Nm (nominal runs);
T6 adversarial steady contact torque <= 6.5 Nm; T8 min T >= T_min - 0.05 J. T7 exploratory.

## Prediction
Q7: the configuration passes T1-T6, T5b and T8 in every run.

## Decision rule
Q7 holds: the configuration is proposed for the spec and M2 starts after approval. Any failure: that test stays open and M2 waits.
