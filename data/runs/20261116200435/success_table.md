# Success table: 20261116200435

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (l2_pilotG_B:320) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 | previous best: l2_pilotG_B:320 | l2_pilotLd_L025:260 |
|---|---|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] | 0.461 [0.40, 0.52] | 0.293 [0.24, 0.35] * (-0.168) |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] | 0.684 [0.62, 0.74] | 0.766 [0.71, 0.82] * (+0.082) |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] | 0.230 [0.18, 0.28] | 0.191 [0.14, 0.24] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] | 0.606 [0.55, 0.67] | 0.642 [0.58, 0.70] |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] | 0.121 [0.08, 0.16] | 0.117 [0.08, 0.16] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] | 0.082 [0.05, 0.12] | 0.062 [0.04, 0.09] |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] | 0.359 [0.30, 0.42] | 0.242 [0.19, 0.30] * (-0.117) |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] | 0.082 [0.05, 0.12] | 0.043 [0.02, 0.07] * (-0.039) |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] | 0.008 [0.00, 0.02] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 | 0.859 | 0.805 |
| P3' contact episodes | 256 | 250 | 249 | 240 |
