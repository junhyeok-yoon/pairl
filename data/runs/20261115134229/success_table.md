# Success table: 20261115134229

Deterministic policy unless stated; 256 evaluation episodes per start; this PC's CPU; bootstrap 95% CI in brackets. '*' = paired difference against the previous best (None) with a CI excluding 0. Definitions: eval/diagnostics/phase_success.py.

| | L1 reference | attempt 3 @300 |
|---|---|---|
| P1 grasp | 1.000 [1.00, 1.00] | 0.137 [0.10, 0.18] |
| P2 carry (alone) | 0.992 [0.98, 1.00] | 0.336 [0.28, 0.39] |
| P3 press (alone) | 0.781 [0.73, 0.83] | 0.137 [0.10, 0.18] |
| P3' class (alone, contact) | 0.996 [0.99, 1.00] | 0.488 [0.43, 0.55] |
| P4 release (alone) | 1.000 [1.00, 1.00] | 0.004 [0.00, 0.01] |
| P4 release (alone, gripper sampled) | 1.000 [1.00, 1.00] | 0.008 [0.00, 0.02] |
| chain P1&P2 | 0.996 [0.99, 1.00] | 0.051 [0.03, 0.08] |
| chain P1-P3 | 0.922 [0.89, 0.95] | 0.004 [0.00, 0.01] |
| **full chain P1-P4** | 0.922 [0.89, 0.95] | 0.000 [0.00, 0.00] |
| late grasp (any time) | 1.000 | 0.680 |
| P3' contact episodes | 256 | 250 |
