# NoLan-Compatible Adversarial 120 Subset

Scope: first 120 POPE-adversarial rows; NoLan is a deterministic compatible port, not an official reproduction.

| Method | Samples | MCC | TPR | FPR | Yes Rate | Related FPR | Delta Related FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 120 | 0.667 | 0.850 | 0.183 | 0.517 | 0.204 | +0.000 |
| NoLan-compatible | 120 | 0.700 | 0.833 | 0.133 | 0.483 | 0.148 | -0.056 |
| SPIN default | 120 | 0.092 | 1.000 | 0.983 | 0.992 | 1.000 | +0.796 |
| DAMRO | 120 | 0.639 | 0.883 | 0.250 | 0.567 | 0.278 | +0.074 |

Reading: NoLan-compatible improves this small adversarial subset over vanilla by lowering FPR and related-present FPR, with a small TPR drop. It is much healthier than SPIN default and DAMRO on the same subset, but it remains subset evidence and should not be reported as a full official NoLan baseline.
