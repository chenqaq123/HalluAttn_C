# Full POPE-Adversarial Baseline Comparison

NoLan-compatible is a deterministic local compatibility port, not an official NoLan reproduction.

| Method | Samples | MCC | TPR | FPR | Yes Rate | Related FPR | Plain FPR | Gap | Delta MCC | Delta Related FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 3000 | 0.666 | 0.812 | 0.147 | 0.479 | 0.164 | 0.053 | 0.111 | 0.000 | 0.000 |
| PAI attention-only | 3000 | 0.665 | 0.807 | 0.143 | 0.475 | 0.159 | 0.053 | 0.106 | -0.001 | -0.005 |
| ClearSight VAF | 3000 | 0.646 | 0.847 | 0.202 | 0.525 | 0.223 | 0.083 | 0.140 | -0.020 | 0.060 |
| VisAttnSink | 3000 | 0.656 | 0.814 | 0.158 | 0.486 | 0.173 | 0.075 | 0.098 | -0.010 | 0.009 |
| VCD-greedy | 3000 | 0.654 | 0.815 | 0.162 | 0.489 | 0.178 | 0.075 | 0.103 | -0.012 | 0.014 |
| NoLan-compatible | 3000 | 0.688 | 0.779 | 0.097 | 0.438 | 0.107 | 0.039 | 0.067 | 0.022 | -0.057 |

Reading: NoLan-compatible is the only full adversarial decoding baseline here that clearly lowers related-present FPR relative to the matched vanilla row, from 0.164 to 0.107, but it pays for this with a TPR drop from 0.812 to 0.779 and only a modest MCC gain, from 0.666 to 0.688. This is helpful as a baseline result, but it does not erase the need for explicit target-vs-neighbor verification.
