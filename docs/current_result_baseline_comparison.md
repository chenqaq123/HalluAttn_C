# Current Result and Baseline Comparison

Date: 2026-06-30

This note is kept as a paper-number audit target for the full semantic-neighbor
control table. It is not the current method-selection document. For the
paper-facing no-external-detector method rows, use
[no_external_detector_summary.md](no_external_detector_summary.md).

## Status

- The full control table below still includes OWLv2 rows because the paper uses
  them as external positive controls and ablations.
- The current main method rows are no-external: POPE hidden+answer verifier,
  CHAIR answer absence, and CHAIR generic-rewrite top-10 intervention.
- OWLv2 hybrid gate+rescue should be described only as an external positive
  control, not as the headline method.

## Full Semantic-Neighbor Control Table

# Semantic-Neighbor Control Table

This table is built from saved POPE metric CSVs. It keeps aggregate behavior
next to the semantic-neighbor stress test: related-present negative FPR versus
plain-absent negative FPR.

| Method | Family | Macro MCC | TPR | FPR | Yes rate | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | base | 0.730 | 0.813 | 0.087 | 0.450 | 0.114 | 0.028 | 0.086 | 0.164 |
| PAI attention-only | attention intervention | 0.728 | 0.808 | 0.084 | 0.446 | 0.110 | 0.028 | 0.082 | 0.159 |
| ClearSight VAF | attention intervention | 0.723 | 0.848 | 0.125 | 0.486 | 0.160 | 0.047 | 0.113 | 0.223 |
| VisAttnSink | attention intervention | 0.722 | 0.815 | 0.096 | 0.455 | 0.123 | 0.038 | 0.085 | 0.173 |
| VCD-greedy | contrastive decoding | 0.719 | 0.816 | 0.100 | 0.458 | 0.127 | 0.039 | 0.088 | 0.178 |
| NoLan-compatible | contrastive decoding | 0.731 | 0.778 | 0.058 | 0.418 | 0.076 | 0.018 | 0.058 | 0.107 |
| OWLv2 target direct | region evidence | 0.777 | 0.911 | 0.134 | 0.522 | 0.184 | 0.025 | 0.159 | 0.281 |
| OWLv2 margin direct | target-vs-neighbor verifier | 0.445 | 0.359 | 0.013 | 0.186 | 0.010 | 0.019 | -0.009 | 0.010 |
| Two-stage direct | target-vs-neighbor verifier | 0.769 | 0.850 | 0.083 | 0.466 | 0.111 | 0.021 | 0.090 | 0.167 |
| Two-stage gate | target-vs-neighbor verifier | 0.751 | 0.793 | 0.051 | 0.422 | 0.069 | 0.012 | 0.056 | 0.104 |
| Hybrid gate+rescue | target-vs-neighbor verifier | 0.763 | 0.806 | 0.051 | 0.429 | 0.069 | 0.012 | 0.057 | 0.105 |

Interpretation: attention-only and VCD-greedy controls do not close the
related-present false-positive gap. NoLan-compatible lowers FPR and related FPR
through language-prior suppression, but with lower TPR and yes rate. Raw
target-region evidence improves aggregate MCC but over-fires on related-present
negatives. Target-vs-neighbor verification gives the strongest external
positive-control tradeoff, while the current paper-facing method rows are the
no-external internal verifier family summarized in
[no_external_detector_summary.md](no_external_detector_summary.md).
