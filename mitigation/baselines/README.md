# Mitigation Baselines

This file tracks mitigation methods that we have discussed and may reproduce.
They are not implemented yet.

## Baseline List

| Method | Type | Status | What to test |
|---|---|---|---|
| ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large Language Models | visual signal enhancement | pending | Whether POPE gains come from true TPR/FPR separation or a yes-rate shift. |
| PAI: Paying More Attention to Image | training-free image-attention emphasis | pending | Whether increasing image attention makes the model answer "yes" more often. |
| VCD-style visual contrastive decoding | contrastive decoding intervention | optional | Useful non-head-selection comparison if time allows. |
| Head-selection / head-enhancement methods | attention-head intervention | pending | Potential positive counterexample; may align with our per-head diagnostic. |

## Required Reporting

For each method and dataset split, report:

| Metric | Meaning |
|---|---|
| Accuracy / F1 | legacy comparability |
| Yes rate | overall tendency to answer yes |
| TPR / Recall | present-object questions answered yes |
| FPR | absent-object questions incorrectly answered yes |
| TNR / Specificity | absent-object questions answered no |
| Balanced accuracy | `(TPR + TNR) / 2` |
| MCC | skew-resistant binary metric |
| `Delta TPR - Delta FPR` | separates grounding gain from yes-bias shift |

Interpretation:

| Pattern | Interpretation |
|---|---|
| `Delta TPR > 0`, `Delta FPR <= 0` | credible mitigation |
| `Delta TPR > Delta FPR > 0` | useful but answer-prior shift exists |
| `Delta FPR >= Delta TPR > 0` | likely yes-bias / threshold shift |
| `Delta TPR ~= Delta FPR` | mostly global answer-prior movement |

## Notes

The head-selection family should be treated carefully. It may become a positive
case rather than a failure case, because our detection-side per-head diagnostic
suggests that useful grounding signal can live in specific attention heads.
