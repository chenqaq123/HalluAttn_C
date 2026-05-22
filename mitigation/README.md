# Mitigation Track

This directory is for hallucination **mitigation** methods and controlled
analysis of their benchmark behavior.

Current status: scaffold only. No mitigation method has been reproduced in this
repo yet.

## Current Question

Some attention/visual-enhancement interventions may improve aggregate POPE
metrics by shifting the model's yes/no answer prior rather than by improving
visual grounding. We will test this with answer-prior diagnostics:

- yes rate;
- TPR / recall on present-object questions;
- FPR on absent-object questions;
- TNR / specificity;
- balanced accuracy;
- MCC;
- `Delta TPR - Delta FPR` relative to the base model.

## Layout

```text
mitigation/
├── baselines/   # method list and reproduction notes
├── docs/        # mitigation-side design and analysis notes
├── scripts/     # future POPE runners and analysis scripts
└── src/         # future reusable mitigation/evaluation code
```

Do not treat this track as evidence yet. It becomes paper-facing only after
the baselines are reproduced and evaluated under the diagnostics above.
