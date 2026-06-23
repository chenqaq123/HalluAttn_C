# Baseline Result Summary

Current result directory:

```bash
detection/baselines/results/coco_llava_7b_baselines/
```

Dataset/object cache:

- object mentions: 16426
- hallucinated: 4009
- non-hallucinated: 12417
- position-only AUROC: 0.8304

Reproduction scope: NLL, Entropy, PAS, and SVAR are direct score
implementations. IC, GLSim, and Beyond-ADS/CGC are adapted/paper-level
implementations and should be reported with that qualifier.

## Main Baselines

| Score | Overall | Within-bin | Matched-pair | Residual | Note |
|---|---:|---:|---:|---:|---|
| PAS | 0.8349 | 0.5928 | 0.5833 | 0.5694 | High overall, but almost entirely position-correlated. |
| SVAR | 0.8336 | 0.5748 | 0.5759 | 0.5570 | Similar position issue as PAS. |
| IC | 0.7761 | 0.6856 | 0.7029 | 0.6331 | Strongest position-controlled baseline. |
| GLSim-local | 0.7724 | 0.5636 | 0.5931 | 0.6006 | Has some residual grounding signal. |
| Entropy | 0.7206 | 0.6366 | 0.6545 | 0.6335 | Reliable uncertainty baseline after position control. |
| NLL | 0.7114 | 0.6360 | 0.6524 | 0.6297 | Similar to Entropy. |

## Beyond Global Scores

We report the two components and their combined detector:

| Score | Overall | Within-bin | Matched-pair | Residual | Interpretation |
|---|---:|---:|---:|---:|---|
| ADS | 0.5033 | 0.5109 | 0.5148 | 0.5012 | Attention dispersion alone is near random. |
| CGC | 0.6917 | 0.5208 | 0.5196 | 0.5181 | Mostly weak after position control. |
| ADS+CGC | 0.6926 | 0.5226 | 0.5235 | 0.5212 | Combined score does not recover much controlled signal. |

## Takeaway

PAS and SVAR should not be judged only by overall AUROC, because the
position-only baseline already reaches 0.8304. IC is the strongest baseline
under position-controlled metrics, followed by Entropy and NLL. For Beyond
Global Scores, reporting ADS, CGC, and ADS+CGC separately is useful because ADS
is near random and CGC carries almost all of the combined score.
