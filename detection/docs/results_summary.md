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
| PAS | 0.8349 | 0.5928 | 0.5833 | 0.5694 | High overall, but almost entirely position-correlated. Faithful to `lanl/pas`. |
| SVAR (fixed, layers 5–18) | 0.8053 | 0.5678 | 0.5664 | 0.5484 | Faithful middle-layer SVAR (Jiang et al.); still collapses under length control. Old layer-0 mass was 0.8336/0.5749. |
| IC | 0.7761 | 0.6856 | 0.7029 | 0.6331 | Strongest position-controlled baseline. **Note:** final-layer logit-lens only (paper maxes over all layers), so this is a conservative lower bound. |
| GLSim-local | 0.7724 | 0.5636 | 0.5931 | 0.6006 | Has some residual grounding signal. |
| Entropy | 0.7206 | 0.6366 | 0.6545 | 0.6335 | Reliable uncertainty baseline after position control. |
| NLL | 0.7114 | 0.6360 | 0.6524 | 0.6297 | Similar to Entropy. |

## Beyond Global Scores (fixed to the paper, 2026-07-08)

Now faithful to arXiv:2604.04863: ADS = `(1−m)·Ĥ_bg` with top-10% foreground +
8-connected blob mass and background entropy ÷ log576; CGC ranks top-5% patches
**by cosine similarity** at the same token/patch layer; ADS+CGC is the paper's
**supervised** detector (image-grouped OOF over the 64-dim per-layer `[ADS‖CGC]`
vector; sklearn HGB, paper uses XGBoost). Computed by
`detection/scripts/{extract_beyond_svar_features,evaluate_beyond_svar}.py`.

| Score | Overall | Within-bin | Matched-pair | Residual | Interpretation |
|---|---:|---:|---:|---:|---|
| ADS (blob-mass, L16) | 0.6883 | 0.5543 | 0.5587 | 0.5374 | Attention dispersion; still weak within-bin (attention-based). |
| CGC (sim-topk, best L18) | 0.8720 | 0.7516 | 0.7500 | 0.6785 | Hidden-state grounding; **survives length control**. |
| ADS+CGC (supervised OOF) | 0.8898 | 0.7930 | 0.7842 | 0.7477 | Strongest controlled detector in the suite. |

(Old broken proxies were ADS 0.503/0.511, CGC 0.692/0.521, ADS+CGC 0.693/0.523.)

## Takeaway

PAS and SVAR should not be judged only by overall AUROC, because the
position-only baseline already reaches 0.8304. IC is the strongest baseline
under position-controlled metrics, followed by Entropy and NLL. For Beyond
Global Scores, reporting ADS, CGC, and ADS+CGC separately is useful because ADS
is near random and CGC carries almost all of the combined score.
