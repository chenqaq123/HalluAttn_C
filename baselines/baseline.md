# Baselines for Token-Level LVLM Object Hallucination Detection

This file lists the baselines we should compare against in the paper-facing
experiments. When a paper proposes multiple analyses or ablations, we keep only
the final method used as its main experimental detector.

## Baseline Set

| Baseline | Category | Main score used for comparison | Notes |
|---|---|---|---|
| NLL | Logit / uncertainty | Negative log-likelihood of the object token | Higher uncertainty implies higher hallucination risk. |
| Entropy | Logit / uncertainty | Entropy of the next-token distribution at the object step | Training-free and does not use hidden states or attention. |
| IC | Representation | Internal confidence / image-object compatibility from model representations | Representation-based baseline used in PAS. |
| GLSim | Representation | Global-local similarity between visual representations and object semantics | Representation-based image-object compatibility baseline. |
| SVAR | Global attention | Summed Visual Attention Ratio from object token to image tokens | Strong global image-attention baseline. |
| PAS | Prelim attention | Layer-0 Prelim Attention Score | Final PAS detector; do not separately list PAS ablations such as instruction/image/BOS attention or MI variants. |
| Beyond Global Scores | Fine-grained patch grounding | Final ADS+CGC combined detector | Combines patch-level Attention Dispersion Score and Cross-modal Grounding Consistency. Do not list ADS and CGC as separate baselines unless doing ablation. |

## Grouping

### Logit-Level Baselines

- **NLL**: uses the model's likelihood for the generated object token.
- **Entropy**: uses uncertainty of the output distribution at the object-token
  step.

These are cheap and broadly applicable, but they can miss confident
hallucinations.

### Representation-Level Baselines

- **IC**: estimates whether internal image representations support the object.
- **GLSim**: compares global/local visual representations against object
  semantics.

These use hidden representations rather than attention mass alone.

### Attention-Level Baselines

- **SVAR**: global image-token attention. It asks how much the object token
  attends to the image as a whole.
- **PAS**: prelim-token attention. It asks how much the object token depends on
  previously generated tokens; higher prelim dependence indicates higher
  hallucination risk.

PAS is the primary baseline for our project because it is the strongest recent
training-free attention-based detector and is directly comparable to our
attention-row pipeline.

### Fine-Grained Grounding Baseline

- **Beyond Global Scores / ADS+CGC**: patch-level detector that combines:
  - Attention dispersion: whether token-to-patch attention is localized or
    diffuse.
  - Cross-modal grounding consistency: whether token and patch representations
    show local semantic alignment.

For final comparison, use the combined detector reported by the paper. ADS-only
or CGC-only should be treated as ablations, not separate baselines.

## Reporting Rules

1. Use each paper's final detector as the baseline.
2. Do not list internal ablations as independent baselines.
3. Report overall AUROC for comparability with prior work.
4. Also report position-controlled metrics for our analysis:
   - position-only AUROC;
   - within-bin AUROC;
   - matched-pair AUROC;
   - residual AUROC.
5. When possible, evaluate all baselines on the same generated captions and
   CHAIR-derived object labels.

## Primary References

- PAS: *Prelim Attention Score for Detecting Object Hallucinations in Large
  Vision-Language Models*, CVPR 2026.
- Beyond Global Scores: *Beyond the Global Scores: Fine-Grained Token
  Grounding as a Robust Detector of LVLM Hallucinations*, CVPR 2026.
