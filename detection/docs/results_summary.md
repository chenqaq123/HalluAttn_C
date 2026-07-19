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

## Beyond Global Scores + 2026 attention detectors (repo-audited, v3, 2026-07-08)

All numbers below are faithful to the official GitHub repos (see
`docs/experiment_results.md` §E 2026-07-08 repo-audit entry). Computed by
`detection/scripts/{extract_beyond_svar_features,evaluate_beyond_svar.py}` on the
full 16426-mention set (`experiments/beyond_svar_v3/`).

- **Beyond ADS/CGC** (arXiv:2604.04863, repo `tuandung2812alt3/token-grounding-detector`):
  ADS = `(1−m)·Ĥ_bg` (top-10% fg, 8-connected blob mass, ÷log576); CGC = mean cosine
  of the object token's **generating-row** hidden state to its top-5% similar patches.
  **Single-layer ADS is diagnostic only** — the paper's detector is the *supervised
  per-layer vector*. CGC readout fixed to `token_pos` (was `token_pos+1`, a bug).
- **SAE** (arXiv:2603.16558, no repo): true = Mask2Former-Swin-L segmentation-region
  entropy; seg-free = raw-576-patch entropy (a dispersion proxy, **not** SAE).
- **LaSCD** (arXiv:2605.11559, repo `macovaseas/LaSCD`): `lascd_raw` = per-head 2D
  Laplacian energy of the visual attention grid (verbatim `_deco_layer_energy`).
- **TVER** (arXiv:2507.00898, repo `zifuwan/ONLY`): layer-0 per-head
  `H(text attn)/H(visual attn)`, mean over heads.

| Score | Type | Overall | Within-bin | Interpretation |
|---|---|---:|---:|---|
| TVER (ICCV'25, L0) | attn, unsup | 0.8303 | 0.5655 | **overall = position baseline (0.8304)**; collapses |
| SAE — true (Mask2Former, L10) | attn×seg, unsup | 0.7833 | 0.5747 | overall≈paper 0.766; **collapses** (segmenter doesn't rescue) |
| SAE — seg-free (L16) | attn disp., unsup | 0.7922 | 0.6083 | dispersion proxy, not the paper's SAE |
| ADS (1-layer, L16) | attn, unsup | 0.6883 | 0.5550 | diagnostic only |
| CGC (faithful, L31) | repr, unsup | 0.7276 | 0.5737 | collapses (old 0.752 was the token+1 bug) |
| LaSCD (norm shape, mean L11–28) | attn, unsup | 0.5585 | 0.5218 | near-chance; energy is a within-token *layer selector*, not a detector (raw energy = mass proxy, anti-correlated) |
| ADS-only supervised | attn | 0.8401 | 0.6639 | multi-layer pooling partly survives |
| CGC-only supervised | repr | 0.8372 | 0.6688 | " |
| ADS+CGC supervised | attn+repr | 0.8661 | 0.7185 | logreg suite ceiling (barely > IC 0.686) |

## Takeaway

Judge these by within-bin, not overall — the position-only baseline is already
0.8304, and **TVER's overall (0.8303) lands exactly on it.** Every unsupervised
single-score attention detector (PAS, SVAR, ADS, SAE incl. its true segmenter
version, LaSCD, TVER) collapses to 0.50–0.61 within-bin. The faithful CGC also
collapses (0.574) — the earlier "grounding survives" reading was an indexing
artifact. What retains moderate length-controlled signal is the single logit-lens
**IC** (0.686) and **supervised 32-layer pooling** (ADS 0.664 / CGC 0.669 /
combined 0.719).
