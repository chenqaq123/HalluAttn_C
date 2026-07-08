# Experiment Results — Consolidated Log

> Append-only. Every new experimental result gets a dated entry at the bottom of
> section E. Do not delete or rewrite past numbers; if a result is superseded,
> add a new entry that says so. Sign convention for detection scores:
> **higher ⇒ more likely hallucination**.
>
> **Restart baseline: 2026-07-07.** On the `restart-two-pronged` branch this log
> was reset to hold only the reproducible baseline evidence for the two-pronged
> critique (§A detection baselines, §B detection shape/per-head reproduction,
> §C mitigation baseline behavior). All prior TDEV / semantic-neighbor /
> no-external-detector result entries were removed here; they remain on `main`.
> New results for the rebuilt method are appended under §E.

## Standing setup

- Model: LLaVA-1.5-7B (HuggingFace), fp16, greedy decoding.
- Detection data: COCO val2014 captions, CHAIR object labels.
  - object mentions: 16426 (hallucinated 4009 / non-hallucinated 12417).
  - **position-only AUROC (gen_pos): 0.8304** — the confound baseline to beat.
- Mitigation data: POPE (random/popular/adversarial, 3000 each = 9000 rows) and
  CHAIR caption regeneration on a fixed COCO manifest.
- Length/position control (detection): within-bin, matched-pair, and residual
  AUROC over generation-position bins. Definitions in [design.md](design.md).

---

## A. Detection — baselines under position control (Prong 1)

Source: `detection/baselines/results/coco_llava_7b_baselines/`,
[detection/docs/results_summary.md](../detection/docs/results_summary.md).

| Score | Overall | Within-bin | Matched-pair | Residual | Note |
|---|---:|---:|---:|---:|---|
| PAS | 0.835 | 0.593 | 0.583 | 0.569 | high overall, almost entirely position-correlated |
| SVAR | 0.834 | 0.575 | 0.576 | 0.557 | same position issue |
| IC | 0.776 | **0.686** | **0.703** | 0.633 | strongest position-controlled baseline |
| GLSim-local | 0.772 | 0.564 | 0.593 | 0.601 | some residual signal |
| Entropy | 0.721 | 0.637 | 0.655 | 0.634 | reliable uncertainty baseline |
| NLL | 0.711 | 0.636 | 0.652 | 0.630 | ~Entropy |
| Beyond ADS | 0.503 | 0.511 | 0.515 | 0.501 | near random |
| Beyond CGC | 0.692 | 0.521 | 0.520 | 0.518 | weak after control |
| Beyond ADS+CGC | 0.693 | 0.523 | 0.524 | 0.521 | CGC carries the combined score |

Takeaway: PAS/SVAR's high overall AUROC is position-confounded (collapse to
~random within-bin). IC is the strongest position-controlled baseline. This is
the Prong-1 reproduction: attention-mass detectors do not survive length control.

## B. Detection — SinkDetect shape scores & per-head diagnostic (Prong 1)

Source: `experiments/coco_llava_7b_rows`, [design.md §16](design.md).

- Mean-over-head shape scores collapse under position control (best global
  `sink_only_conc_top10_mass_layer_0` = 0.8099 overall but same-bin **0.5087**).
- Best within-bin shape scores are weak: `no_rope_topmass_only_conc_top1_mass_layer_1`
  0.5751; `clc_gen_jsd` 0.5670.
- **Per-head supervised probe** (logistic, position-controlled, image_id-grouped
  CV, no leakage): within-bin AUROC rises monotonically with depth —
  layer 1 → 0.523, layer 0 → 0.603, layer 10 → 0.645, layer 22 → 0.690,
  **layer 31 → 0.730**. Layer 31 alone beats the all-layer combination (0.672).
- Conclusion: grounding signal exists but lives in late-layer *individual heads*
  and is washed out by mean-over-head aggregation.
- ⚠️ Positioning: this is **supervised** and **largely pre-empted by HaloProbe**
  (per-head + confidence probe, 93.5 AUROC). Report as background, not headline.

## C. Mitigation — attention interventions (POPE/CHAIR behavior) (Prong 2)

Scope: PAI (attention-only, no CFG branch), ClearSight VAF, Visual Attention
Sink. Vanilla greedy is the anchor.

- **PAI-attention-only:** essentially no benefit; slight POPE acc/F1 drop, no
  CHAIR reduction. (Attention-only reweight is inert — not a backfire.)
- **ClearSight:** POPE yes-rate +~3.6pts; TPR +~3.5 but FPR +~3.8 ⇒ net
  ΔTPR−ΔFPR negative; adversarial accuracy drops; Macro MCC 0.730 → 0.723.
- **Visual Attention Sink:** mild yes-rate/FPR increase on POPE; on CHAIR
  longer captions, more object mentions, more hallucinated mentions.

Takeaway: attention *amplification* changes routing/output style and raises the
yes-rate without improving target-object verification — the Prong-2 reproduction.
The related-present vs. plain-absent FPR decomposition that makes this measurable
is part of the rebuild backlog (was on `main`; to be re-derived here).

---

## E. New results (append below, newest last)

<!-- Template for each new entry:
### YYYY-MM-DD — <short title>
- Setup: model / data / script / commit
- Result: key numbers
- Interpretation: 1-3 sentences, tie back to proposal claim
- Supersedes: (if applicable)
-->

_(empty — rebuilt method results start here)_
