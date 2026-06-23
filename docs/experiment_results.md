# Experiment Results — Consolidated Log

> Append-only. Every new experimental result gets a dated entry at the bottom of
> the relevant section. Do not delete or rewrite past numbers; if a result is
> superseded, add a new entry that says so. Sign convention for detection
> scores: **higher ⇒ more likely hallucination**.

## Standing setup

- Model: LLaVA-1.5-7B (HuggingFace), fp16, greedy decoding.
- Detection data: COCO val2014 captions, CHAIR object labels.
  - object mentions: 16426 (hallucinated 4009 / non-hallucinated 12417).
  - **position-only AUROC (gen_pos): 0.8304** — the confound baseline to beat.
- Mitigation data: POPE (random/popular/adversarial, 3000 each = 9000 rows) and
  CHAIR caption regeneration on a fixed COCO manifest.

---

## A. Detection — baselines under position control

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
~random within-bin). IC is the strongest position-controlled baseline.

## B. Detection — SinkDetect shape scores & per-head diagnostic

Source: `experiments/coco_llava_7b_rows`, [design.md §16](design.md).

- Mean-over-head shape scores collapse under position control (best global
  `sink_only_conc_top10_mass_layer_0` = 0.8099 overall but same-bin 0.5087).
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

## C. Mitigation — attention interventions (POPE/CHAIR behavior)

Scope: PAI (attention-only, no CFG branch), ClearSight VAF, Visual Attention
Sink. Vanilla greedy is the anchor.

- **PAI-attention-only:** essentially no benefit; slight POPE acc/F1 drop, no
  CHAIR reduction.
- **ClearSight:** POPE yes-rate +~3.6pts; TPR +~3.5 but FPR +~3.8 ⇒ net
  ΔTPR−ΔFPR negative; adversarial accuracy drops.
- **Visual Attention Sink:** mild yes-rate/FPR increase on POPE; on CHAIR
  longer captions, more object mentions, more hallucinated mentions.

Takeaway: interventions change routing/output style but not target-object
verification.

## D. Semantic-neighbor stress test (the headline)

### D.1 Subset construction (full POPE coverage)

Source: [semantic_neighbor_audit.md](semantic_neighbor_audit.md). All 9000 rows
parsed (0 unparsed, 0 missing ids).

| Split | Rows | Related-present neg | Plain-absent neg | Related rate among neg |
|---|---:|---:|---:|---:|
| random | 3000 | 840 | 660 | 56.0% |
| popular | 3000 | 980 | 520 | 65.3% |
| adversarial | 3000 | 1272 | 228 | **84.8%** |

### D.2 FPR gap by negative subset (related-present vs plain-absent)

Attention-only methods + VCD — the gap is never closed and grows on harder splits:

| Split | Method | All neg | Related-present | Plain absent | Gap |
|---|---|---:|---:|---:|---:|
| adversarial | vanilla | 14.7% | 16.4% | 5.3% | +11.1% |
| adversarial | PAI | 14.3% | 15.9% | 5.3% | +10.6% |
| adversarial | ClearSight | 20.2% | 22.3% | 8.3% | +14.0% |
| adversarial | VisAttnSink | 15.8% | 17.3% | 7.5% | +9.8% |
| adversarial | VCD-greedy | 16.2% | 17.8% | 7.5% | +10.3% |

(random/popular splits show the same pattern at smaller magnitude — see source.)

### D.3 TDEV control table (POPE, with semantic-neighbor stress)

Source: `mitigation/results/semantic_neighbor_audit/paper_control_table/`.

| Method | Family | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | base | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.086 | 0.164 |
| PAI attn-only | attn interv. | 0.728 | 0.808 | 0.084 | 0.110 | 0.028 | 0.082 | 0.159 |
| ClearSight VAF | attn interv. | 0.723 | 0.848 | 0.125 | 0.160 | 0.047 | 0.113 | 0.223 |
| VisAttnSink | attn interv. | 0.722 | 0.815 | 0.096 | 0.123 | 0.038 | 0.085 | 0.173 |
| VCD-greedy | contrastive | 0.719 | 0.816 | 0.100 | 0.127 | 0.039 | 0.088 | 0.178 |
| NoLan-compatible | contrastive | 0.731 | 0.778 | 0.058 | 0.076 | 0.018 | 0.058 | 0.107 |
| OWLv2 target direct | region evidence | 0.777 | 0.911 | 0.134 | 0.184 | 0.025 | 0.159 | 0.281 |
| OWLv2 margin direct | target-vs-neighbor | 0.445 | 0.359 | 0.013 | 0.010 | 0.019 | -0.009 | 0.010 |
| Two-stage direct | target-vs-neighbor | 0.769 | 0.850 | 0.083 | 0.111 | 0.021 | 0.090 | 0.167 |
| Two-stage gate | target-vs-neighbor | 0.751 | 0.793 | 0.051 | 0.069 | 0.012 | 0.056 | 0.104 |
| **Hybrid gate+rescue** | target-vs-neighbor | **0.763** | 0.806 | **0.051** | 0.069 | 0.012 | 0.057 | **0.105** |

Notes: attention-only and VCD do not close the related-present gap. Raw target
detection (OWLv2 direct) boosts MCC but over-fires on related-present
(adv. related FPR 0.281). Target-vs-neighbor verification (hybrid gate+rescue)
gives the best current tradeoff. **Caveat: this uses an external OWLv2 backend —
the proposal calls for replacing it with internal self-contrastive readout.**

### D.4 TDEV on CHAIR (not POPE-only)

Source: [tdev_detector_positioning.md](tdev_detector_positioning.md).

- Hybrid MCC positive branch: overall AUROC 0.874, within-bin 0.853,
  matched-pair 0.855.
- Continuous "target-absence + 0.25·neighbor-dominance" score raises residual
  AUROC from 0.711 → 0.722.

---

## E. New results (append below, newest last)

<!-- Template for each new entry:
### YYYY-MM-DD — <short title>
- Setup: model / data / script / commit
- Result: key numbers
- Interpretation: 1-3 sentences, tie back to proposal claim
- Supersedes: (if applicable)
-->

_(none yet — first new result goes here)_

### 2026-06-23 — Internal IC/logit-lens TDEV route A, 120-row POPE bounded audit
- Setup: LLaVA-1.5-7B, POPE random/popular/adversarial first 120 rows per split (360 rows total), no external detector. Script: `mitigation/scripts/evaluate_internal_ic_tdev_pope.py`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py`. Evidence backend projects final visual-token hidden states through the LM head and compares `target_score` against the best top-10 semantic neighbor score.
- Artifacts: `mitigation/results/semantic_neighbor_audit/internal_ic_tdev_120/`.
- Result: scoring completed for 360/360 rows, failures 0. On the same subset, vanilla has MCC 0.739, TPR 0.850, FPR 0.111, related FPR 0.149, plain FPR 0.000. Calibrated target-only IC gate is nearly unchanged: MCC 0.745, TPR 0.850, FPR 0.106, related FPR 0.142. Calibrated margin gate is exactly unchanged from vanilla because calibration chooses a very low threshold. Strict `target_score > neighbor_score` gate lowers FPR to 0.017 and related FPR to 0.022, but collapses TPR to 0.267 and MCC to 0.358. Direct margin scoring is also too conservative: calibrated direct margin has MCC 0.241, TPR 0.267, FPR 0.083, related FPR 0.097.
- Interpretation: route A is useful as a cheap internal baseline but not the headline method. Pure final-layer visual logit-lens evidence is not target-discriminative enough; adding the neighbor margin suppresses related false positives only by sacrificing recall. This pushes the next implementation toward route B (attention-region discriminability) and route C/HaloProbe-style supervised contrastive probes.
- Supersedes: none; this is the first no-external-detector TDEV backend audit.
