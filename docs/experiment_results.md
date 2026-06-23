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

### 2026-06-23 — Internal attention-region TDEV route B, 120-row POPE bounded audit
- Setup: LLaVA-1.5-7B, same 360-row bounded POPE subset as route A, no external detector. Script: `mitigation/scripts/evaluate_attention_region_tdev_pope.py`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py`. Evidence backend compares visual-token attention regions for the target question and top-1 semantic-neighbor question using layers 22 and 31.
- Artifacts: `mitigation/results/semantic_neighbor_audit/attention_region_tdev_120/`.
- Result: scoring completed for 360/360 rows, failures 0. Direct combined attention-region score is weak: MCC 0.173, TPR 0.550, FPR 0.378, related FPR 0.418, plain FPR 0.261. Gate-on-vanilla with calibrated threshold is unchanged from vanilla: MCC 0.739, TPR 0.850, FPR 0.111, related FPR 0.149, plain FPR 0.000. JSD-only and concentration-only variants show the same pattern: direct detection is poor and calibrated gate chooses no effective suppression.
- Interpretation: route B in its current mean-region form is a negative result. Target and semantic-neighbor object-token attention regions overlap too strongly; this does not yield a reliable target-vs-neighbor verifier. Keep as an ablation against the claim that raw attention localization is enough.
- Supersedes: none.

### 2026-06-23 — Contrastive per-head supervised probe route C, 120-row POPE bounded audit
- Setup: LLaVA-1.5-7B, same 360-row bounded POPE subset, no external detector. Script: `mitigation/scripts/evaluate_contrastive_head_probe_pope.py`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py`. The probe extracts per-head attention-shape features for target prompt, semantic-neighbor prompt, and target-minus-neighbor contrast over layers 22/31, then trains an image-grouped 5-fold out-of-fold logistic readout. This is a supervised diagnostic ceiling/foil, not the training-free headline method.
- Artifacts: `mitigation/results/semantic_neighbor_audit/contrastive_head_probe_120/`.
- Result: scoring completed for 360/360 rows, failures 0, feature dim 768. Intrinsic absent-target AUROC is 0.731 and OOF absent detector MCC is 0.415. Converted to POPE yes/no support score, direct probe gets MCC 0.419, TPR 0.550, FPR 0.150, related FPR 0.179, plain FPR 0.065. Calibrated gate-on-vanilla is unchanged from vanilla because random-split MCC calibration prefers no suppression. A stricter zero-threshold gate gives MCC 0.537, TPR 0.567, FPR 0.067, related FPR 0.090, plain FPR 0.000.
- Interpretation: route C confirms internal per-head target-vs-neighbor features contain signal, but the signal is not strong enough to replace the external verifier or to improve vanilla without a large recall loss. It is useful as a HaloProbe-style supervised ceiling/foil and as evidence that simple internal attention probes are not yet the final method.
- Supersedes: route B as the stronger internal diagnostic, but not as a deployable main method.

#### 2026-06-23 bounded internal-route comparison table

| Method | Subset | Samples | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|
| Vanilla | all | 360 | 0.739 | 0.850 | 0.111 | 0.481 |
| Vanilla | related-present neg | 134 | 0.000 | 0.000 | 0.149 | 0.149 |
| Route A IC target gate | all | 360 | 0.745 | 0.850 | 0.106 | 0.478 |
| Route A strict IC margin gate | all | 360 | 0.358 | 0.267 | 0.017 | 0.142 |
| Route A strict IC margin gate | related-present neg | 134 | 0.000 | 0.000 | 0.022 | 0.022 |
| Route B attention-region direct | all | 360 | 0.173 | 0.550 | 0.378 | 0.464 |
| Route B attention-region gate | all | 360 | 0.739 | 0.850 | 0.111 | 0.481 |
| Route C contrastive probe direct | all | 360 | 0.419 | 0.550 | 0.150 | 0.350 |
| Route C contrastive probe direct | related-present neg | 134 | 0.000 | 0.000 | 0.179 | 0.179 |
| Route C contrastive probe strict gate | all | 360 | 0.537 | 0.567 | 0.067 | 0.317 |
| Route C contrastive probe strict gate | related-present neg | 134 | 0.000 | 0.000 | 0.090 | 0.090 |

Bottom line: none of A/B/C currently supports replacing OWLv2-backed TDEV as the main result. A and C can reduce related-present FPR only by sacrificing recall; B is mostly non-discriminative. The next step should not be polishing these exact signals, but either (1) a stronger internal hidden-state/logit contrast beyond attention shape, or (2) caption-side atomic claim generation + internal target-vs-neighbor verification where the decision surface is local rather than a global yes/no gate.

### 2026-06-23 — Hidden-state contrast probe route D, 120-row POPE bounded audit
- Setup: LLaVA-1.5-7B, same 360-row bounded POPE subset, no external detector. Script: `mitigation/scripts/evaluate_hidden_contrast_probe_pope.py`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py`. The probe extracts selected-layer hidden states for target and semantic-neighbor prompts: object question-token hidden, visual-token mean hidden, and target-minus-neighbor contrasts over layers 22/31. Readout is image-grouped 5-fold out-of-fold logistic regression, so this remains a supervised diagnostic/ceiling rather than a training-free final method.
- Artifacts: `mitigation/results/semantic_neighbor_audit/hidden_contrast_probe_120/`.
- Result: scoring completed for 360/360 rows, failures 0, feature dim 49152. Intrinsic absent-target AUROC is 0.863 and OOF absent detector MCC is 0.536, clearly stronger than the route C attention-shape probe (AUROC 0.731, MCC 0.415). Direct support scoring still over-fires: MCC 0.594, TPR 0.861, FPR 0.272, related FPR 0.313. Calibrated gate gives a tiny improvement over vanilla/A-like behavior: MCC 0.745, TPR 0.850, FPR 0.106, related FPR 0.142. Tuned gate at `support_score > -12.1203` gives the best practical tradeoff so far among internal probes: MCC 0.707, TPR 0.778, FPR 0.078, related FPR 0.104, plain FPR 0.000. Stricter zero-threshold gate gives related FPR 0.075 but TPR drops to 0.633.
- Interpretation: route D is the strongest internal replacement candidate so far. It proves hidden-state target-vs-neighbor contrast contains substantially more useful evidence than raw attention regions or per-head attention shape. However, it still does not match the original OWLv2-backed TDEV tradeoff or beat vanilla/A on aggregate MCC without recall loss. Keep it as the current best internal diagnostic route and tune it next; do not claim external-detector removal is solved yet.
- Supersedes: route C as the strongest internal supervised diagnostic.

#### Updated internal-route comparison including hidden contrast

| Method | Subset | Samples | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|
| Vanilla | all | 360 | 0.739 | 0.850 | 0.111 | 0.481 |
| Vanilla | related-present neg | 134 | 0.000 | 0.000 | 0.149 | 0.149 |
| Route A IC target gate | all | 360 | 0.745 | 0.850 | 0.106 | 0.478 |
| Route C attention-shape strict gate | all | 360 | 0.537 | 0.567 | 0.067 | 0.317 |
| Route C attention-shape strict gate | related-present neg | 134 | 0.000 | 0.000 | 0.090 | 0.090 |
| **Route D hidden contrast tuned gate** | **all** | **360** | **0.707** | **0.778** | **0.078** | **0.428** |
| **Route D hidden contrast tuned gate** | **related-present neg** | **134** | **0.000** | **0.000** | **0.104** | **0.104** |
| Route D hidden contrast strict gate | all | 360 | 0.608 | 0.633 | 0.056 | 0.344 |
| Route D hidden contrast strict gate | related-present neg | 134 | 0.000 | 0.000 | 0.075 | 0.075 |

Route D is the first internal path that materially improves the related-present FPR while preserving moderate recall. The remaining gap is that the tradeoff is still worse than external TDEV and not yet a clear aggregate improvement over vanilla/A. Next tuning should try lower-dimensional hidden features, layer sweeps, and unsupervised/weakly supervised margins derived from the learned direction so the final method does not rely on supervised POPE labels.
