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

### 2026-06-23 — Training-free hidden-margin D-lite, 120-row POPE bounded audit
- Setup: LLaVA-1.5-7B, same 360-row bounded POPE subset, no external detector and no supervised readout. Script: `mitigation/scripts/evaluate_hidden_margin_tdev_pope.py`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py`. The score uses selected-layer hidden vectors from target and semantic-neighbor prompts: object-token vs visual-mean alignment margin, cross-prompt margin, and small object/visual separation terms. Formula: `hidden_align_margin + hidden_cross_margin + 0.25*hidden_obj_separation + 0.25*hidden_vis_separation`, selecting the worst semantic neighbor.
- Artifacts: `mitigation/results/semantic_neighbor_audit/hidden_margin_tdev_120/`.
- Result: scoring completed for 360/360 rows, failures 0. Calibrated gate behaves like the cheap A-style gate: MCC 0.745, TPR 0.850, FPR 0.106, related FPR 0.142. The meaningful training-free operating point is the zero-threshold gate (`hidden_margin_score > 0`): MCC 0.670, TPR 0.750, FPR 0.089, related FPR 0.119, plain FPR 0.000. Strict `hidden_align_margin > 0` lowers related FPR to 0.104 but drops TPR to 0.650 and MCC to 0.595. Direct hidden-margin classification is poor (MCC 0.024 for the combined score), so D-lite should be used as a gate, not a standalone yes/no detector.
- Interpretation: D-lite is the first training-free, external-detector-free candidate that gives a nontrivial related-FPR reduction without catastrophic recall collapse. It still does not match supervised Route D or OWLv2-backed TDEV, but it is the best deployable internal score so far. Next tuning should optimize the D-lite formula/layers against the supervised D direction, then test full POPE.
- Supersedes: route A as the stronger training-free internal candidate for semantic-neighbor FPR reduction, but not route D as the supervised ceiling.

#### Current best internal candidates after D-lite

| Method | Supervised | Deployable without external detector | MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---|---|---:|---:|---:|---:|---:|
| Vanilla | no | yes | 0.739 | 0.850 | 0.111 | 0.149 | 0.000 |
| Route A IC target gate | no | yes | 0.745 | 0.850 | 0.106 | 0.142 | 0.000 |
| **Route D-lite hidden margin zero gate** | **no** | **yes** | **0.670** | **0.750** | **0.089** | **0.119** | **0.000** |
| Route D hidden contrast tuned gate | yes | no | 0.707 | 0.778 | 0.078 | 0.104 | 0.000 |
| Route D hidden contrast strict gate | yes | no | 0.608 | 0.633 | 0.056 | 0.075 | 0.000 |

Operational read: A remains best if aggregate MCC is the only objective. D-lite is better if the paper's actual target is reducing the semantic-neighbor related-present false-positive gap while keeping recall above ~0.75. The method is not ready to replace the external detector in the main table until the D-lite tradeoff improves or holds on full POPE.

### 2026-06-23 — D-lite cache-only formula sweep, 120-row POPE bounded audit
- Setup: Cache-only tuning on `hidden_margin_tdev_120/hidden_margin_tdev_predictions.csv`, no VLM rerun and no external detector. Script: `mitigation/scripts/tune_hidden_margin_formula.py`. The sweep uses POPE-random as the calibration split, a small interpretable weight grid over `[hidden_align_margin, hidden_cross_margin, hidden_obj_separation, hidden_vis_separation]`, and vanilla predictions as the gate base. Popular/adversarial are therefore not used to select the formula.
- Artifacts: `mitigation/results/semantic_neighbor_audit/hidden_margin_formula_sweep_small_vanillagate_120/`.
- Result: best calibrated gate with TPR >= 0.85 uses weights `[0, 1, -0.5, -0.5]` and threshold `-0.3032`. On the full 360-row bounded set it reaches MCC 0.751, TPR 0.850, FPR 0.100, related FPR 0.134, plain FPR 0.000. This improves over A IC target gate (MCC 0.745, FPR 0.106, related FPR 0.142) while preserving vanilla TPR. A lower-FPR calibrated point (`[0.5, -0.5, -0.5, -0.5]`, threshold `-0.1981`) gets MCC 0.735, TPR 0.833, FPR 0.100, related FPR 0.134.
- Interpretation: this is the first training-free internal D-lite variant that slightly beats route A on aggregate MCC and semantic-neighbor FPR while keeping TPR at 0.850. The gain is small on the 360-row bounded set, so it needs full-POPE validation before replacing the external detector in the main table. Still, this is now the best no-external-detector deployable candidate.
- Supersedes: the earlier zero-threshold D-lite operating point as the preferred bounded-set deployable candidate.

#### Updated deployable internal candidate table

| Method | Calibration | MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---|---:|---:|---:|---:|---:|
| Vanilla | none | 0.739 | 0.850 | 0.111 | 0.149 | 0.000 |
| Route A IC target gate | random threshold | 0.745 | 0.850 | 0.106 | 0.142 | 0.000 |
| D-lite hidden margin zero gate | fixed zero | 0.670 | 0.750 | 0.089 | 0.119 | 0.000 |
| **D-lite tuned formula gate** | **random formula/threshold** | **0.751** | **0.850** | **0.100** | **0.134** | **0.000** |
| Route D hidden contrast tuned gate | supervised OOF probe | 0.707 | 0.778 | 0.078 | 0.104 | 0.000 |

Next validation gate: rerun D-lite tuned formula on full POPE, ideally with formula fixed from the bounded/random calibration and no further tuning on popular/adversarial/full rows.

### 2026-06-23 — D-lite tuned formula full-POPE validation
- Setup: Full POPE random/popular/adversarial (9000 rows), LLaVA-1.5-7B, no external detector and no supervised readout. Script: `mitigation/scripts/evaluate_hidden_margin_tdev_pope.py` run in 5 shards, merged under `mitigation/results/semantic_neighbor_audit/hidden_margin_tdev_full/`. The deployed formula is fixed from the bounded/random calibration: `formula_score = hidden_cross_margin - 0.5*hidden_obj_separation - 0.5*hidden_vis_separation`, threshold `-0.3032374829053879`; evaluator: `mitigation/scripts/evaluate_pope_score_subset.py` with vanilla as the base prediction.
- Result: fixed-formula gate reaches Macro MCC 0.730, TPR 0.792, FPR 0.069, related FPR 0.093, plain FPR 0.016, gap 0.077, adversarial related FPR 0.142. Relative to full vanilla in §D.3 (MCC 0.730, TPR 0.813, FPR 0.087, related FPR 0.114, plain FPR 0.028, gap 0.086, adv. related FPR 0.164), it lowers FPR/related-FPR but loses recall and does **not** improve aggregate MCC. The original zero gate is more conservative (MCC 0.653, TPR 0.686, FPR 0.055, related FPR 0.072). Full-set calibrated formula gives an upper-bound operating point (MCC 0.741, TPR 0.811, FPR 0.075, related FPR 0.099), but this uses full POPE labels for threshold selection and should not be reported as the deployable main result.
- Interpretation: the bounded-set D-lite gain does not robustly transfer to full POPE. D-lite remains useful evidence that internal hidden target-vs-neighbor margins can suppress semantic-neighbor false positives without OWLv2, but it is not yet a replacement for the external TDEV verifier in the main table. Current paper-safe position: report it as a no-external-detector ablation/negative boundary, and continue with a stronger integrated internal readout rather than claiming detector removal is solved.
- Supersedes: the bounded-set D-lite tuned formula claim as the preferred deployable candidate; full validation shows it is not a clear main-method improvement.

#### Full POPE no-external-detector D-lite comparison

| Method | Calibration | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | none | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.086 | 0.164 |
| D-lite hidden margin zero gate | fixed zero | 0.653 | 0.686 | 0.055 | 0.072 | 0.016 | 0.056 | 0.120 |
| **D-lite tuned formula gate** | **bounded/random fixed** | **0.730** | **0.792** | **0.069** | **0.093** | **0.016** | **0.077** | **0.142** |
| D-lite tuned formula gate | full-label calibrated upper bound | 0.741 | 0.811 | 0.075 | 0.099 | 0.021 | 0.078 | 0.147 |

Decision: do **not** put fixed D-lite as the last-row main method in the paper table yet. It is a credible internal ablation because it removes the external detector and reduces semantic-neighbor FPR, but the main-method row still needs either (1) a stronger internal verifier that preserves TPR, or (2) a revised objective/table emphasizing FPR-gap reduction with an explicit recall tradeoff.

### 2026-06-23 — Cross-split calibrated internal hidden-margin verifier
- Setup: Full POPE 9000 rows, no external detector. Script: `mitigation/scripts/cross_split_hidden_margin_verifier.py`; input cache: `mitigation/results/semantic_neighbor_audit/hidden_margin_tdev_full/hidden_margin_tdev_predictions.csv`; vanilla base predictions are read from `mitigation/results/coco_llava_7b_attention_only/pope/<split>/vanilla/predictions.jsonl`. The verifier tunes a shared linear score over `[hidden_align_margin, hidden_cross_margin, hidden_obj_separation, hidden_vis_separation]` on two POPE splits and evaluates on the held-out split, then reports the concatenated held-out predictions. This avoids full-label threshold leakage while still using POPE labels for calibration.
- Result: best held-out MCC objective gives Macro MCC 0.741, TPR 0.811, FPR 0.075, related FPR 0.100, plain FPR 0.021, gap 0.079, adversarial related FPR 0.147. Compared with vanilla (MCC 0.730, TPR 0.813, FPR 0.087, related FPR 0.114, plain FPR 0.028, gap 0.086, adv. related FPR 0.164), it preserves recall almost exactly while reducing FPR and the semantic-neighbor FPR gap. The lower-FPR objective with held-out TPR >= 0.80 gives MCC 0.736, TPR 0.802, FPR 0.071, related FPR 0.096, plain FPR 0.017. The TPR >= 0.75 objective gives stronger suppression but too much recall loss: MCC 0.703, TPR 0.751, FPR 0.060, related FPR 0.080.
- Interpretation: after fixing the base-prediction source, cross-split calibration recovers a small but real no-external-detector improvement over vanilla. This is stronger than the bounded fixed-threshold D-lite result, but still weaker than the original OWLv2-backed hybrid gate+rescue (MCC 0.763, FPR 0.051, related FPR 0.069). It should be positioned as the current internal replacement candidate, not yet the final main method. The next step is to add answer-confidence/yes-logit evidence and validate the same shared verifier on CHAIR-style caption claims.
- Supersedes: the fixed bounded-threshold D-lite result as the best current internal no-external-detector candidate.

#### Full POPE internal verifier comparison after cross-split calibration

| Method | Calibration | External detector | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | none | no | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.086 | 0.164 |
| D-lite tuned formula gate | bounded/random fixed | no | 0.730 | 0.792 | 0.069 | 0.093 | 0.016 | 0.077 | 0.142 |
| **Cross-split hidden-margin verifier** | **leave-one-split-out** | **no** | **0.741** | **0.811** | **0.075** | **0.100** | **0.021** | **0.079** | **0.147** |
| Cross-split hidden-margin verifier | held-out TPR >= 0.80 / min FPR | no | 0.736 | 0.802 | 0.071 | 0.096 | 0.017 | 0.079 | 0.143 |
| Hybrid gate+rescue | full control table | yes / OWLv2 | 0.763 | 0.806 | 0.051 | 0.069 | 0.012 | 0.057 | 0.105 |

Current method design: use one shared internal verifier rather than separate scenario-specific rules. For each target object, construct a semantic-neighbor prompt, extract hidden object/visual vectors from layers 22/31, compute target-vs-neighbor alignment/cross/separation margins, then gate only vanilla `yes` answers whose internal target evidence falls below the calibrated threshold. The method is correlated across scenarios through the same score components and calibration objective, not separately hand-designed per table.

### 2026-06-23 — Answer-confidence evidence added to the internal verifier
- Setup: Full POPE 9000 rows, no external detector. New script: `mitigation/scripts/evaluate_answer_confidence_tdev_pope.py`; artifacts: `mitigation/results/semantic_neighbor_audit/answer_confidence_tdev_full/`. For each target and semantic-neighbor prompt, the scorer caches first-answer-token yes/no logits and derived margins (`answer_support_score`, `answer_contrast_margin`, `answer_absence_score`, `answer_neighbor_dominance`). The features are merged with the full hidden-margin cache and evaluated by `mitigation/scripts/cross_split_hidden_margin_verifier.py` using leave-one-split-out calibration and train-fold standardization.
- Result: answer-confidence alone is weaker than hidden-margin evidence: Macro MCC 0.721, TPR 0.796, FPR 0.080, related FPR 0.104. With the TPR >= 0.80 objective it reaches MCC 0.728, TPR 0.808, FPR 0.084. Standardized hidden-only remains MCC 0.741, TPR 0.811, FPR 0.075, related FPR 0.099. Mixed hidden+answer (`hidden_align_margin`, `hidden_cross_margin`, `hidden_obj_separation`, `hidden_vis_separation`, `answer_support_score`, `answer_contrast_margin`) gives the current best internal POPE verifier: Macro MCC 0.742, TPR 0.810, FPR 0.073, related FPR 0.097, plain FPR 0.020, gap 0.077, adversarial related FPR 0.144. The lower-FPR operating point at held-out TPR >= 0.80 gives MCC 0.740, TPR 0.800, FPR 0.067, related FPR 0.089, plain FPR 0.018, adversarial related FPR 0.139.
- Interpretation: answer confidence is a weak but real auxiliary signal; it does not replace hidden target-vs-neighbor evidence. The best no-external-detector method now slightly improves over vanilla and over hidden-only, but the gain is small and still below OWLv2-backed hybrid gate+rescue (MCC 0.763, FPR 0.051, related FPR 0.069). This should be reported as the current internal verifier, not as a solved replacement.
- Supersedes: the hidden-only cross-split verifier as the best current no-external-detector POPE result, by a small margin.

#### Full POPE internal verifier after adding answer confidence

| Method | Calibration | Components | External detector | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | none | none | no | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.086 | 0.164 |
| Answer-confidence verifier | leave-one-split-out | answer yes/no logits | no | 0.721 | 0.796 | 0.080 | 0.104 | 0.026 | 0.079 | 0.159 |
| Hidden-margin verifier | leave-one-split-out + train z-score | hidden target-vs-neighbor | no | 0.741 | 0.811 | 0.075 | 0.099 | 0.021 | 0.079 | 0.147 |
| **Hidden+answer verifier** | **leave-one-split-out + train z-score** | **hidden margins + answer support/contrast** | **no** | **0.742** | **0.810** | **0.073** | **0.097** | **0.020** | **0.077** | **0.144** |
| Hidden+answer verifier | held-out TPR >= 0.80 / min FPR | hidden margins + answer support/contrast | no | 0.740 | 0.800 | 0.067 | 0.089 | 0.018 | 0.071 | 0.139 |
| Hybrid gate+rescue | full control table | OWLv2 target/neighbor boxes | yes | 0.763 | 0.806 | 0.051 | 0.069 | 0.012 | 0.057 | 0.105 |

Decision: the current integrated internal design is coherent and shared across POPE splits, but not yet strong enough to replace the external detector as the paper's final method. Further POPE-only linear tuning is unlikely to change the story; the next validation should move to CHAIR/caption-side atomic claims and check whether the same internal verifier can support caption intervention without OWLv2.

### 2026-06-23 — CHAIR caption-claim internal verifier without external detector
- Setup: Full CHAIR object-mention cache from `detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl` (16,426 generated object mentions; 4,009 hallucinated), LLaVA-1.5-7B, no external detector. New scripts: `mitigation/scripts/evaluate_chair_internal_verifier.py` and `mitigation/scripts/evaluate_chair_verifier_scores.py`; artifacts: `mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full/`. Each CHAIR object mention is treated as an atomic caption claim. The scorer prompts the model with the mentioned object and its semantic neighbor, then caches hidden target-vs-neighbor margins plus first-token yes/no answer confidence.
- Result: unlike POPE, CHAIR caption-claim detection is dominated by answer confidence, not hidden target-vs-neighbor margins. `answer_absence_score = -target_yes_margin` reaches AUROC 0.898 and MCC 0.596 at its full-data threshold (TPR 0.797, FPR 0.155). Image-grouped 5-fold answer-only grid gives AUROC 0.893 and MCC 0.593. Mixed hidden+answer gives AUROC 0.898 and MCC 0.593, essentially no gain over answer absence. Hidden-only is weak: AUROC 0.622 and MCC 0.175. Position-only (`gen_pos`) is lower: AUROC 0.830 and MCC 0.473.
- Position control: `answer_absence_score` remains strong after generation-position control: within-bin AUROC 0.872, matched-pair AUROC 0.859, residual AUROC 0.782. By comparison, `gen_pos` has within-bin AUROC 0.572, matched-pair AUROC 0.546, residual AUROC 0.522. The existing best position-controlled baseline IC is within-bin 0.686 / matched-pair 0.703 (§A), so answer absence is a substantially stronger internal CHAIR detector.
- Interpretation: this is the first strong caption-side no-external-detector result. It also clarifies task structure: POPE semantic-neighbor yes/no gating needs target-vs-neighbor hidden/answer contrast, while CHAIR object-mention detection is mostly solved by direct answer-confidence absence. The next intervention step should use this score to filter or rewrite high-risk caption object claims and rerun official CHAIR; do not keep tuning hidden margins for CHAIR.
- Supersedes: prior expectation that the same hidden-margin verifier would be the main CHAIR signal; answer absence is the CHAIR-side main evidence.

#### CHAIR object-mention detection, no external detector

| Score / verifier | Overall AUROC | Within-bin AUROC | Matched AUROC | Residual AUROC | MCC | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Position only (`gen_pos`) | 0.830 | 0.572 | 0.546 | 0.522 | 0.473 | 0.807 | 0.267 |
| Hidden-only verifier | 0.622 | - | - | - | 0.175 | 0.839 | 0.652 |
| `answer_absence_score` | **0.898** | **0.872** | **0.859** | **0.782** | **0.596** | 0.797 | 0.155 |
| Answer-only 5-fold verifier | 0.893 | - | - | - | 0.593 | 0.790 | 0.153 |
| Mixed hidden+answer 5-fold verifier | 0.898 | 0.872 | 0.859 | 0.781 | 0.593 | 0.792 | 0.154 |

Decision: for CHAIR/caption claims, the deployable internal detector should be `answer_absence_score` or the answer-only verifier, not hidden-margin TDEV. For POPE, keep hidden+answer; for CHAIR, use direct target yes/no absence. This is an integrated method family with task-specific heads over internal evidence, still with no external detector.


### 2026-06-23 — Answer-absence CHAIR caption intervention
- Setup: Full CHAIR object-mention cache from `detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl`, using the no-external-detector `answer_absence_score` from `chair_internal_verifier_scores.csv`. New script: `mitigation/scripts/evaluate_chair_answer_caption_intervention.py`. The intervention selects the top-scoring object mentions, edits the caption deterministically (`delete` or `generic_noun`), and reruns the official CHAIR evaluator on the same 4,977-image caption set.
- Result: the score selects hallucinated mentions with high precision. Top-1% delete changes 163 mentions with 0.859 precision and reduces CHAIRi from 0.1340 to 0.1309. Top-10% delete gives the largest CHAIR reduction: CHAIRi 0.1340 -> 0.1074 and CHAIRs 0.4921 -> 0.4099, changing 1,195 hallucinated mentions and 430 grounded mentions. Generic noun rewriting also improves CHAIR while preserving caption length better, but the CHAIR reduction is smaller than deletion.
- Interpretation: caption-side intervention is feasible without OWLv2 or any external detector. The cleanest result is not a new decoder yet; it is a post-hoc claim filter/rewrite driven by internal answer absence. Deletion is strongest on CHAIR but has a clear faithfulness/coverage cost because some grounded object mentions are removed. Generic rewriting is a more conservative presentation candidate if caption completeness matters.
- Supersedes: the previous CHAIR detection-only result by adding official CHAIR intervention metrics.

#### CHAIR caption intervention from internal answer-absence score

| Method | Top frac | Selected precision | Changed hall. | Changed grounded | CHAIRi | CHAIRs | Mean object mentions | Mean hall. mentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | - | - | - | - | 0.1340 | 0.4921 | 7.641 | 1.024 |
| Delete high-risk mention | 1% | 0.860 | 140 | 23 | 0.1309 | 0.4786 | 7.608 | 0.996 |
| Delete high-risk mention | 5% | 0.769 | 620 | 190 | 0.1204 | 0.4467 | 7.480 | 0.901 |
| **Delete high-risk mention** | **10%** | **0.738** | **1195** | **430** | **0.1074** | **0.4099** | **7.317** | **0.786** |
| Generic rewrite | 5% | 0.769 | 595 | 188 | 0.1250 | 0.4639 | 7.532 | 0.941 |
| Generic rewrite | 10% | 0.738 | 1164 | 424 | 0.1133 | 0.4322 | 7.383 | 0.836 |

Decision: use answer-absence filtering as the CHAIR-side no-external-detector intervention baseline. For paper positioning, report deletion as an upper-bound CHAIR reduction and generic rewrite as the less destructive variant; do not claim it fully solves caption intervention until semantic/fluency preservation is evaluated beyond CHAIR.

### 2026-06-23 — Utility-aware CHAIR intervention selection
- Setup: Cache-only utility summary over the five completed answer-absence CHAIR intervention runs. New script: `mitigation/scripts/summarize_chair_intervention_utility.py`; artifacts: `mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full/intervention_utility_summary.{csv,json}`. Metrics include token-LCS caption retention, changed-caption rate, CHAIR Recall/F1 deltas, object-mention retention, hallucinated-mention retention, and a preservation gate (`mean_lcs_retention >= 0.995`, `object_mention_retention >= 0.965`, `recall_delta >= -0.011`).
- Result: the CHAIR-only optimum is still delete top-10 (CHAIRi 0.1074; reduction 0.0265), but it fails the preservation gate because LCS retention is 0.9946, object retention is 0.9577, and Recall drops by 0.0127. Under the preservation gate, the best CHAIRi reduction is generic-rewrite top-10: CHAIRi 0.1133, CHAIRs 0.4322, LCS retention 0.9958, object retention 0.9663, Recall delta -0.0104, F1 delta +0.0127.
- Interpretation: this changes the recommended main CHAIR intervention row. Delete top-10 is the upper-bound hallucination-removal row; generic-rewrite top-10 is the better paper-facing no-external-detector intervention because it retains more caption/object coverage while still reducing CHAIRi by 15.4% relative and CHAIRs by 12.2% relative. This directly addresses the concern that the earlier CHAIR gain could be mostly aggressive deletion.
- Supersedes: the previous CHAIR intervention ranking if utility preservation is part of the objective.

#### Utility-aware CHAIR intervention comparison

| Method | Top frac | CHAIRi | CHAIRi red. | CHAIRs | LCS retention | Object retention | Recall delta | Pass gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Delete | 1% | 0.1309 | 0.0031 | 0.4786 | 0.9995 | 0.9958 | -0.0006 | yes |
| Delete | 5% | 0.1204 | 0.0135 | 0.4467 | 0.9974 | 0.9790 | -0.0052 | yes |
| Delete | 10% | **0.1074** | **0.0265** | **0.4099** | 0.9946 | 0.9577 | -0.0127 | no |
| Generic rewrite | 5% | 0.1250 | 0.0090 | 0.4639 | 0.9981 | 0.9859 | -0.0035 | yes |
| **Generic rewrite** | **10%** | **0.1133** | **0.0207** | **0.4322** | **0.9958** | **0.9663** | **-0.0104** | **yes** |

Decision: for the main no-external-detector CHAIR intervention result, prefer generic-rewrite top-10 as the utility-aware operating point. Keep delete top-10 as an upper-bound ablation that shows the maximum CHAIR reduction achievable by aggressive filtering.

### 2026-06-23 — Consolidated no-external-detector main rows
- Setup: Artifact-only consolidation with `mitigation/scripts/build_no_external_detector_summary.py`; tracked summary: `docs/no_external_detector_summary.md`; ignored machine-readable output: `mitigation/results/semantic_neighbor_audit/no_external_detector_summary.json`. The script reads the POPE control table, POPE hidden+answer cross-split verifier summary, CHAIR answer-only detector metrics, CHAIR position-control metrics, and CHAIR intervention utility summary.
- Result: the current no-external-detector method family is fixed as follows. POPE uses the leave-one-split-out hidden+answer verifier (best-MCC: MCC 0.742, TPR 0.810, FPR 0.073, related FPR 0.097; stricter FPR row: MCC 0.740, TPR 0.800, FPR 0.067, related FPR 0.089). CHAIR detection uses `answer_absence_score` (AUROC 0.898, within-bin 0.872, matched 0.859, residual 0.782, MCC 0.596). CHAIR intervention uses utility-preserved generic rewrite top-10 (CHAIRi 0.113, CHAIRs 0.432, LCS retention 0.9958, object retention 0.9663). OWLv2 hybrid gate+rescue remains only an external positive control, not the main no-external method.
- Interpretation: this is the current answer to "which internal path is best after removing the external detector." It is not a claim that the internal verifier matches OWLv2; POPE still trails the external positive control. It does, however, provide a coherent no-external-detector table across POPE gating, CHAIR detection, and CHAIR intervention.
- Supersedes: scattered per-experiment selection notes for deciding which row should be treated as the current no-external-detector main result.
