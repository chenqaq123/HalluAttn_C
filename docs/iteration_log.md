# Iteration Log

> Append-only. One entry per completed iteration (a decision, pivot, or
> implementation milestone). Newest at the bottom. Each entry records the date,
> what changed, why, and what it implies for the proposal / next steps.

<!-- Template:
## YYYY-MM-DD — <title>
**What changed:** ...
**Why:** ...
**Evidence/refs:** ...
**Implications / next:** ...
-->

## 2026-06-23 — Repositioning after discovering HaloProbe

**What changed:**
- Established three living docs (`proposal.md`, `experiment_results.md`,
  `iteration_log.md`) + root `CLAUDE.md` as the steering documents.
- Demoted the position-confound chapter and the per-head diagnostic from
  headline to background.
- Promoted the **semantic-neighbor stress test + target-discriminative (TDEV)
  criterion** to the single headline.
- Decided to **remove the external OWLv2 detector** from the headline method and
  replace it with an internal, self-contrastive target-vs-neighbor readout.

**Why:**
- Literature check found **HaloProbe** (arXiv:2604.06165, Apr 2026; A./M.
  Rohrbach among authors — original CHAIR authors) already establishes
  position as a confounder for attention-based detection (framed via Simpson's
  paradox), uses a per-head attention+confidence probe (93.5 AUROC), adds object
  repetition as a second confounder, and evaluates on 5 models. This pre-empts
  two of our three pillars.
- PARALLAX (arXiv:2605.17028) similarly works on benchmark artifacts in
  hallucination detection.
- Our OWLv2-backed TDEV invites the "why not just use OWLv2 / this is
  Woodpecker-lite" critique and carries the main latency cost.

**Evidence/refs:**
- HaloProbe overlap analysis and full read recorded in this session.
- Semantic-neighbor results show all attention/VCD methods leave an unclosed
  related-minus-plain FPR gap (see `experiment_results.md` §D).

**Implications / next:**
1. Implement internal self-contrastive TDEV. Primary = **B (attention-region
   discriminability)**; cheap baseline = **A (IC/logit-lens target-vs-neighbor
   margin)**; supervised ceiling = **C (contrastive per-head probe)**. See
   `proposal.md` §3.
2. First validation experiment: replace OWLv2 with IC target-vs-neighbor margin
   on the related-present negative subset; check whether the FPR gap closes.
3. Run a HaloProbe-style internal probe on the related-present subset to show it
   leaves the gap open (turn the threat into a foil).
4. Add object-repetition control to detection eval (free, borrowed from HaloProbe).
5. Add ≥1 modern model (Qwen-VL / InternVL) for the semantic-neighbor headline.
6. Borrow HaloProbe's *explanation framing* (Simpson's paradox; internal vs.
   external factorization) for the intro — not its experiments.

## 2026-06-23 — Route A internal TDEV baseline implemented and rejected as headline
**What changed:** Implemented `evaluate_internal_ic_tdev_pope.py` for no-external-detector IC/logit-lens target-vs-neighbor evidence, plus `evaluate_pope_score_subset.py` for bounded direct/gate evaluation.

**Why:** The proposal requires replacing OWLv2 with internal self-contrastive evidence. Route A is the lowest-cost test because it reuses one LLaVA forward and the LM head.

**Evidence/refs:** `docs/experiment_results.md` entry "Internal IC/logit-lens TDEV route A". On 360 bounded POPE rows, strict IC margin gate reduces related FPR from 0.149 to 0.022 but collapses TPR from 0.850 to 0.267; calibrated target-only gate barely changes vanilla.

**Implications / next:** Do not use route A as the main method. Keep it as a cheap baseline/ablation. Next implementation should prioritize route B (target-vs-neighbor attention-region discriminability) and route C/HaloProbe-style supervised contrastive probe on the related-present subset.

## 2026-06-23 — Routes B/C internal TDEV implemented and rejected as main method
**What changed:** Implemented `evaluate_attention_region_tdev_pope.py` for target-vs-neighbor attention-region discriminability and `evaluate_contrastive_head_probe_pope.py` for a HaloProbe-style supervised contrastive per-head internal probe. Both run without an external detector and were evaluated on the same 360-row bounded POPE subset as route A.

**Why:** The proposal required testing whether internal VLM evidence can replace the OWLv2-backed verifier while preserving the semantic-neighbor TDEV criterion.

**Evidence/refs:** `docs/experiment_results.md` entries for routes B/C. Route B direct is poor (MCC 0.173, FPR 0.378), and calibrated gate is unchanged from vanilla. Route C has real internal signal (absent AUROC 0.731; direct MCC 0.419), but the useful strict gate trades too much recall for FPR reduction (TPR 0.567, FPR 0.067, related FPR 0.090). Calibrated gate again chooses no suppression.

**Implications / next:** A/B/C should be reported as internal baselines/negative controls, not the headline. The current main claim remains the semantic-neighbor failure mode and target-vs-neighbor verification criterion. To remove the external detector, the next implementation should either move beyond attention-shape features to stronger hidden-state/logit contrast, or pivot to caption-side atomic claim verification where local accept/reject decisions may be easier than a global POPE gate.

## 2026-06-23 — Route D hidden-state contrast becomes the best internal candidate
**What changed:** Implemented `evaluate_hidden_contrast_probe_pope.py`, a no-external-detector hidden-state target-vs-neighbor contrast probe. It uses object-token hidden states, visual-token mean hidden states, and target-minus-neighbor contrasts from layers 22/31, with image-grouped OOF logistic evaluation. Added `fixed` threshold support to `evaluate_pope_score_subset.py` to save tuned gate operating points.

**Why:** Routes A/B/C showed that final-layer logit-lens evidence and attention-shape features were not strong enough. The next plausible internal reference path was richer hidden-state contrast rather than more attention aggregation.

**Evidence/refs:** `docs/experiment_results.md` entry "Hidden-state contrast probe route D". On the same 360 bounded POPE rows, route D reaches intrinsic absent AUROC 0.863 and tuned gate MCC 0.707, TPR 0.778, FPR 0.078, related FPR 0.104. This is weaker than vanilla/A in aggregate MCC but a much better recall/FPR tradeoff than route C strict gating.

**Implications / next:** Route D should become the current internal replacement candidate. It still is supervised and not final. Next work should tune layer/feature choices and derive a smaller deployable score, then re-run on the full POPE subset and check whether the semantic-neighbor FPR gap closes without unacceptable recall loss.

## 2026-06-23 — D-lite training-free hidden margin implemented
**What changed:** Implemented `evaluate_hidden_margin_tdev_pope.py`, a deployable no-external-detector and no-supervised-readout hidden-margin scorer. It computes target-vs-neighbor hidden alignment/cross-prompt/separation margins and uses the score as a gate over vanilla.

**Why:** Route D showed strong supervised signal but was not a final method. The next requirement was to distill that signal into a training-free score that could plausibly replace the external detector.

**Evidence/refs:** `docs/experiment_results.md` entry "Training-free hidden-margin D-lite". On the 360 bounded POPE rows, zero-threshold D-lite gate reaches MCC 0.670, TPR 0.750, FPR 0.089, related FPR 0.119, plain FPR 0.000. This is weaker than supervised route D but better than route A at reducing related-present FPR without collapsing recall.

**Implications / next:** D-lite is the current best deployable internal candidate, but not sufficient for final claims. Next work should tune its formula/layers against the supervised D ceiling, run full POPE, and only then decide whether external detector removal is viable for the main method.

## 2026-06-23 — D-lite formula sweep gives the best deployable internal candidate
**What changed:** Added `tune_hidden_margin_formula.py`, a cache-only formula/threshold sweep for D-lite hidden-margin components. The script now only runs gate sweeps when a real vanilla `base_prediction` CSV is supplied, avoiding accidental self-gating.

**Why:** The initial D-lite zero gate lowered related FPR but lost too much recall. We needed a bounded, calibration-only way to tune the training-free score without rerunning the VLM.

**Evidence/refs:** `docs/experiment_results.md` entry "D-lite cache-only formula sweep". Using random split calibration, the best TPR>=0.85 gate uses weights `[0, 1, -0.5, -0.5]` and threshold `-0.3032`, reaching MCC 0.751, TPR 0.850, FPR 0.100, related FPR 0.134 on the 360-row bounded set.

**Implications / next:** This becomes the current best deployable, no-external-detector candidate. The improvement over route A is small, so the next required step is full-POPE validation with this formula fixed; if it holds, external-detector removal becomes plausible for the main method.
