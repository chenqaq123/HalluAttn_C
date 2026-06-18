# TDEV Positioning Against Detector-Based Hallucination Work

Date: 2026-06-18

This note records the current positioning after checking detector/tool-based
hallucination work. The goal is to avoid framing TDEV as "add an external
detector to an LVLM", which is already a crowded and less interesting space.

## What Existing Work Already Covers

| Work family | Representative work | What it already claims | Implication for us |
|---|---|---|---|
| Tool-chain correction | Woodpecker, arXiv:2310.16045 | Extract concepts, ask questions, validate with expert visual models, then correct the response. It explicitly uses open-set object detection for object-level validation. | We should not sell TDEV as a generic post-hoc correction pipeline or as detector-backed fact checking. |
| Unified tool-based detection | UNIHD/MHaluBench, arXiv:2402.03190 | Uses a suite of auxiliary tools to validate hallucinations across categories. | A broad external-tool detector is not novel enough. |
| Revision from statistical factors | LURE, arXiv:2310.00754 | Uses co-occurrence, uncertainty, and generation position to revise hallucinated captions. | Our co-occurrence/semantic-neighbor analysis must be target-discriminative, not just another co-occurrence prior. |
| Grounding as training signal | "Does Object Grounding Really Reduce Hallucination?", arXiv:2406.14492 | Finds grounding objectives have little to no effect on open-generation object hallucination under a stricter protocol. | This supports our "grounding is not verification" framing, but also means we must show more than box availability. |
| Context/semantic evaluation | CAOS, arXiv:2501.15046 | Uses object statistics, semantic relationships, generated captions, and LVLM-based verification to evaluate hallucinations. | Semantic relationships are already used for evaluation; our novelty must be the target-vs-neighbor stress protocol and decision rule, not just semantic grouping. |
| Global/local similarity detectors | GLSim, arXiv:2508.19972 | Combines global and local embedding similarity for training-free object hallucination detection. | We need controlled semantic-neighbor evidence showing that target-discriminative margins beat generic similarity scores. |
| Pre-generation/internal probes | HALP, arXiv:2603.05465 | Probes internal VLM states before generation and reports strong AUROC across modern VLMs. | A practical ICML version of TDEV can use late-query/internal probes, but should still require semantic-neighbor discrimination. |
| Fine-grained grounding detectors | Fine-Grained Token Grounding, arXiv:2604.04863 | Detects token hallucinations through patch-level localization and semantic alignment. | Our method should distinguish localized related-object evidence from target evidence; localization alone is not enough. |
| Caption/head steering | CAI/CAST, arXiv:2506.23590 and arXiv:2605.04641 | Uses caption-query attention patterns or steering directions to reduce hallucination with low inference cost. | These are important head-specific baselines or inspirations; TDEV should require target margin for any steered visual evidence. |
| Region-aware attention recalibration | Region-Aware Attention Recalibration, arXiv:2605.24957 | Uses inter-head disagreement across regions for training-free attention correction. | Closest recent mitigation direction; we need to compare conceptually and, if possible, test whether it passes related-present negatives. |
| Internal decoding/attention methods | VCD, OPERA, SPIN, DAMRO | Try to reduce language priors, over-trust, or attention/head failures without external detectors. | These are the right behavioral baselines for showing semantic-neighbor failures. |

## Differentiation That Still Looks Publishable

TDEV should be framed as a **diagnostic criterion and decision rule**, not as a
detector wrapper.

The core claim:

> A visual claim is grounded only when the image provides stronger evidence for
> the target object than for plausible semantic neighbors. Region evidence is an
> implementation vehicle; target-vs-neighbor discrimination is the contribution.

This separates us from existing detector-heavy work in five ways:

1. **Failure mode:** existing work often asks whether an object can be grounded
   or counted. We isolate the harder semantic-neighbor case where related
   objects are present and can support the wrong claim.
2. **Decision criterion:** existing tool pipelines validate target presence.
   TDEV compares target evidence against semantic alternatives and reports
   related-minus-plain FPR as a primary metric.
3. **Asymmetric behavior:** for POPE-style yes/no decisions, TDEV acts mainly as
   a verifier for unsafe positive claims, with a stricter rescue path for no
   answers. This is not a symmetric detector threshold.
4. **Localized but discriminative evidence:** recent fine-grained grounding and
   region-aware attention work makes localization a crowded claim. TDEV must
   show that the localized evidence supports the target more than plausible
   alternatives.
5. **Cross-task score:** for CHAIR object mentions, TDEV becomes a continuous
   target-vs-neighbor hallucination score. The current neighbor-dominance score
   improves residual AUROC, which is a different role from caption rewriting.

## Current Evidence Supporting This Framing

- External-style raw target detection is strong but insufficient on POPE:
  OWLv2 target direct reaches macro MCC 0.777 but adversarial related-present
  FPR is 0.281.
- Semantic-aware two-stage verification improves the related-present tradeoff:
  direct two-stage lowers adversarial related-present FPR to 0.167 and improves
  adversarial MCC to 0.701.
- Hybrid asymmetric verification is currently the best POPE method-shaped
  result: macro MCC 0.763, macro FPR 0.051, adversarial MCC 0.717, and
  adversarial related-present FPR 0.105.
- Hybrid calibration is stable over representative settings: macro MCC remains
  0.763-0.765.
- CHAIR post-hoc detection confirms that this is not POPE-only. The hybrid MCC
  positive branch reaches 0.874 overall AUROC, 0.853 within-bin AUROC, and
  0.855 matched-pair AUROC. The continuous target-absence plus 0.25 neighbor
  dominance score raises residual AUROC from 0.711 to 0.722.

## What We Should Avoid

- Do not pitch OWLv2 itself as the method.
- Do not claim generic visual grounding solves hallucination.
- Do not make the contribution a high-latency correction pipeline like
  Woodpecker.
- Do not rely only on POPE aggregate accuracy or F1; that would erase the
  semantic-neighbor failure mode.
- Do not overclaim mitigation until the method is evaluated on caption-style
  generation or object-mention detection controls.

## Recommended Next Steps

1. **Detector-agnostic abstraction:** rename method language from "OWLv2 TDEV"
   to "TDEV with a proposal/evidence backend". Treat OWLv2 as the strongest
   current backend and CLIP/attention backends as negative or efficiency
   baselines.
2. **Semantic-neighbor ablation table:** report raw target score, target-minus-
   neighbor margin, two-stage rule, hybrid rule, and neighbor-dominance score
   side by side on POPE and CHAIR.
3. **Practicality path:** emphasize cached image-object scores and explore a
   cheaper backend only after the criterion is stable. A weaker backend is
   acceptable as an efficiency ablation, not the main claim.
4. **Related-work contrast:** compare against Woodpecker/UNIHD as
   tool-validation pipelines, LURE/CAOS as co-occurrence or semantic evaluation,
   GLSim/fine-grained grounding as similarity or localization detectors,
   HALP as an internal-probe baseline, and grounding-objective work as evidence
   that grounding alone is insufficient.
5. **Attention baseline priority:** CAI/CAST and region-aware attention
   recalibration are the most relevant recent positive counterexamples. If code
   is available, run a subset semantic-neighbor audit; otherwise discuss them as
   closely related head/region-selection methods and keep the claim scoped.
6. **Paper thesis:** "Looking is not verifying" should become "Grounding is also
   not enough unless it is target-discriminative under semantic-neighbor
   controls."

## Sources Checked

- Woodpecker: Hallucination Correction for Multimodal Large Language Models,
  https://arxiv.org/abs/2310.16045
- Unified Hallucination Detection for Multimodal Large Language Models,
  https://arxiv.org/abs/2402.03190
- Analyzing and Mitigating Object Hallucination in Large Vision-Language Models
  (LURE), https://arxiv.org/abs/2310.00754
- Does Object Grounding Really Reduce Hallucination of Large Vision-Language
  Models?, https://arxiv.org/abs/2406.14492
- Evaluating Hallucination in Large Vision-Language Models based on
  Context-Aware Object Similarities, https://arxiv.org/abs/2501.15046
- GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity,
  https://arxiv.org/abs/2508.19972
- HALP: Detecting Hallucinations in Vision-Language Models without Generating a
  Single Token, https://arxiv.org/abs/2603.05465
- Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector
  of LVLM Hallucinations, https://arxiv.org/abs/2604.04863
- CAI: Caption-Sensitive Attention Intervention for Mitigating Object
  Hallucination in Large Vision-Language Models, https://arxiv.org/abs/2506.23590
- CAST: Mitigating Object Hallucination in Large Vision-Language Models via
  Caption-Guided Visual Attention Steering, https://arxiv.org/abs/2605.04641
- Mitigating Object Hallucinations in Vision-Language Models through
  Region-Aware Attention Recalibration, https://arxiv.org/abs/2605.24957
- Mitigating Object Hallucinations in Large Vision-Language Models through
  Visual Contrastive Decoding, https://arxiv.org/abs/2311.16922
- OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via
  Over-Trust Penalty and Retrospection-Allocation,
  https://arxiv.org/abs/2311.17911
