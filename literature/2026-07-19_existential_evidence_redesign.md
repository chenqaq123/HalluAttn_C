# Constructive Route Audit: From Relative Allocation to Existential Evidence

Date: 2026-07-19
Status: active design audit; no new model result

## Decision

Cancel Null-Evidence Attention (NEA) before implementation. Although NEA places
its learned null branch inside an attention head, its behavioral role is still
to form an absence-conditioned correction to the ordinary answer. That is too
close to contrastive/calibrated decoding to support a method contribution.

The replacement hypothesis is **Existential Evidence Aggregation (EEA)**. It
does not create a second image, second answer distribution, or inference-time
subtraction. It treats object existence as a set predicate and uses a
training-time auxiliary objective over unnormalized target-patch compatibility.
The auxiliary head is not used to produce deployment answers.

This is a hypothesis, not a novelty or effectiveness claim. The first iteration
is a frozen-feature mechanism test with explicit kill criteria.

## Why the intermediate alternatives were rejected

### Learned null evidence

NEA preserved the pretrained visual-attention mass and mixed its conditional
visual value with a learned null value. This was architecturally cleaner than
image/no-image contrast, but the output still had the form "ordinary evidence
minus an absence-conditioned correction." Generic null/sink attention is also
established. Moving a correction inside the head does not change the functional
method family enough.

### Signed visual Jacobian training

A local derivative of the Yes/No margin with respect to visual-path strength
would directly measure the project's amplification finding. However, OPPO
already trains ordered strong/anchor/weak evidence preferences and explicitly
derives a positive lower bound on local visual sensitivity. A signed derivative
would remove finite views, but its headline remains evidence-strength
sensitivity. That boundary is too narrow, and second-order training on Qwen3-VL
would be expensive.

### Object slots or one-to-one patch ownership

Cross-target competition could expose one patch being claimed by incompatible
objects, but object-centric slot tokenization is already an active MLLM route.
It also creates a candidate-universe and assignment problem before showing that
the underlying patch states contain usable identity evidence.

### Query-conditioned visual feedback

The causal ordering of image tokens before query tokens motivates a late
query-conditioned visual refresh. Recent work already covers visual perception
tokens, query-aware high-resolution reprocessing, and decoder-boundary
query-guided visual residuals. A feedback adapter would therefore be another
query-aware focusing mechanism, still vulnerable to the project's finding that
better focus is not target verification.

### Visual-token logit reading or injection

The CVPR 2026 paper *Same Attention, Different Truths* applies a logit lens to
high-attention visual-token states, defines semantic consistency, masks harmful
visual tokens, and injects visual-token logits. Decoding the attended patch into
a word and using that word-level agreement is no longer open.

## Remaining operator gap

For a target `t` and visual patch states `{v_i}`, softmax attention computes a
relative distribution:

    a_i = softmax_i(score(t, v_i)).

This distribution sums to one regardless of whether any patch is a good target
match. In a bathroom without a sink, a toilet patch can therefore win the same
relative competition that a real sink would win in another image. Attention
amplification strengthens the winner but cannot determine whether the winner
clears an absolute semantic boundary.

Object existence has different logic:

    exists(t, I) = OR_i match(t, v_i).

EEA parameterizes a shared open-vocabulary match logit `s_i(t,I)` and uses a
smooth, patch-count-normalized maximum:

    E(t,I) = tau * logmeanexp_i(s_i(t,I) / tau).

The target representation comes from its input embedding before multimodal
fusion. The patch representation comes from native Qwen3-VL visual states. No
box, category-specific classifier, candidate generator, or attention weight is
needed. The same-image target-swap control holds the entire patch set fixed and
asks whether `E` ranks a visible object above a related absent object.

## Proposed training use

Stage A trains only the shared patch matcher on natural image-level labels. It
must beat parameter-matched relative/global pooling controls on categories never
used as training targets.

Only after Stage A passes, a separate iteration may train identical LoRA
adapters with either:

    L_answer

or:

    L_answer + lambda_set * L_EEA + lambda_agree * L_agree.

The EEA head is training-only. Behavior evaluation uses the model's ordinary
single-forward answer logits. This makes answer-only SFT, not contrastive
decoding, the primary baseline.

## Novelty boundary

The following are background, not contributions:

- images-as-bags and multiple-instance learning;
- weakly supervised object localization from image-level labels;
- patch-text matching and smooth maximum aggregation;
- open-vocabulary detection as a task;
- auxiliary grounding losses in multimodal training.

The only defensible intended claim is their use as an **absolute set-existence
objective inside an LVLM hallucination study**, together with evidence that:

1. it separates present targets from related absent targets on unseen classes;
2. it beats relative attention-style aggregation under the same matcher;
3. its training signal improves ordinary one-pass behavior beyond answer-only
   SFT and matched auxiliary heads.

If any one of these is missing, EEA is only a small internal detector and should
be rejected rather than renamed.

## Closest primary work checked

- [OPPO: Clearer Sight, Fewer Lies (2026)](https://arxiv.org/abs/2606.29805):
  strong/anchor/weak visual-view preference optimization and local visual
  sensitivity.
- [Same Attention, Different Truths (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.pdf):
  visual-token logit lens, semantic-consistency detection, masking, and visual
  logit injection.
- [CausalLens (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.pdf):
  head-level visual sensitivity and single-pass visual-path amplification.
- [Slot-MLLM (2025)](https://arxiv.org/abs/2505.17726): object-centric visual
  tokenization with slots.
- [Visual Perception Token (2025)](https://arxiv.org/abs/2502.17425): model-
  controlled region selection and visual re-encoding.
- [TSR-MLLM (2026)](https://arxiv.org/abs/2607.04149): query-conditioned
  decoder-boundary visual focus with single-pass inference.
- [Language-Prior-Free Verification (2026)](https://arxiv.org/abs/2601.22451):
  object validation over sampled candidate captions followed by caption
  selection or aggregation.
- [Weakly Supervised Object Localization with Multi-fold MIL (2015)](https://arxiv.org/abs/1503.00949):
  image-level presence supervision and image-as-bag MIL background.

## Immediate decision rule

Run only `docs/iteration_protocols/eea_iteration_0.md`. Do not implement LoRA,
logit correction, caption reranking, or an open-ended candidate loop in this
iteration. A failed frozen-feature set-operator gate closes EEA.
