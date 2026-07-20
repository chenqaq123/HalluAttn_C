# Pair-Centered Preference Alignment: Route Audit

Date: 2026-07-20
Status: working literature boundary for PCPA iteration 1

## Decision

Replace the unstarted Existential Evidence Aggregation route with direct
pairwise preference alignment, but do not use a conventional faithful-response
versus hallucinated-response pair. That space is already mature. The active
unit is one unchanged image with two related object queries:

- preferred: `Yes` for an annotated visible object and `No` for a related
  absent object;
- rejected: the two answers reversed.

The joint preference contains one `Yes` and one `No` on each side. PCPA adds an
explicit zero-common-mode penalty on the reference-relative answer margins so
the optimization cannot explain a gain by globally suppressing or amplifying
`Yes`.

## Closest primary work

### FINER (CVPR 2026)

FINER identifies hallucination under fine-grained negative queries, especially
when mismatched and genuinely present elements co-occur, and proposes DPO-based
FINER-Tuning. It reports gains on its new benchmarks and existing hallucination
suites.

Source: https://arxiv.org/abs/2603.17662

Boundary: related co-occurrence, fine-grained negative questions, hard-negative
data, and DPO are occupied. PCPA must contribute through the two-query joint
outcome and constrained common mode, not through semantic-neighbor mining.

### mDPO (EMNLP 2024)

mDPO shows that multimodal DPO can learn an unconditional language preference:
removing images during DPO can produce similar behavior. It adds a conditional
image-preference objective, created by pairing the original image with a weakly
cropped image, and a chosen-reward anchor.

Source: https://arxiv.org/abs/2406.11839

Boundary: image-conditioned preference and absolute anchoring are occupied.
PCPA never changes the image and does not claim an anchor. It asks whether
answer assignment between two target identities on the same pixels can improve
while the average answer-margin update is constrained.

### OPPO (2026)

OPPO's Faithful Preference Optimization keeps one faithful answer fixed and
orders its reference-relative likelihood under an attention-warped strong view,
the original anchor view, and a stochastically degraded weak view. Its premise is
that semantic content remains invariant while evidence strength changes.

Source: https://arxiv.org/abs/2606.29805

Boundary: strong/anchor/weak evidence ordering is occupied and depends on view
construction. PCPA borrows only the training-time preference perspective. It
does not assume that an attention-selected crop is stronger evidence, which is
especially unsafe given SinkDetect's measured related-negative `Yes` shift.

### Object-aware and response-level hallucination DPO

CHAIR-DPO selects non-hallucinated answers over hallucinated answers using
CHAIR-derived preference labels. HA-DPO, HALVA, and related methods likewise
establish faithful-response preference tuning as a standard mitigation family.

Source: https://arxiv.org/abs/2508.20181

Boundary: PCPA cannot be presented as the first object-aware preference method
or as evidence that DPO reduces hallucination. Its rejected item is a jointly
flipped answer assignment across two matched prompts, not a generated caption.

### DCD and likelihood displacement

DCD argues that pairwise DPO can reduce general reasoning capability through
likelihood displacement and replaces pairwise optimization with decoupled
positive/negative projections used for contrastive decoding.

Source: https://arxiv.org/abs/2504.08809

Boundary: pairwise optimization carries a real utility risk. PCPA's balanced
sign anchor and common-mode constraint target the narrower binary answer-prior
failure; they do not establish general-capability preservation. Confirmation
must include utility evaluation if the first POPE gate passes.

## Method boundary

For pair `(I, t+, t-)`, define the answer margin `m_theta(t,I)` and its update
from the frozen reference `Delta(t,I)`. PCPA optimizes

    separation = Delta(t+,I) - Delta(t-,I)
    common     = (Delta(t+,I) + Delta(t-,I)) / 2.

The preference term increases `separation`; the centering term constrains
`common` to zero; a balanced sign anchor requires the visible and absent margins
to finish on opposite sides. The proposed scientific claim is not that this is
a new generic preference loss. It is that **common-mode answer-prior movement is
the wrong degree of freedom for object verification, and eliminating it inside
a same-image semantic pair improves related-negative discrimination**.

## Required falsification

The method claim is rejected if any of the following occurs:

1. balanced answer SFT or ordinary per-query DPO matches full PCPA;
2. removing the centering term does not change common-mode drift or behavior;
3. gains survive image removal or image permutation, indicating a category-name
   shortcut;
4. related-absent FPR falls only through a global `No` shift, TPR loss, or
   yes-rate collapse;
5. gains do not transfer to categories excluded from both pair roles during
   training.

## Search limitation

This audit establishes a defensible working boundary against the closest known
primary papers as of the date above; it is not a formal novelty or priority
guarantee. A broader citation and concurrent-work audit is required before paper
submission.
