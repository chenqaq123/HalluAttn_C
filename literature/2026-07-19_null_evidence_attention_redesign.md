# Route Redesign: Null Evidence Instead of Image Counterfactuals

Date: 2026-07-19

## Decision

Cancel Target-Causal Selectivity (TCS) before implementation and replace it with
Null-Evidence Attention (NEA), subject to a routing-feasibility gate.

TCS was experimentally testable but not a sufficiently clean method. The
four-view labels already encode most of target-removal sensitivity and neighbor-
removal invariance, so ordinary counterfactual SFT could absorb the proposed
regularizer. Inpainting then adds a large shortcut and causal-validity burden
without guaranteeing a distinct mechanism. The user's rejection is therefore
treated as a design failure, not as a request to rename or lightly modify TCS.

## Constraints from Our Evidence

The replacement must respect five results already established in this project:

1. Mean-over-head attention shape is not an object verifier after position
   control.
2. Increasing visual-attention mass moves answers toward `Yes`, including on
   related-present negatives.
3. A target-conditioned region can contain another real object, so local
   identity conflict does not prove image-level absence.
4. Asking Qwen3-VL to judge its own boxes/crops preserves the same semantic
   prior and unstable label boundary.
5. Sparse intervention counts can create nominal aggregate gains that disappear
   immediately under confirmation.

Consequently, the new method may use attention as a computation substrate, but
may not read its magnitude as evidence, amplify its visual mass, or feed its
derived view back to the same model for another semantic vote.

## Updated Literature Boundary

### Routes now explicitly occupied

- **Object-Presence Verification (OPV)** (2026),
  <https://openreview.net/pdf?id=VcjFWt5TfO>, trains layer-wise linear probes at
  object-completion boundaries and rolls generation back when a calibrated
  grounding score is low. A hidden-state detector plus veto is therefore not an
  open contribution.
- **PostAlign** (ICLR 2026),
  <https://openreview.net/forum?id=zJnPyb2xrp>, adds `<LOC>`/`<REJ>` outputs and
  a visual grounding encoder plus mask/box decoder. Explicit rejection through
  a separate grounding module is occupied.
- **Ground-V** (CVPR 2025),
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Zong_Ground-V_Teaching_VLMs_to_Ground_Complex_Instructions_in_Pixels_CVPR_2025_paper.pdf>,
  and **GREC/GREx**, <https://arxiv.org/abs/2308.16182>, already cover
  no-target grounding as a supervised task.
- **MARINE** (ICML 2025),
  <https://proceedings.mlr.press/v267/zhao25j.html>, **Visual Evidence
  Prompting** (ACL 2025), <https://aclanthology.org/2025.acl-long.205/>, and
  **Kestrel** (2026), <https://arxiv.org/abs/2603.16664>, use independent visual
  experts or grounding agents. This is likely effective but not our headline.
- **SAGE** (2026), <https://arxiv.org/abs/2603.27898>, uses existing sink tokens
  as generation-time checkpoints, then sharpens or broadens attention using
  attention/gradient grounding agreement. Generic sink-triggered attention
  modulation is occupied and conflicts with our amplification result.

### The null primitive is not new

- **Quantizable Transformers** (Bondarenko et al., 2023),
  <https://arxiv.org/abs/2306.12929>, links outliers to heads attempting a no-op
  and proposes clipped/gated attention.
- **StreamingLLM** (ICLR 2024),
  <https://proceedings.iclr.cc/paper_files/paper/2024/file/5e5fd18f863cbe6d8ae392a93fd271c9-Paper-Conference.pdf>,
  evaluates zero-sink/Softmax1 and learned sink tokens. Softmax1 is equivalent to
  adding a zero key/value option.
- **PriorTR** (2026), <https://arxiv.org/abs/2606.24156>, introduces a null token
  as an instruction-agnostic probe to subtract attention priors for visual-token
  pruning.

Therefore NEA cannot claim to invent null attention, sink tokens, rejection, or
visual grounding.

## Remaining Mechanism Gap

The project exposes a narrower missing operation. In a joint LVLM attention
head, the pretrained model chooses some total visual mass `rho`. Conditional on
that choice, ordinary attention must distribute the visual contribution over
real visual tokens. On an absent-target question, a semantically associated
region can win this relative competition and then be amplified into a `Yes`.

The missing outcome is not a generic no-op for the entire transformer and not a
generated `<REJ>` answer. It is:

> For this target-conditioned query, the visual branch has been allocated mass,
> but none of its real tokens supplies target evidence.

NEA adds one learned null key/value per selected layer/head to gate this
conditional visual contribution. Crucially, it preserves `rho`, the original
post-RoPE distribution among real visual tokens, and the text contribution
exactly. Pre-RoPE content compatibility decides only whether to retain that
original contribution or replace it with null.

For pre-RoPE visual energy `e_vis` and learned null logit `z_null`:

    g_null = sigmoid(z_null - e_vis)
    g_eff = alpha * g_null,                         0 <= alpha <= 1
    o_NEA = o_text + rho * ((1 - g_eff) * sum_i (a_i / rho) v_i
                            + g_eff * v_null).

The null value is norm-bounded by ordinary visual values. This prevents NEA from
becoming an unconstrained negative steering vector. A zero-null ablation tests
whether suppressing misleading visual content is sufficient; a learned bounded
value tests whether explicit negative evidence is needed to overcome the
language prior. The per-head `alpha` is initialized to zero, so NEA starts
exactly at vanilla rather than a no-RoPE redistribution of visual tokens.

## Why This Is Better Aligned Than TCS

- It uses natural positive, related-absent, and random-absent images; no edit
  artifacts or questionable object-removal interventions exist.
- It changes the architectural reason related objects become proxies rather
  than adding another answer-level loss to ordinary SFT.
- It cannot gain by increasing visual mass because visual mass is preserved by
  construction.
- It has no candidate vocabulary. The target-conditioned query supplies the
  concept, and all learned parameters are shared across category strings.
- It is cheap to falsify: null keys can first be fit from cached pre-RoPE queries
  and visual energies, before any behavior-changing value is trained.

## Main Threats and Required Controls

### Null key degenerates into a hidden probe

Because the query is already multimodal, `q dot k_null` may classify presence
without competing with visual keys. Compare against query-only and energy-only
probes, shuffle visual-key energy while holding the query fixed, and rerun on
within-target present/absent image swaps. Full null odds must beat both
components, depend on the explicit visual-key term, and fail when the complete
route receives the wrong image.

### Null value degenerates into answer bias

The learned value may simply push `No`. Preserve `rho`, cap value norm, balance
labels within category, and compare with a learned logit bias plus a parameter-
matched residual adapter. NEA has no mechanism claim if either matches it.

### COCO supervision becomes a closed vocabulary

Hold out 20 COCO target categories from all training. The operator has no
category-specific weights. Only after transfer succeeds should training expand
to LVIS annotations on the same COCO images; no open-vocabulary claim follows
from COCO aggregate performance alone.

### Binary QA does not transfer to captions

Iteration 1 is explicitly scoped to object-existence decisions. Free-generation
object-commitment training and CHAIR are later work only after POPE/MME
confirmation. Do not imply captioning mitigation from the first pilot.

## Falsification Standard

NEA is worth implementation only if null routing on unseen categories exceeds
query-only and visual-energy-only controls and depends on the correct image. It
is worth a method claim only if the bounded null value improves related-present
FPR and MCC beyond zero-null, logit-bias, and parameter-matched residual
adapters, with a non-sparse correction count and no material TPR/yes-rate shift.

Failure at either stage closes NEA without adding backbone LoRA or post-hoc
thresholds.
