# General TDEV — A Format-General Target-Discriminative Criterion

> Detailed design for the unified method. The high-level pivot and contribution
> framing live in [proposal.md](proposal.md); this doc is the full specification.
> Created 2026-06-24.

## 1. Why the method had to become format-general

The original method is **POPE-shaped**: yes/no existence questions plus a
constructed semantic-neighbor contrast. Three problems forced a redesign.

1. **The POPE absolute gain is small.** The internal verifier reaches macro
   MCC 0.742 vs. vanilla 0.730 (+0.012); F1 +0.005. A single-shot yes/no probe
   leaves little room — when vanilla "yes" precision is already 0.906, a gate can
   only flip ~9% of answers and every wrong flip is a new error.
2. **The strong CHAIR signal comes from a *different* mechanism.** CHAIR
   detection reaches AUROC 0.898 because it is implicitly **two-stage**: the
   model first *freely commits* to an object in its caption, then *reveals
   uncertainty* when that object is re-queried in isolation. The commit-vs-verify
   inconsistency is the signal. POPE's single-shot format has no prior commitment
   to contradict, so this signal is absent.
3. **The next evaluations are multiple-choice and true/false.** A POPE-only
   method does not transfer. Generality across question formats is now a
   first-class requirement, and it is also what justifies validating on the other
   standard datasets (MME, AMBER, MMBench/SEED-Bench).

The strategic consequence: the contribution axis moves from *"a marginally
better POPE verifier"* (weak) to *"a single discriminative criterion that
instantiates across question formats and exposes a shared failure mode"*
(strong). The weak POPE number becomes one honest data point inside a generality
story rather than the headline.

## 2. The unifying principle

> A visual claim is grounded only when the image provides stronger evidence for
> the **target** than for its strongest **discriminative alternative**.

Formally, for an image `I` and a claim set
`C = {c_0, c_1, ..., c_k}` in which `c_0` is the claim under test (the *target*)
and `c_1..c_k` are mutually-contrastive *alternatives*, define an evidence
readout `e(c | I)`. The target-discriminative decision is:

```
target c_0 is grounded   iff   e(c_0 | I) − max_{j≠0} e(c_j | I) ≥ τ
```

**Contrast with prior work.** Confidence-, calibration-, and attention-based
methods evaluate `e(c_0 | I)` *in isolation* — "is the evidence for the target
high?" We make the decision *relative* to the contrast set — "is the evidence for
the target higher than for its best competitor?" HaloProbe factors neighbors
*out* as confounding noise; we treat the target-vs-neighbor contrast as the
*signal*. This relative criterion is the moat and it is format-independent.

## 3. Three components

The method factors cleanly into three parts. Only the first depends on the
question format; the other two are shared across all formats.

### 3.1 Contrast-set construction (format-dependent)

The only part that changes per task. The harder the format, the more of the
contrast set we must construct ourselves.

| Format | Example datasets | Contrast set `C` | Construction cost |
|---|---|---|---|
| **Multiple choice (MCQ)** | MMBench, SEED-Bench | the explicit options A/B/C/D | **free** — given by the question |
| **True / False** | MME, parts of AMBER | `{claim, ¬claim}` | **free** — templated negation |
| **Yes/No existence** | POPE, AMBER discriminative | `{target present, semantic-neighbor present}` | needs a neighbor table (co-occurrence + supercategory) |
| **Free-form caption** | CHAIR, Object HalBench | per-mention `{object, semantic-neighbor}` | needs a neighbor table + the model's own generation |

Key insight this table encodes: **POPE is the hardest case, not the easiest**,
because its alternatives are *implicit* and must be constructed. MCQ is the
easiest because the discriminative alternatives are written into the question —
"结合 question 内的内容来回答" is literally free for MCQ/TF.

### 3.2 Evidence readout (format-agnostic)

A VLM-internal score of how strongly image `I` supports an arbitrary claim `c`.
We reuse the existing internal-verifier features, generalized from "is object X
present" to "is claim c supported":

- **Hidden-margin** `e_hidden(c|I)`: cosine alignment between the claim's object
  hidden representation and the visual hidden representation, read at layers
  22/31 (the existing `hidden_align_margin`, `hidden_cross_margin`,
  `hidden_obj_separation`, `hidden_vis_separation`).
- **Answer-margin** `e_answer(c|I)`: the yes/support logit margin when `c` is
  posed as a direct probe (the existing `answer_support_score`,
  `answer_contrast_margin`).

Both are already implemented for yes/no and CHAIR; the generalization is to emit
them for *each* candidate claim in `C`, not only for a present/absent pair.

> Open question for MCQ. If the evidence readout is just the answer logit, the
> decision rule reduces to plain argmax decoding and we add nothing. The MCQ
> contribution must therefore come from either (a) the **hidden-margin** readout
> carrying signal beyond the logit, or (b) **selective prediction** — flagging
> answers whose margin to the runner-up option is below `τ` as
> hallucination-prone / abstain. (b) is a legitimate, testable contribution even
> if (a) fails. This must be validated empirically before MCQ is claimed.

### 3.3 Discriminative decision rule (format-agnostic)

Apply the relative criterion of §2 with a calibrated `τ`. Two operating modes,
both already used on POPE/CHAIR:

- **Detection / scoring**: report `e(c_0|I) − max_j e(c_j|I)` as a continuous
  hallucination score (AUROC, within-bin AUROC under position control).
- **Gate / mitigation**: suppress or rewrite the target claim when the margin
  falls below `τ` (POPE answer flip; CHAIR generic-noun rewrite).

## 4. Optional commit-verify two-stage wrapper

This is the mechanism behind the strong CHAIR result, made explicit and
portable. For any format where the model can also generate freely:

- **Stage 1 (commit)**: free generation produces claims the model is willing to
  assert unprompted (a caption, or a rationale).
- **Stage 2 (verify)**: run §3 on each committed claim against its contrast set.
  The *inconsistency* between a confident free commitment and a weak
  discriminative margin is itself a strong hallucination signal.

**Porting to POPE (a concrete next experiment).** Add a caption pre-pass per
POPE question: (a) does the target object appear in the free-form caption?
(b) fuse that with the direct-query hidden/answer margin. This imports the
two-stage inconsistency signal that POPE's single-shot format otherwise lacks.
Cost: one extra forward pass per question (~9000, ~1–2 h on the current GPUs).

## 5. Concrete method modifications & the unified architecture (preliminary design)

> **Status: preliminary design.** This section records the intended refactor from
> the current POPE-specific implementation to the format-general core. Interfaces
> and decisions here are proposals to be validated, not finalized contracts.

### 5.1 Why the current implementation is POPE-specific

The current `evaluate_answer_confidence_tdev_pope.py` /
`evaluate_chair_internal_verifier.py` path hardcodes four existence-question
assumptions:

1. it regex-parses an **object noun** out of "Is there a X?";
2. the **contrast** comes only from a co-occurrence neighbor table;
3. the hidden readout is anchored on an **object token** span;
4. the probe template is **yes/no existence**.

All four must be lifted to make the method general.

### 5.2 Four unifications

**(1) Input → a declarative claim under test.** Stop extracting an object noun;
template every question+candidate into a declarative statement.

| Format | Original | → unified claim |
|---|---|---|
| Existence (POPE) | "Is there a snowboard?" | "There is a snowboard in the image." |
| Attribute (MME) | "Is the car red?" | "The car is red." |
| True/False | the proposition | the proposition itself |
| MCQ | option B "A dog is playing" | "A dog is playing." |

**(2) Contrast set → a pluggable adapter** behind one interface
`build_contrast_set(claim, format) -> [target_claim, alt_1, ..., alt_k]`:

- Existence: alt = semantic neighbor ("There is a skateboard") — current logic
  becomes one plugin.
- Attribute: alt = other values in the attribute's value set ("blue", "green").
- True/False: alt = the negation.
- MCQ: alt = the remaining options (**free — given by the question**).

Unifying definition of an alternative: *the hypothesis the model is most likely
to confuse the target with*. For MCQ this is given; for existence it is the
co-occurrence neighbor; for attribute it is the model's runner-up value.

> **Design decision A (alternative source).** Keep the external co-occurrence
> table (stable, but reviewer-attackable as "external knowledge", existence-only)
> vs. switch to **self-derived** alternatives (ask the model for its runner-up /
> next-highest-probability candidate — removes the external dependency and
> unifies all formats). Proposed resolution: **do both** — self-derived as the
> main method, co-occurrence as an ablation, which also rebuts the "needs
> external knowledge" critique. Cost: self-derived adds queries per item.

**(3) Evidence readout → one format-agnostic claim scorer**
`ClaimScorer(image, claim) -> { hidden_margin, answer_verify_margin }`:

- hidden-margin: generalize span localization from "object token" to the
  **claim's content span** (or mean-pool over claim content), then align to the
  visual hidden as today; compute the existing align/cross/separation features
  **per candidate, taking max over alternatives**.
- answer-verify-margin: pose every claim as a uniform probe —
  `"<claim> Is this true? Answer yes or no."` — and read the yes-logit. This is
  what makes the readout universal: any claim in any format can be asked "is this
  true?", including each MCQ option.

> **Design decision B (claim span).** Mean-pool over the whole claim content
> (simple, general) vs. the differentiating span between target and alternative
> (more discriminative, harder to locate). Proposed: start with mean-pool to get
> the core working; add differentiating-span as an enhancement.

> **Design decision C (MCQ probe).** Verify each option via the uniform "Is this
> true?" probe (general, but multiple forward passes) vs. read the native option
> logit directly (cheap, but may degrade to plain argmax). This is the MCQ open
> question of §3.2; resolve empirically.

**(4) Decision rule → discriminative margin, four output modes.** Compute
`score = e(target) − max_j e(alt_j)`; use it in one of four modes:

| Mode | Use | Format |
|---|---|---|
| `gate` | flip vanilla "yes" when margin < τ | yes/no existence (POPE) |
| `argmax+abstain` | pick top; flag when top1 − top2 < τ | MCQ |
| `score` | output the margin directly | detection (any format) |
| `rank+rewrite` | rewrite/delete the weakest-margin claim | free-form mitigation (CHAIR) |

### 5.3 The unified architecture

```
                    ┌─ ExistenceAdapter (co-occ / self-derived)
ContrastSetBuilder ─┼─ AttributeAdapter (value set)
   (format plugin)  ├─ TrueFalseAdapter (negation)
                    └─ MCQAdapter (options)  ← free
                          │
GenerationFrontEnd  ──────┤   ← optional; free generation = the "commit" stage
   (commit-verify)        │
                          ▼
   ClaimScorer(image, claim) → (hidden_margin, answer_verify_margin)   ← format-agnostic core
                          │
                          ▼
   DiscriminativeRule: e(target) − max e(alt)  → { gate | argmax+abstain | score | rank+rewrite }
```

The current POPE and CHAIR scripts become thin wrappers: each is "one adapter +
the shared core", differing only in input adapter and output mode.

### 5.4 CHAIR as the generative instance (and what it reveals)

CHAIR is **not a special case** — it is the generative instantiation of the same
core:

- `GenerationFrontEnd` = the free caption (the *commit* stage); each mention →
  "There is a {object}".
- `ContrastSetBuilder` = the **same** ExistenceAdapter as POPE.
- `ClaimScorer` = the **same** core as POPE, unchanged.
- Output mode = `score` (detection, AUROC 0.898) and `rank+rewrite` (intervention,
  generic-noun rewrite).

So **CHAIR = POPE existence-adapter + shared core, wrapped by the generation
front-end, with a generative output mode.**

This unification explains the otherwise-puzzling gap (CHAIR detection AUROC 0.898
vs. POPE gate MCC +0.012 over vanilla): in CHAIR the model has **already
committed** to the object in its caption, so the answer-verify-margin measures
*commit-vs-verify inconsistency* — a strong signal. In POPE there is no prior
commitment, so the same readout measures only isolated evidence — weak. The
answer-verify-margin therefore does double duty in the free-form setting (it is
both the evidence readout and the inconsistency measure).

This is exactly why the `GenerationFrontEnd` is part of the core architecture and
not a CHAIR-only detail: porting commit-verify to POPE (the §4 caption pre-pass)
is *the same* front-end. The two views converge:

- CHAIR = POPE core + generation front-end.
- "POPE + commit-verify" = POPE core + CHAIR's generation front-end.

### 5.5 Refactor implication

Split the codebase into: a shared `ClaimScorer`, a `ContrastSetBuilder` protocol
with per-format adapters, a claim-templating helper, and an optional
`GenerationFrontEnd`. The existing `evaluate_answer_confidence_tdev_pope.py` and
`evaluate_chair_internal_verifier.py` are reimplemented as wrappers over these
pieces, which should be behaviour-preserving for the current POPE/CHAIR rows
(a regression check: the refactored POPE path must reproduce MCC 0.742 and the
CHAIR path AUROC 0.898 before any new format is added).

## 6. What generality buys the paper

- The contribution is reframed as **one criterion, many formats, one shared
  failure mode** — not a POPE leaderboard entry.
- The **alternative-confusion failure mode** (semantic neighbor / strong
  distractor option) is shown to recur across yes/no, T/F, and MCQ, and across
  datasets — turning a single-dataset observation into a systematic finding.
- It directly answers the "why only POPE?" and "does this transfer?" reviewer
  questions before they are asked.

## 7. Honest scope and risks

- **POPE absolute number stays modest.** The internal verifier still trails the
  OWLv2 positive control (MCC 0.742 vs 0.763) and there is a confirmed structural
  ceiling near MCC 0.740. Generality is a *complementary* contribution axis, not
  a fix for that number. Do not reframe generality as if it raised the POPE
  score.
- **MCQ may collapse to argmax** (see §3.2). Validate the hidden-margin / abstain
  variant before claiming the MCQ instantiation.
- **Two-stage adds latency** (one extra forward pass per item). Report it.
- **Contrast-set quality for constructed formats** (POPE/CHAIR) depends on the
  neighbor table; document its construction and sensitivity.

## 8. Validation plan (datasets × formats)

| Dataset | Format | Contrast set | Status |
|---|---|---|---|
| POPE | yes/no existence | constructed neighbors | **done** (MCC 0.742; OWLv2 control 0.763) |
| CHAIR | free-form caption | per-mention neighbors | **done** (detect AUROC 0.898; intervene CHAIRi 0.113) |
| Qwen2.5-VL on POPE | yes/no existence | constructed neighbors | **done** (2nd model; TDEV MCC 0.769 vs vanilla 0.765) |
| AMBER (discriminative) | yes/no | constructed neighbors | to do |
| MME (hallucination subset) | yes/no + T/F | given negation | to do — first new format test (lowest cost) |
| MMBench or SEED-Bench | MCQ | explicit options | to do — validates the MCQ instantiation (§3.2) |

For each dataset, report (1) the discriminative-margin detector/gate under the
shared criterion, and (2) whether the alternative-confusion failure mode appears
— i.e. whether current methods over-fire on the strong-distractor subset.

Recommended order: **MME first** (closest to existing POPE format, exercises the
T/F instantiation cheaply), then **MMBench/SEED MCQ** (tests the hardest
generality claim and the abstain fallback), then **AMBER** (reinforces the yes/no
neighbor story on a second object-hallucination benchmark).
