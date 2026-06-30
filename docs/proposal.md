# Research Proposal — Looking Is Not Verifying

> Single source of truth for *what we are trying to publish and why*. When this
> conflicts with older planning docs (`design.md`, `aaai2027_paper_plan.md`,
> `icml_*`, `tdev_*`), this doc wins. Last major revision: 2026-06-24
> (pivot to a format-general criterion; full spec in `general_tdev_design.md`).

## 1. Thesis

Object hallucination in LVLMs is treated by most methods as an attention
problem: inspect or amplify visual attention and hallucination should drop.
This rests on a brittle proxy:

    visual attention allocation  ≈  object-level evidence verification

We argue the proxy fails, and we sharpen the claim through its evolution:

- v1 (diagnostic): **Looking is not verifying.** Attention can be plausible but
  wrong — it can point at a real, semantically *related* region while the
  *target* object is absent.
- v2: **Grounding is not enough unless it is target-discriminative under
  semantic-neighbor controls.** Even a strong, de-confounded grounding signal
  does not separate "the image supports the target" from "the image supports a
  plausible neighbor of the target."
- v3 (current headline): **The target-discriminative criterion is one rule that
  generalizes across question formats.** The same "stronger evidence for the
  target than for its strongest alternative" decision instantiates for yes/no,
  true/false, and multiple-choice — only the source of the *alternative* changes
  (constructed semantic neighbors for yes/no, the negation for T/F, the explicit
  options for MCQ). The contribution is this one criterion and the shared failure
  mode it exposes across formats and datasets, not a single-benchmark score.

One-sentence spine:

> A visual claim is grounded only when the image provides stronger evidence for
> the target than for its strongest discriminative alternative — whatever the
> question format supplies that alternative.

## 2. Positioning vs. prior work (what is and is not ours)

The space is crowded. We must be explicit about what is background vs. novel.

### Background — do NOT pitch as our contribution

- **Position/length → more hallucination** is well established (LURE; "Why
  LVLMs Are More Prone to Hallucinations in Longer Responses", arXiv:2510.20229;
  "Do More Details…", arXiv:2406.12663).
- **Position as a confounder for attention-based detection** is *already
  published*: **HaloProbe** (Zohrabi, Hasani, Gupta, Soleymani Baghshah, A.
  Rohrbach, M. Rohrbach; arXiv:2604.06165, Apr 2026). It frames the reversal as
  **Simpson's paradox**: conditioned on position, hallucinated tokens show
  comparable/higher image attention; marginalized, correct tokens look higher.
  It also uses a **per-head attention + confidence** probe, and identifies
  **object repetition** as a second confounder. It evaluates on 5 models
  (LLaVA-1.5, Shikra, MiniGPT-4, Qwen3-VL, InternVL3.5) and hits 93.5 AUROC.
- **Benchmark artifacts in hallucination detection** are also being studied
  generally (PARALLAX, arXiv:2605.17028).

Implication: our **position-confound chapter and per-head diagnostic are no
longer headline material.** They are largely pre-empted by HaloProbe. We keep
them as *background that motivates the real question*, and we cite HaloProbe
explicitly. We borrow HaloProbe's **explanation framing** (Simpson's paradox;
internal-signal vs. external-confounder factorization) to set up our point —
but not its experimental burden.

### Our moat — what no prior work does

- **Format-general target-discriminative criterion (TDEV).** One decision rule —
  the image must support the target more than its strongest discriminative
  alternative — that instantiates across yes/no, true/false, and multiple-choice.
  Only the *contrast set* changes by format (constructed semantic neighbors;
  negation; explicit options). Prior confidence/attention methods evaluate target
  evidence *in isolation*; HaloProbe factors neighbors *out* as noise. We make the
  decision *relative* to the contrast set and keep it the same across formats.
  Full spec: [general_tdev_design.md](general_tdev_design.md).
- **Semantic-neighbor stress test.** We construct POPE/COCO negative subsets
  where the target is absent but a co-occurring / same-supercategory neighbor is
  present, and show all attention methods leave an unclosed **related-minus-plain
  FPR gap**. Empirically, POPE adversarial negatives are 84.8% related-present —
  the difficulty is structural, not anecdotal. The same alternative-confusion
  failure mode is what we test for across the other formats/datasets.
- **Both tasks under matched controls.** Detection (position-controlled) and
  mitigation (answer-prior controlled, POPE yes-rate/TPR/FPR/MCC) in one
  framework.

## 3. Method design — format-general no-external-detector family

### The general method in one place

The headline method is now a **format-general** instantiation of TDEV, specified
in full in [general_tdev_design.md](general_tdev_design.md). It factors into
three parts:

1. **Contrast-set construction** (the only format-dependent part): enumerate the
   discriminative alternatives — explicit options for MCQ, the negation for T/F,
   constructed semantic neighbors for yes/no existence and free-form captions.
2. **Evidence readout** (format-agnostic): the internal hidden-margin and
   answer-margin features, generalized to score any candidate claim.
3. **Discriminative decision rule** (format-agnostic): the target is grounded iff
   its evidence exceeds the strongest alternative's by a calibrated `τ`.

An optional **commit-verify two-stage** wrapper (free generation, then
discriminative verification of each commitment) explains why CHAIR detection is
strong (AUROC 0.898) and can be ported back to POPE to import the same
inconsistency signal the single-shot format lacks.

The rest of this section records the no-external-detector evidence work that the
general framework is built on.

### Why the design changed

The original strongest TDEV result used **OWLv2** as an external
open-vocabulary evidence backend. That remains a useful positive control, but it
cannot be the headline method: it invites the ``why not just OWLv2?'' critique
and makes the method look like an external-tool pipeline.

We therefore implemented and validated several internal reference paths. The
current conclusion is empirical, not aspirational: **the external detector can be
removed for the current method family, but the internal POPE verifier still
trails the OWLv2 positive control.**

### Evaluated internal reference paths

| Route | Evidence | Status | Main lesson |
|---|---|---|---|
| A | IC / final-layer logit-lens target-vs-neighbor margin | rejected as main | cheap but too conservative or too weak |
| B | attention-region target-vs-neighbor discriminability | rejected as main | raw attention-region overlap is not sufficiently discriminative |
| C | contrastive per-head attention-shape probe | diagnostic only | internal signal exists but recall/FPR tradeoff is not enough |
| D | hidden-state target-vs-neighbor contrast | strongest internal family | hidden+answer evidence gives the best no-external POPE gate |

The current internal method uses task-specific heads over the same broad idea:
query the VLM itself for target-object support rather than invoking an external
region detector.

### Confirmed ceiling (2026-06-24)

Two experiments with full POPE (9000 rows) confirm that MCC ≈ 0.740 is the
internal-method ceiling:

- **Experiment A**: per-layer hidden margins (layers 16/22/27/31 separately,
  18 features total) with logistic regression → MCC 0.739. No improvement over
  the existing 6-feature scalar verifier (MCC 0.742).
- **Experiment B**: full 49152-dim hidden-state features + OOF supervised
  logistic regression → AUROC 0.9427, gate MCC 0.740. Essentially the same
  as the scalar method.

The gap to OWLv2 (MCC 0.763, Related FPR 0.069) is **structural**: internal
semantic distribution features cannot replicate the spatial evidence that an
external region detector provides. This gap should be reported honestly in the
paper; do not invest more effort trying to engineer it away.

### Current selected no-external rows

Source of truth: [no_external_detector_summary.md](no_external_detector_summary.md),
generated by `mitigation/scripts/build_no_external_detector_summary.py` from the
current artifacts.

| Scenario | Current no-external method | Result summary | Role |
|---|---|---|---|
| POPE semantic-neighbor gate | hidden+answer verifier, leave-one-split-out calibration | MCC 0.742, TPR 0.810, FPR 0.073, related FPR 0.097 | main POPE internal row |
| POPE stricter-FPR ablation | same verifier, `min_fpr_tpr0.80` | MCC 0.740, TPR 0.800, FPR 0.067, related FPR 0.089 | FPR-focused ablation |
| CHAIR object-claim detection | `answer_absence_score = -target_yes_margin` | AUROC 0.898, within-bin 0.872, MCC 0.596 | main CHAIR detector |
| CHAIR caption intervention | answer-absence selected generic rewrite top-10 | CHAIRi 0.113, CHAIRs 0.432, LCS retention 0.9958 | main preserved intervention |
| CHAIR aggressive ablation | answer-absence selected delete top-10 | CHAIRi 0.107, CHAIRs 0.410, but fails preservation gate | upper-bound only |

OWLv2 hybrid gate+rescue remains in tables only as an **external positive
control** (MCC 0.763, FPR 0.051, related FPR 0.069 on POPE). It should not be
described as the paper-facing method.

### What this means for the paper

The contribution should now be framed as a **target-discriminative internal
verification family**, not as an external detector pipeline and not as a fully
solved hallucination mitigator. The honest claim is:

> Internal target/answer evidence can reduce semantic-neighbor false positives
> and drive CHAIR claim correction without an external detector, but it still
> does not fully match a strong external region-verification positive control.

This keeps the novelty centered on the semantic-neighbor criterion and the
internal target-vs-neighbor evidence readout, while avoiding overclaiming.

## 4. Contribution framing

Frame as an **analysis + diagnostic-criterion** paper, not a new-detector paper.
The contribution axis is **generality across question formats**, not a
single-benchmark score:

1. A **format-general target-discriminative criterion (TDEV)**: one decision rule
   — image support for the target must exceed support for its strongest
   discriminative alternative — instantiated for yes/no, true/false, and
   multiple-choice, with only the contrast set changing by format. See
   [general_tdev_design.md](general_tdev_design.md).
2. A controlled **semantic-neighbor / alternative-confusion stress test** showing
   the *same* failure mode recurs across formats and datasets and survives
   position/repetition de-confounding — current methods over-fire on the
   strong-alternative subset.
3. Evidence across **detection and mitigation** under matched controls, with no
   external detector in the main method rows, OWLv2 retained only as an external
   positive control, and HaloProbe-style probes as background/diagnostic
   references.

Honest scope: the POPE absolute gain is modest (MCC 0.742 vs 0.730) and trails
the OWLv2 control (0.763) at a confirmed structural ceiling. Generality is a
*complementary* axis — it does not raise that number and must not be framed as if
it did.

## 5. Known risks / must-dos

- **Format generality is now claimed but not yet tested beyond yes/no + caption.**
  The generality claim requires validating the criterion on T/F and MCQ. Plan
  (see [general_tdev_design.md](general_tdev_design.md) §8): **MME first**
  (yes/no + T/F, lowest cost, closest to POPE), then **MMBench / SEED-Bench**
  (MCQ — hardest generality claim), then **AMBER** (second yes/no neighbor
  benchmark). For each, report the discriminative-margin detector/gate and whether
  the alternative-confusion failure mode appears.
- **MCQ may reduce to argmax.** If the answer logit carries all the signal, the
  rule adds nothing; the MCQ contribution must then come from the hidden-margin
  readout or from selective prediction (abstain on small runner-up margins).
  Validate before claiming.
- **Single model.** HaloProbe uses 5. We must add ≥1 modern model (Qwen-VL /
  InternVL) at least for the semantic-neighbor headline. (Qwen2.5-VL is already
  done on POPE; extend to the new formats/datasets as they come online.)
- **HaloProbe head-to-head.** If time permits, run a HaloProbe-style internal
  probe on our related-present subset; otherwise cite it as motivation and keep
  our target-vs-neighbor stress test distinct.
- **Object repetition control.** Add repetition as a second confounder control
  (free, borrowed from HaloProbe) so detection results are not re-attackable.
- **Generation-time integration.** CHAIR intervention is currently post-hoc
  claim editing. A decoding-time or regeneration-time variant would strengthen
  the method claim.
- **Keep OWLv2 scoped.** OWLv2 stays only as an external positive control /
  upper-bound ablation, never the headline method.

## 6. Pointers

- **Format-general method spec: [general_tdev_design.md](general_tdev_design.md).**
- Why each score is computed: [design.md](design.md), [detection/docs/scores.md](../detection/docs/scores.md).
- Results so far: [experiment_results.md](experiment_results.md).
- Current no-external main rows: [no_external_detector_summary.md](no_external_detector_summary.md).
- Decision history: [iteration_log.md](iteration_log.md).
- Older venue plan (background): [aaai2027_paper_plan.md](aaai2027_paper_plan.md).
- Detector positioning notes (background): [tdev_detector_positioning.md](tdev_detector_positioning.md).
