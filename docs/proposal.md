# Research Proposal — Looking Is Not Verifying

> Single source of truth for *what we are trying to publish and why*. When this
> conflicts with older planning docs (`design.md`, `aaai2027_paper_plan.md`,
> `icml_*`, `tdev_*`), this doc wins. Last major revision: 2026-06-23.

## 1. Thesis

Object hallucination in LVLMs is treated by most methods as an attention
problem: inspect or amplify visual attention and hallucination should drop.
This rests on a brittle proxy:

    visual attention allocation  ≈  object-level evidence verification

We argue the proxy fails, and we sharpen the claim through its evolution:

- v1 (diagnostic): **Looking is not verifying.** Attention can be plausible but
  wrong — it can point at a real, semantically *related* region while the
  *target* object is absent.
- v2 (current headline): **Grounding is not enough unless it is
  target-discriminative under semantic-neighbor controls.** Even a strong,
  de-confounded grounding signal does not separate "the image supports the
  target" from "the image supports a plausible neighbor of the target."

One-sentence spine:

> A visual claim is grounded only when the image provides stronger evidence for
> the target object than for its plausible semantic neighbors.

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

- **Semantic-neighbor stress test.** We construct POPE/COCO negative subsets
  where the target is absent but a co-occurring / same-supercategory neighbor is
  present, and show all attention methods leave an unclosed **related-minus-plain
  FPR gap**. Empirically, POPE adversarial negatives are 84.8% related-present —
  the difficulty is structural, not anecdotal.
- **Target-discriminative criterion (TDEV).** A decision rule that compares
  target evidence against semantic-neighbor evidence, rather than validating
  target presence in isolation. HaloProbe factors neighbors *out* as noise; we
  treat the target-vs-neighbor contrast as the *signal*.
- **Both tasks under matched controls.** Detection (position-controlled) and
  mitigation (answer-prior controlled, POPE yes-rate/TPR/FPR/MCC) in one
  framework.

## 3. Method design — drop the external detector

### Problem with the current design

The strongest current TDEV result uses **OWLv2** (an external open-vocabulary
detector) as the evidence backend. Two issues:

1. **Novelty/credibility:** reviewers ask "why not just use OWLv2?" — it reads
   as a Woodpecker-style external-tool pipeline.
2. **Cost:** OWLv2 region inference per image is the main latency source.

Both are solved by the same move: **read evidence from the verifier VLM's own
internals, not from an external detector.** We already have evidence the signal
is internal: IC is our strongest position-controlled baseline; our per-head
probe reaches within-bin AUROC 0.73; HaloProbe's internal probe reaches 93.5.

### Core idea: self-contrastive target-vs-neighbor from internal signals

For an object claim (e.g. *sink*), read the verifier's internal evidence for the
**target** and for its top-k **semantic neighbors** (e.g. *toilet*), and compare:

    score_hallucination  ∝  evidence(neighbor) − evidence(target)   (neighbor dominance)

Neighbors come from a precomputed text-only co-occurrence / supercategory table
(no detector, no image pass). Evidence is read from the single teacher-forced
forward we already run.

### Three distinct evidence-readout approaches (independent methods)

| ID | Approach | Cost | Training-free | Overlap w/ HaloProbe | Notes/risk |
|---|---|---|---|---|---|
| **A** | **IC / logit-lens margin**: target vs. neighbor internal confidence at the object position | ~0 (reuse logits) | yes | medium (IC-like) | strongest cheap baseline; may reflect language prior, not vision |
| **B** | **Attention-region discriminability**: do target & neighbor queries attend to the *same* visual region? same region ⇒ non-discriminative ⇒ hallu | ~0 (reuse cached attention rows) | yes | **low — this is uniquely ours** | needs a good region-overlap metric (peak overlap / distribution distance) |
| **C** | **Contrastive per-head probe**: per-head features of `target − neighbor`, fed to a probe | training needed | no | **high** | use only as a supervised *ceiling/diagnostic*, not the main method |

Orthogonal switch — how the neighbor evidence is obtained:

- **Passive:** read only from the original target query; neighbor word from the
  co-occurrence table. Cheapest.
- **Active:** actually pose the neighbor question and compare two reads. Cleaner
  contrast, one extra (batchable) forward — still far cheaper than OWLv2.

### Recommended bet

- **B is the primary method** — unique to us, zero extra cost, and it directly
  operationalizes "looking is not verifying."
- **A is the cheap strong baseline / ablation** (and an upgrade to plain IC).
- **C is the supervised ceiling** ("the signal exists; a training-free variant
  recovers X% of it"), reported separately and contrasted with HaloProbe.

### Cost summary

| | OWLv2 backend (current) | Internal self-contrastive (proposed) |
|---|---|---|
| Extra model | full OWLv2 detector | none |
| Extra forward per image | OWLv2 region inference (slow) | 0 (reuse teacher-forced forward) |
| Neighbor source | label/box alignment | offline co-occurrence table lookup |
| Reviewer framing | "why not just OWLv2?" | "the VLM itself lacks target-discriminative verification" |

## 4. Contribution framing

Frame as an **analysis + diagnostic-criterion** paper, not a new-detector paper:

1. A controlled **semantic-neighbor stress test** that exposes a failure mode
   surviving position/repetition de-confounding.
2. A **target-discriminative criterion (TDEV)** implementable from VLM internals
   at near-zero extra cost — no external detector.
3. Evidence across **detection and mitigation** under matched controls, with
   HaloProbe's strong de-confounded probe as the reference that *still* misses
   the semantic-neighbor case.

## 5. Known risks / must-dos

- **Single model.** HaloProbe uses 5. We must add ≥1 modern model (Qwen-VL /
  InternVL) at least for the semantic-neighbor headline.
- **HaloProbe head-to-head.** Run a HaloProbe-style internal probe on our
  related-present subset; the target result is that it leaves the FPR gap open.
- **Object repetition control.** Add repetition as a second confounder control
  (free, borrowed from HaloProbe) so detection results are not re-attackable.
- **Keep the criterion backend-agnostic.** OWLv2 stays only as an
  upper-bound/efficiency ablation, never the headline.

## 6. Pointers

- Why each score is computed: [design.md](design.md), [detection/docs/scores.md](../detection/docs/scores.md).
- Results so far: [experiment_results.md](experiment_results.md).
- Decision history: [iteration_log.md](iteration_log.md).
- Older venue plan (background): [aaai2027_paper_plan.md](aaai2027_paper_plan.md).
- Detector positioning notes (background): [tdev_detector_positioning.md](tdev_detector_positioning.md).
