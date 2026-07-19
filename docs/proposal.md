# Research Proposal — Looking Is Not Verifying

> Single source of truth for *what we are trying to publish and why*.
> **Restart baseline: 2026-07-07.** On this branch (`restart-two-pronged`) the
> project was reset to its original two-pronged motivation. Only the reproducible
> detection and mitigation *baselines* were retained; all prior method design
> (TDEV / semantic-neighbor verifier / no-external-detector family / CHAIR
> intervention) was torn down and is to be rebuilt from this starting point. The
> full torn-down history remains on `main`.

## 1. Thesis

Object hallucination in LVLMs (LLaVA-1.5-7B baseline) is treated by most methods
as an **attention problem**: inspect visual attention to *detect* hallucination,
or amplify visual attention to *mitigate* it. Both rest on one brittle proxy:

    visual attention allocation  ≈  target-object evidence verification

**We argue the proxy fails, and it fails in both operational directions.** This
two-pronged failure is the motivation spine of the project.

### Prong 1 — Reading attention is unreliable (detection)

Attention-based detectors look reliable only because their signal is confounded
by **position / response depth** (a RoPE-driven length effect: tokens deeper in
the response are both more likely to be hallucinated and to carry a different
attention geometry). Once we control for generation position — our
length-controlled *within-bin* metric — mean-over-head attention-shape scores
**collapse to chance**:

- `sink_only_conc_top10_mass`: overall AUROC 0.8099 → **within-bin 0.5087**.
- PAS 0.835 → 0.593, SVAR 0.834 → 0.575 (the position-only baseline alone is 0.830).
- This holds for **every unsupervised single-score attention detector we could
  reproduce faithfully from its official repo**, including three 2026 methods:
  **SAE** (ICLR-adjacent, with its real Mask2Former segmenter) 0.783 → 0.556,
  **LaSCD** 0.574 → **0.500 (chance)**, and **TVER** (ICCV 2025) — whose overall
  AUROC (0.830) sits *exactly on the position-only baseline* → within-bin 0.566.

Reading the shape of attention cannot separate a hallucinated token from a
grounded one once the length confounder is removed. Notably, the 2026 detection
literature is itself pivoting away from attention toward hidden-state/uncertainty
signals — independent corroboration of this prong.

**Scope (must state honestly).** The collapse is specific to *mean-over-head
attention-shape* scores. Not every internal signal dies under the control:
IC stays at within-bin 0.686, and a late-layer *per-head* supervised probe
reaches 0.730. So the precise claim is **"attention-shape allocation ≠
verification,"** not "no internal signal survives length control."

### Prong 2 — Pushing attention is unreliable, and backfires (mitigation)

Amplifying visual attention does not add *target-object* evidence, so where it
moves a decision it moves it toward **"yes."** The hard negatives in existence
questions are **semantic neighbors** (the bathroom-with-a-*toilet* image, asked
"is there a *sink*?"): attention is already on a real, related region, and
amplifying it pushes the answer to the wrong "yes." Measured on POPE baselines:

- ClearSight: yes-rate 0.450 → 0.486, related-present FPR 0.114 → 0.160
  (adversarial 0.164 → 0.223), while Macro MCC *drops* 0.730 → 0.723.
- Visual Attention Sink shows the same, milder pattern.

Apparent benchmark gains from attention amplification are **over-fitting to the
benchmark's answer prior**, not improved verification.

**Scope (must state honestly).** This is the attention-*amplification* family.
A pure attention-only reweight (PAI) is inert (yes-rate 0.446, MCC 0.728) — it
neither helps nor backfires. The claim is scoped to attention *amplification*,
not every attention method.

### The two prongs reduce to one root

> Attention allocation is not target-object evidence verification. Reading it
> (Prong 1) collapses to chance under length control; pushing it (Prong 2) only
> adds yes-bias.

The constructive answer — a verification criterion that neither reading nor
amplifying attention provides — is **to be rebuilt** from this starting point
(see §3).

## 2. Positioning vs. prior work

### Background — not our contribution
- **Position/length → more hallucination** is well established (LURE;
  arXiv:2510.20229; arXiv:2406.12663).
- **Position as a confounder for attention-based detection** is *already
  published*: **HaloProbe** (arXiv:2604.06165, Apr 2026), which frames the
  reversal as **Simpson's paradox**, uses a per-head attention+confidence probe
  (93.5 AUROC), adds object repetition as a second confounder, and evaluates on
  5 models. **Implication: Prong 1 is background/motivation, not headline.** We
  cite HaloProbe and borrow its explanation framing; we do not re-claim it.
- **Benchmark artifacts in hallucination detection** (PARALLAX, arXiv:2605.17028).

### What we still own (to be developed on this branch)
- The **mitigation-side critique** made measurable: the *related-present vs.
  plain-absent* FPR decomposition that exposes attention-amplification's
  answer-prior over-fitting — invisible in prior aggregate-POPE tables.
- A **verification criterion** that is relative to the target's strongest
  discriminative alternative, rather than reading/amplifying attention.
  (Design deliberately open — the previous TDEV family was torn down.)

## 3. Method design — clean slate

The prior method superstructure (OWLv2-backed TDEV, internal hidden-margin /
answer-confidence verifier, CHAIR answer-absence intervention, format-general
adapter core) has been **removed** from this branch. What survives is only the
apparatus needed to reproduce the two-pronged critique:

- **Detection reproduction:** `detection/baselines/` (NLL, Entropy, IC, GLSim,
  SVAR, PAS, Beyond-ADS/CGC) under the position/length-controlled metrics defined
  in [design.md](design.md).
- **Mitigation reproduction:** `mitigation/baselines/` (vanilla, PAI, ClearSight,
  VisAttnSink, VCD, DAMRO, OPERA, SPIN) under the POPE/CHAIR behavioral protocol.

The new method is to be designed from the two-pronged motivation. Nothing here
commits to the previous TDEV instantiation; only the *problem statement* and the
*baseline evidence* carry over.

## 4. Contribution framing (intended)

Frame as an **analysis + diagnostic-criterion** paper, not a new-detector paper:

1. The **two-pronged critique** — attention-based detection and mitigation both
   fail under matched controls (length control for detection; answer-prior /
   related-vs-plain control for mitigation).
2. A controlled **semantic-neighbor / alternative-confusion** view that makes the
   mitigation failure measurable.
3. A constructive verification criterion (TBD) that answers both failures.

Honest guardrails carried over from the tear-down:
- Prong 1 is pre-empted by HaloProbe → background only.
- Prong 2 is scoped to attention *amplification* (PAI is inert).
- Any absolute-gain claim must be reported against the honest baselines here.

## 5. Known risks / must-dos (rebuild backlog)
- **Rebuild the constructive method** from the two motivations; do not resurrect
  the torn-down TDEV code without re-justifying it.
- **Prong 2 generality:** current evidence is ClearSight/VisAttnSink/PAI/VCD on
  LLaVA-1.5. Extend the yes-rate / related-FPR decomposition to more mitigation
  baselines (OPERA/DAMRO/M3ID) and ≥1 modern model (Qwen-VL / InternVL).
- **Prong 1 boundary:** keep it as background; its only forward use is to
  motivate a *relative* (target-vs-alternative) readout that survives the control.
- **Object repetition control** (free, borrowed from HaloProbe) for detection.

## 6. Pointers
- Score/metric definitions and the position/length control: [design.md](design.md).
- Repo layout: [project_structure.md](project_structure.md).
- Reproduced baseline results: [experiment_results.md](experiment_results.md).
- Decision history: [iteration_log.md](iteration_log.md).
