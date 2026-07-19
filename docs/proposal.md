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
amplifying attention provides — is being rebuilt as **region-grounded
target-vs-alternative verification** (see §3). Its first Qwen3-VL pilot has
already rejected one tempting but invalid causal score; the alternative-
competition mechanism remains under test rather than established.

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

## 3. Constructive method under test — XEV

**Status after iteration 3 development (2026-07-19):** the first usable method
candidate is implemented, but is not yet a confirmed result. The original
symmetric keep/drop score and exact open/closed candidate intersection are both
rejected. A zero-threshold, counterbalanced pairwise identity certificate
corrects 2/4 baseline false positives with no true-positive veto on the original
120-row development set (balanced accuracy 0.8833 -> 0.9000). It is frozen for a
new disjoint holdout; no positive claim is made until that gate passes.

### 3.1 Principle and scope

**XEV (eXclusive-alternative Evidence Verification)** asks whether a proposed
visual region supports the queried target more strongly than a locally
incompatible object identity. Attention is allowed only to generate high-recall
region proposals; its magnitude is never treated as object evidence.

For an object hypothesis `h`, image/view `I`, and yes/no question `q_h`, define
the model-native margin

    m(h, I) = log p(Yes | q_h, I) - log p(No | q_h, I).

The implementation captures the first-answer-token attention row from Qwen3-VL
layers 18–29, averages heads/layers, and turns the top 30% cumulative visual mass
into connected proposals. Each proposal has a clear-region view `I_keep`, a
region-removed view `I_drop`, and a random same-area control.

### 3.2 Rejected iteration-1 score

The initial score treated

    E(h, R) = m(h, I_keep) - m(h, I_drop)

as region evidence, then subtracted the strongest positive alternative `E`.
This is **rejected**. On balanced POPE-adversarial, target-only `E` has AUROC
0.343 (random-region control 0.181), despite the full-image margin reaching
0.935. The problem is structural: `I_keep` removes most global context while
`I_drop` preserves it. The keep-minus-drop contrast therefore mixes target
evidence with a large, asymmetric context-removal effect and reverses the class
ordering. Oracle candidates cannot repair it (AUROC 0.348).

### 3.3 Same-view criteria: rejected margins, current pairwise certificate

The surviving direction keeps target and alternatives on the **same region view**.
Let `T(t)` contain the target plus semantic equivalents/more-specific supports,
and `C(t)` contain only competing whole-object identities (not hypernyms, parts,
attributes, agents, associated objects, or unrelated objects):

    u_R(t) = max_{h in T(t)} m(h, I_keep)
    c_R(t) = max_{a in C(t)} m(a, I_keep)
    d_R(t) = max(0, c_R(t) - u_R(t))
    s_R(t) = u_R(t) - d_R(t).

The dominance gate `d_R` matters: an alternative is not penalized merely because
it is visible elsewhere or coexists with the target; it must locally outrank the
target. In the iteration-1 post-hoc diagnostic, this reduces related-present FPR
at matched 80% TPR from 0.50 (target-only keep margin) to 0.25 with closed/oracle
candidates. It does **not** improve overall AUROC or held-out POPE behavior, so it
is a mechanism lead, not a claimed method gain. Any behavior correction

    m_final(t) = m(t, I_full) - lambda * max_R d_R(t)

must pass held-out TPR/FPR and MCC gates before being adopted.

Iteration 2 tested a deliberately high-precision hard-veto certificate: require
the full-image answer to be yes; require one candidate to occur in the exact
intersection of target-free open proposals and closed COCO-NPMI proposals; label
it a competing identity; require positive candidate keep margin and at least a
6-logit candidate-over-target excess; then veto yes to no. The threshold was
fixed on iteration 1 before evaluation. On a disjoint 480-row POPE-adversarial
window, the certificate triggered **zero times** and changed no behavior. This
variant is **rejected for zero coverage**. Exact lexical agreement between two
noisy candidate generators is not evidence verification.

Iteration 3 implements that explicit comparison. For every proposed alternative
`a`, ask both directions on the same `I_keep`: whether the main clear-region
identity is better identified as `t` than `a`, and whether it is better
identified as `a` than `t`. If their yes/no margins are `m(t>a,R)` and
`m(a>t,R)`, define the counterbalanced preference

    v_R(t,a) = 0.5 * [m(t>a,R) - m(a>t,R)].

The reversal cancels shared yes-bias and first-mentioned-identity preference;
`v<0` has a natural meaning rather than a fitted threshold. Candidate identity
and visual discrimination remain separate: open/closed sources propose up to
three positive-keep-margin hypotheses, while the pairwise verifier decides. An
exact target mention in the target-free open inventory acts as a support guard.

Pairwise preference alone is unsafe: on the 120-row development set, `v<0`
vetoes 13 true positives and only 2 false positives. The current identity-switch
certificate therefore requires three independent sign/semantic conditions:

    full-image prediction is yes
    m(t, I_keep) < 0
    exists a in C(t): v_R(t,a) < 0.

All thresholds are fixed at zero. This conjunction triggers twice on development
(`dining table -> chair`, `handbag -> backpack`), both true corrections: FPR
0.0667 -> 0.0333, unchanged TPR 0.8333, balanced accuracy 0.8833 -> 0.9000,
and MCC 0.7705 -> 0.8072. This is a development result, not confirmation. The
disjoint holdout must produce at least 3 triggers, correction precision >= 2/3,
lower FPR, TPR loss <= 0.005, and non-decreasing MCC.

### 3.4 Candidate universe

XEV is **not limited to COCO categories**. Candidate sources are deliberately
separated:

- `closed`: top COCO-80 train-set NPMI neighbors, used as a reproducible
  diagnostic baseline;
- `open`: target-free object phrases generated from the clear region, followed
  by an S/C/N semantic relation filter;
- `oracle`: COCO ground-truth objects present in the image, used only as a
  ceiling and kill criterion.

Iterations 1--2 show that vocabulary size is not yet the bottleneck. The open
inventory returns 4.63 phrases/region with zero relation-output parse failures,
but covers only 31.25% of oracle competing identities on related negatives,
versus 81.25% for the closed diagnostic. More importantly, even oracle candidates
do not improve overall behavior. Therefore candidate expansion is frozen until
the region/scoring ceiling is fixed; open vocabulary remains the intended final
scope, not the next engineering target. Exact open/closed intersection is also
too sparse to serve as a certificate (zero triggers on 480 held-out rows).

## 4. Contribution framing (intended)

Frame as an **analysis + diagnostic-criterion** paper, not a new-detector paper:

1. The **two-pronged critique** — attention-based detection and mitigation both
   fail under matched controls (length control for detection; answer-prior /
   related-vs-plain control for mitigation).
2. A controlled **semantic-neighbor / alternative-confusion** view that makes the
   mitigation failure measurable.
3. A constructive target-vs-alternative verification criterion that is accepted
   only if it improves the related-present failure mode without trading away
   plain-negative FPR or target recall.

Honest guardrails carried over from the tear-down:
- Prong 1 is pre-empted by HaloProbe → background only.
- Prong 2 is scoped to attention *amplification* (PAI is inert).
- Any absolute-gain claim must be reported against the honest baselines here.

## 5. Known risks / must-dos (rebuild backlog)
- **XEV confirmatory gate:** evaluate the frozen zero-threshold pairwise
  certificate on a new disjoint POPE window. Only after it passes test
  multi-region proposal recall. Do not revive keep-minus-drop or exact candidate
  intersection.
- **Oracle ceiling before open-vocabulary scaling:** keep candidate expansion
  frozen until oracle/closed dominance improves held-out behavior, not only the
  related-negative slice.
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
