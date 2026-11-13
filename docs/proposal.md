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

The completed **training-free detector audit** (§3) gives a stronger negative
boundary. Removing RoPE slightly reduces position correlation but does not
improve an attention probe beyond noise, sink deletion significantly harms it,
and the fixed label-free local-gain score is below chance. **Visual Value
Evidence** (§4) initially appeared to cross that boundary: its supervised probe
reached 0.887 within-position AUROC versus 0.799 for the available two-layer
attention-plus-confidence probe. The mandatory capacity control overturns that
attribution. An equal-dimension, all-layer/all-head visual-allocation trajectory
reaches 0.881 versus 0.878 for VVE-core under the same seed, and full VVE exceeds
it by only 0.004 with a bootstrap interval crossing zero. Label-free VVE also
fails at 0.552. VVE is therefore retained as a predictive but capacity-confounded
audit, not the constructive method. The two-pronged critique remains the paper
spine while a new constructive route is selected on independent evidence.

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
- A controlled negative result for **content-only, position-deconfounded
  attention detection**: preserving heads, removing RoPE, deleting sink
  patches, and contrasting nearby rows does not beat IC or the same-run raw
  per-head probe. This sharpens the critique but is not a constructive method.
- A controlled **capacity-confound result for source-decomposed visual value
  evidence**: the apparent gain over a narrow attention baseline disappears
  against an equal-dimension all-layer allocation probe. This constrains future
  detector claims but is not itself a successful constructive method.

## 3. Completed constructive audit — Position-Free Detection Failed

**Status after the 2026-07-20 full audit:** XEV remains closed; TCS, NEA, EEA,
and PCPA were cancelled before their new-stage pilots. PCPA violated the desired
deployment constraint because it fine-tuned the LVLM. The subsequent frozen-
LVLM position-free attention detector failed both pre-registered gates, so this
constructive route is also closed.

The audit asked whether removing the explicit position prior at the attention
source could support either a fixed unsupervised score or a small external
supervised classifier. The LVLM stayed frozen and the detector only read one
ordinary teacher-forced forward over each already generated caption. The answer
is no under the locked design and controls below.

### 3.1 What "removing position" must mean

No-RoPE alone is not enough. RoPE removal deletes one geometric term from Q/K,
but later mentions are still more likely to hallucinate and model states may
encode response depth indirectly. A valid detector must therefore pass three
separate controls:

1. **Source control:** recompute per-head visual attention from pre-RoPE Q/K;
   remove key-norm sink patches and renormalize only over the remaining visual
   tokens. The original model forward and logits are untouched.
2. **Learning control:** never provide `gen_pos`, absolute token index, response
   length, or position-bin ID to a classifier. Supervised weights are estimated
   from hallucinated/grounded differences matched inside the same position bin,
   with all mentions from one image kept in one fold.
3. **Evaluation control:** overall AUROC is secondary. Within-bin, matched-pair,
   and train-fold residual AUROC are mandatory, together with per-bin dispersion
   and an explicit position-only baseline.

The detector is training-free with respect to the LVLM. The unsupervised score
uses no hallucination labels. The supervised variant may fit a lightweight
external probe, but must be named and reported separately rather than called an
unsupervised training-free method.

### 3.2 Tested clean per-head grounding gain

Let `q_t` be the generating row for an object mention and let `q_n` be up to
eight nearest generated non-object rows within 16 tokens. At late decoder layer
`l` and head `h`, compute pre-RoPE visual attention

    p_t^(l,h) = softmax(q_t k_vis^T / sqrt(d))
    p_n^(l,h) = mean_r softmax(q_r k_vis^T / sqrt(d)),

after deleting visual keys detected as sinks from their pre-RoPE key norms.
Renormalize each distribution over the retained visual tokens. The local row is
matched in response context and therefore controls much of the remaining depth,
syntax, and image-level salience without using labels.

For each head measure two complementary gains:

    g_jsd^(l,h) = JSD(p_t^(l,h), p_n^(l,h))
    g_conc^(l,h) = top5(p_t^(l,h)) - top5(p_n^(l,h)).

A grounded object need not affect every head. The fixed unsupervised score keeps
the strongest quartile of positive gains instead of averaging all heads first:

    G(t) = mean_top25%_h [rank(g_jsd^(31,h))
                          + rank(max(0, g_conc^(31,h)))]
    S_unsup(t) = -G(t).

Ranks are estimated from an unlabeled calibration split and frozen for the test
fold. Higher `S_unsup` means more likely hallucinated. Layer 31 is fixed from the
existing depth diagnostic, not selected on the new outcomes; layer 22 is an
ablation. Raw post-RoPE and no-RoPE-without-sink versions are matched controls.

The supervised ceiling uses the same per-head features plus object-row entropy,
top-1, and top-5 concentration. A linear probe is fitted on within-position-bin
hallucinated-minus-grounded feature differences, with reversed differences as
negative examples and no intercept. This conditional objective removes the
common position component before estimating head weights. Predictions are
strictly image-grouped out of fold.

### 3.3 First dataset and leakage contract

Use the existing LLaVA-1.5-7B COCO-val2014 caption cache: 16,426 first object
mentions from 4,977 images, with CHAIR labels and a hallucination rate of 0.244.
This permits direct comparison to IC (`0.686` within-bin) and the raw late-layer
per-head probe (`0.726`) without regenerating captions or changing labels.

All calibration, classifier fitting, standardization, position residualization,
and head weighting are performed inside five image-grouped folds. Mentions from
one image never cross train/test. The unsupervised method may use unlabeled
training-fold feature distributions only; it may not choose layers, heads,
quantiles, signs, or fusion weights from test labels. The supervised probe may
use train-fold labels only through same-bin matched differences.

COCO/CHAIR is a feasibility dataset, not evidence of broad transfer. A positive
route must next reproduce on a modern model and an independent hallucination
dataset. Qwen3-VL is the preferred second model; no mitigation benchmark is
opened during this detector iteration.

### 3.4 Novelty boundary from the refreshed literature

- **HaloProbe (2026)** already identifies position/repetition confounding and
  trains per-head attention-plus-confidence probes across models. Consequently
  neither "position matters" nor "a supervised head probe works" is ours.
- **DHCP, Beyond ADS/CGC, and related supervised detectors** already train on
  per-layer/per-head attention or representation tensors. A linear external
  classifier is not itself novel.
- **PAS, SVAR, SAE, TVER, and LaSCD** cover label-free attention mass, shape,
  entropy, segmentation, and layer-energy scores. Our reproductions show their
  mean/single-score variants collapse under position control.
- **IC, entropy, and NLL** are strong non-attention uncertainty controls. A new
  attention score must beat IC rather than merely exceed chance.

The only defensible method boundary would be the conjunction of **source-level
de-rotation, sink deletion, local-row differencing, and sparse per-head
aggregation**, validated under a position-conditional learning objective. That
boundary is provisional. If the unsupervised score does not beat IC or the
supervised clean probe does not beat the raw layer-31 probe, this becomes a
negative analysis rather than a method claim.

### 3.5 Full result and decision

Iteration 1 extracted raw, no-RoPE, and no-RoPE-plus-sink-clean per-head features
at layers 22 and 31 for all 16,426 mentions, then evaluated the fixed
unsupervised grounding gain and an image-grouped position-matched linear probe.

| Detector | Overall | Within-bin | Matched | OOF residual |
|---|---:|---:|---:|---:|
| IC baseline | 0.7761 | 0.6856 | 0.7029 | 0.6334 |
| unsupervised raw local | 0.3965 | 0.4646 | 0.4678 | 0.4765 |
| unsupervised no-RoPE local | 0.4164 | 0.4632 | 0.4700 | 0.4820 |
| unsupervised clean local | 0.3881 | 0.4553 | 0.4633 | 0.4766 |
| supervised raw conditional probe | 0.8546 | 0.7405 | 0.7393 | 0.6762 |
| supervised no-RoPE conditional probe | 0.8462 | **0.7431** | 0.7404 | 0.6633 |
| supervised clean conditional probe | 0.8140 | 0.6998 | 0.7072 | 0.6491 |

The unsupervised route advances only if it reaches at least `0.70` within-bin
AUROC, exceeds IC by at least `0.01`, and improves both matched-pair and residual
AUROC. The supervised route advances only if the clean conditional probe reaches
at least `0.75` within-bin AUROC, beats the raw layer-31 probe by at least `0.02`,
and does not obtain that gain from overall-position correlation. For either
route, no-RoPE and sink cleaning must each have a positive matched ablation.

Both gates failed. No-RoPE minus raw is only +0.0026 within-bin and its paired
95% interval crosses zero (`[-0.0102, 0.0174]`); residual AUROC decreases.
Sink deletion is reliably harmful: clean minus no-RoPE is -0.0434 with interval
`[-0.0574, -0.0302]`. The unsupervised local score loses even to its clean
object-only ablation (0.4553 vs. 0.5236 within-bin), directly rejecting the
local-grounding-gain hypothesis.

Per the frozen rule, failure of both gates closes attention-based constructive
detection. The no-RoPE probe is retained only as a supervised external ceiling:
it is not unsupervised, does not meet the 0.75 gate, does not significantly beat
raw attention, and overlaps existing per-head probe work. The full protocol is
[position_free_detector_iteration_1.md](iteration_protocols/position_free_detector_iteration_1.md).

### 3.6 Rejected pre-experiment design — PCPA

Pair-Centered Preference Alignment proposed a same-image two-query preference
loss with a zero-common-mode constraint. It was cancelled before manifest
construction or LoRA training because it modifies the LVLM, contrary to the
required training-free deployment setting. Its protocol remains a historical
design record and must not be run as the active route.

### 3.7 Rejected pre-experiment design — EEA

Existential Evidence Aggregation proposed an unnormalized target-patch matcher
with log-mean-exp set pooling. It was cancelled before its frozen-feature pilot.
Although it avoided attention and contrastive decoding, it introduced a
training-time detector and made the claim depend on smooth pooling beating
established MIL, global-pooling, and patch-text controls. More importantly, it
did not directly prevent the common `Yes` shift measured in Prong 2. Its
pre-registered protocol is retained as a historical design record.

### 3.8 Rejected pre-experiment design — NEA

Null-Evidence Attention proposed a learned null key/value inside selected
visual-attention heads while preserving the original visual mass. It was
cancelled before Stage A. Although implemented inside a head, its behavioral
effect is still a learned absence-conditioned subtraction from the ordinary
answer, so its distinction from contrastive/calibrated decoding is mainly where
the correction is applied. Generic null/sink attention is established prior
art, and the learned null value can become an unconstrained answer-steering
direction. The retained protocol documents the rejected design; it must not be
implemented as the active method.

### 3.9 Rejected pre-experiment design — TCS

Target-Causal Selectivity proposed target-removal sensitivity and neighbor-
removal invariance using inpainted triples. It was cancelled before Stage A for
three reasons. First, four-view binary supervision already enforces the desired
labels, so the claimed paired regularizer may reduce to ordinary counterfactual
SFT. Second, target and neighbor removals introduce unequal artifacts and can
change scene semantics, weakening the causal interpretation. Third, oDPO already
uses target removal and VISE/VES-RFT occupy nearby perturbation/sensitivity
training. The design remains documented in its protocol and literature memo but
must not be implemented as the active method.

### 3.10 Closed route evidence — XEV

Everything below this heading is the preserved design/failure trace and is
superseded by §§3.1--3.5. Terms such as "current" and "next" describe the state
at that historical iteration; they are not active instructions.

**XEV (eXclusive-alternative Evidence Verification)** asks whether a proposed
visual region supports the queried target more strongly than a locally
incompatible object identity. Attention is allowed only to generate high-recall
region proposals; its magnitude is never treated as object evidence.

For an object hypothesis `h`, image/view `I`, and yes/no question `q_h`, define
the model-native margin

    m(h, I) = log p(Yes | q_h, I) - log p(No | q_h, I).

The implementation captures the first-answer-token Q/K from Qwen3-VL layers
18–29 before multimodal RoPE, recomputes the localization-only attention row,
averages heads/layers, and turns the top 30% cumulative visual mass into connected
proposals. The original post-RoPE forward still produces the model answer and
Yes/No margin; removing RoPE changes only region localization. A `raw` map remains
available for controlled ablation. Each proposal has a clear-region view
`I_keep`, a region-removed view `I_drop`, and a random same-area control.

#### XEV iteration 1: rejected keep/drop score

The initial score treated

    E(h, R) = m(h, I_keep) - m(h, I_drop)

as region evidence, then subtracted the strongest positive alternative `E`.
This is **rejected**. On balanced POPE-adversarial, target-only `E` has AUROC
0.343 (random-region control 0.181), despite the full-image margin reaching
0.935. The problem is structural: `I_keep` removes most global context while
`I_drop` preserves it. The keep-minus-drop contrast therefore mixes target
evidence with a large, asymmetric context-removal effect and reverses the class
ordering. Oracle candidates cannot repair it (AUROC 0.348).

#### XEV iterations 2--15: self-verification evidence trail

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

It fails that holdout decisively. On POPE-adversarial rows 600--1079, the frozen
rule triggers 6 times, all on true positives (`motorcycle -> car`, `skateboard ->
fence`, `spoon -> knife`, `truck -> vehicle`, `car -> concrete curb`, `bed ->
couch`). TPR falls 0.8250 -> 0.8000, FPR stays 0.0833, balanced accuracy falls
0.8708 -> 0.8583, and MCC falls 0.7448 -> 0.7216. **Reject the single-region hard
veto.** Its local comparison may correctly identify another visible object, but
that does not imply the queried target is absent elsewhere.

Iteration 5 separates local contradiction from global coverage. A local proposal
requires a top-3 positive-keep candidate from the closed NPMI inventory, relation
C, and `v_R(t,a)<0`. Closed candidates are a current high-precision proposal
gate, not the intended vocabulary boundary; open phrases are still generated
and may name the same candidate. Only for a local proposal, generate two
target-free full-image inventories with different prompts. Let `g_j(t)=1` when
inventory `j` contains a lexical target equivalent or an S relation. The veto is

    local(t,a,R) and not [g_1(t) and g_2(t)].

The conjunction is deliberately conservative: one inventory hallucinated
`bicycle` for an absent bicycle and one returned generic `bag` for an absent
handbag, but neither false support repeated in the second inventory. On the 10
selected prior triggers, both inventories support all 5 true targets and fail
to jointly support all 5 absent targets, yielding 5 corrections and no TP vetoes.
Retrospectively this would correct 2/4 development FPs and 3/20 iteration-4 FPs
without recall loss, but both sets informed the design and are closed. The rule
must pass unchanged on rows 1080--1559 before adoption.

It does not pass. On rows 1080--1559, 11 local proposals invoke the guard; 7 are
suppressed and the remaining 4 are all true positives (`backpack -> suitcase`,
`traffic light -> train`, `train -> car`, `handbag -> backpack`). TPR falls
0.7958 -> 0.7792, FPR stays 0.0667, balanced accuracy falls 0.8646 -> 0.8563,
and MCC falls 0.7362 -> 0.7211. **Reject absence from two generated inventories
as image-level absence evidence.**

The next local criterion must require an actual directional answer switch,
`m(t>a,R)<0` and `m(a>t,R)>0`, not merely their difference below zero. Global
inventory support must use lexical identity plus an explicit COCO alias table,
not the unreliable generated S relation. This preserves open-vocabulary
inventory phrases while making support matching auditable for the POPE pilot.

Iteration 7 implements this repair. The alias table covers only transparent
surface variants (for example `dining table/table`, `couch/sofa`,
`motorcycle/motorbike`, `tv/television`) and deliberately excludes generic
hypernyms such as `bag` for handbag/backpack. On the now-development-only
iteration-6 window, the four prior TP vetoes fail the directional sign test. Two
baseline FPs pass it and are no longer falsely suppressed by parts/hypernyms:
`car -> truck` and `backpack -> suitcase`. Retrospective behavior is FPR 0.0667
-> 0.0583, unchanged TPR 0.7958, balanced accuracy 0.8646 -> 0.8688, and MCC
0.7362 -> 0.7455. This is not a held-out result.

Iteration 8 confirms it on rows 1560--2519 (960 rows, 160 images). Seven strict
local proposals invoke global support; three real targets are unanimously found
and suppressed. The four remaining certificates correct `chair -> couch` once
and `cup -> wine glass` twice, while incorrectly vetoing one real `couch` as
chair because the first 12-item inventory omits a later `sofa` mention. The
pre-registered gate passes: 4 triggers >=3, precision 0.75 >=2/3, FPR decreases,
TPR drop is 0.0021 <=0.005, and MCC increases. **Adopt this frozen rule as XEV
v1**, while reporting the single support-recall failure and low 0.42% action rate.

Iteration 9 audits that low action rate rather than treating it as an attention
localization failure. The v1 funnel is 441 baseline-yes rows -> 242 rows with a
pairwise comparison -> 176 negative-preference rows -> 7 strict local proposals
-> 3 global-support suppressions -> 4 actions. On a paired 120-row development
window, no-RoPE changes localization materially (only 42/120 proposal boxes are
identical to raw; mean box IoU 0.628), but raw and no-RoPE both yield one strict
local proposal, which global support suppresses, and zero actions. Therefore
no-RoPE is retained as localization hygiene, not a coverage fix. The dominant
bottleneck is the conjunction of closed-source provenance, C relation, and strict
directional signs. The next method iteration must improve this proposal recall
without weakening the independent image-level support check.

Iteration 10 tests exactly that broader local proposal on the iteration-9
development window. Allowing any open/closed non-S identity with the same strict
directional switch raises local proposals from 1 to 21. The proposed global
guard combines `m(t,I_drop)>=0` with unanimous target mentions in two target-free
full-image inventories. It fails: 18/21 proposals are suppressed, while the
remaining three are all true-target vetoes (`cell phone`, `backpack`, and
`handbag`). None of the five baseline false positives is corrected. Inventory
support is especially misaligned: it often regenerates the queried absent object
from scene context and misses small real targets. Complement support is
directionally better but still yields only 2 corrections among 6 actions in a
post-hoc source ablation. **Reject the support union and do not threshold-tune it.**

The resulting design constraint is sharper. A target-free full-image list is
still free-form generation, not grounded verification; a single region
complement is still an ambiguous global view. The next constructive hypothesis
must cover the image with multiple diverse attention proposals, verify each
proposal locally with counterbalanced target-vs-alternative questions, and
aggregate only verified regional identities. Attention continues to propose
where to inspect and contributes no decision score. A local conflict can support
a veto only when no independently proposed region provides a positive target
identity switch.

Iteration 11 tests that multi-region hypothesis without global generation. It
extracts three spatially distinct residual no-RoPE proposals and uses mirrored
strict pairwise directions for local target support. The second proposal helps:
among true-positive conflict rows, support recall rises from 0.292 at K=1 to
0.750 at K=2. But K=3 adds conflicts without adding support, and the final rule
still makes 14 actions with only 4 corrections and 10 true-positive vetoes
(precision 0.286). TPR falls 0.9000 -> 0.7333 and MCC 0.8168 -> 0.7402.

This failure exposes a deeper structural issue than proposal count. Five of the
ten vetoed real targets are explicitly named by a target-free regional inventory,
and eight have positive target keep margin in at least one proposal, yet no
region passes the strict target-support direction. Ten rows even produce target
support against one candidate and target conflict against another candidate in
the same box. Attention-selected rectangles contain multiple valid identities;
there is no single "main object" for a candidate-relative tournament to recover.
**Reject residual multi-region attention coverage.** The next proposal mechanism
must produce target-focused or object-pure regions before local verification;
adding more coarse attention boxes or tuning pairwise margins is not justified.

Iteration 12 tests target-focused proposals from Qwen3-VL's native 2D grounding.
The returned/missing box and generated label are ignored; positive support
requires a target-free regional inventory to recover the target, positive target
margin, and no strict C/N conflict. On the 14 failure-enriched iteration-11
actions, every case yields a valid box, but only `wine glass` and `umbrella` pass
the full cycle. Support is precise (2/2 real targets and 0/4 absent targets) but
recall is only 2/10, so the pre-registered stage-1 gate fails and the 49-row
development extension is not run.

An oracle-only localization audit keeps the proposal lead alive: for the eight
true targets with matching COCO detection annotations, maximum grounding-box IoU
has mean 0.607 and median 0.576. The verifier, not coordinates alone, is now the
limitation. The current keep view preserves the full canvas, leaving a small
grounded object small after model preprocessing. It also rejects real bench and
backpack boxes because an overlapping person produces an N-relation conflict.
The next test should crop and zoom the grounded box before target-free
re-identification, and should reserve C for mutually exclusive box identities;
neither change may be retrofitted to iteration 12.

Iteration 13 performs that frozen paired test. Exact crops recover real
`wine glass`, `bench`, `cell phone`, and `backpack` with no support on four
absent targets, improving full-cycle recovery from 2/10 to 4/10 at precision
1.0. The gate still fails. Moreover, crop support is not a superset of global
support: the 19x7 umbrella box is recognized on the full-canvas clear view but
becomes `cloud` when isolated, while a 15x12 cell-phone crop succeeds. This
rules out one universal verification scale. The next constructive hypothesis is
a fixed geometric context pyramid in which every scale is independently
verified; no scale is selected by grounding scores or downstream labels.

Iteration 14 evaluates that fixed pyramid on fresh rows 2640--2759. All 41 local
conflicts have valid boxes and 29 receive target support. Only two rows repeat
the same open C alternative over multiple scales; both are real targets and are
correctly blocked by target support. No absent row forms the registered
certificate, so the rule takes zero actions. This rules out lexical alternative
repetition as the cross-scale invariant. A more defensible invariant is the
fixed queried hypothesis itself: require negative target margins at multiple
scales, plus one target-free C alternative that independently passes the strict
directional identity test.

Iteration 15 stage 1 freezes that conjunction. A box must have negative target
margins at two or more scales and at least one target-free open C alternative
with a strict directional switch; any scale with cycle-consistent target support
blocks the action. On ten historical development conflicts, it corrects absent
`handbag` and `bicycle` with no TP veto. In the 480-row replay, FPR falls 0.0833
-> 0.0750 at unchanged TPR and MCC rises 0.7448 -> 0.7538. Because these rows are
not fresh, the rule must remain a candidate until it passes a separately
committed 480-row evaluation unchanged.

The actual POPE-adversarial remainder contains 240 rather than 480 rows; this
boundary was corrected in a committed amendment before result inspection. On
those 240 rows, 14 strict local conflicts yield only one final action. It vetoes
a real `vase` because target-free `pot`/`flowerpot` crops pass the visual switch
and the relation filter treats these near-equivalent names as competing. Metrics
worsen and the gate fails. No untouched adversarial rows remain, so this failure
cannot be repaired or revalidated on the same split.

#### XEV candidate-universe audit

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
Iteration 9 sharpens the diagnosis: no-RoPE increases pairwise-scored rows from
27 to 31 and negative-preference rows from 21 to 25 on the paired 120-row window,
but both maps produce only one strict local proposal. Candidate/semantic gating,
not the position prior, currently determines action coverage.
Iteration 10 shows that removing those candidate/semantic restrictions is
sufficient to recover local-conflict coverage (21/120), including candidates
from the open inventory, but is not sufficient for safe image-level correction.
Candidate acquisition is therefore no longer the immediate bottleneck; grounded
multi-region target coverage is.
Iteration 11 then shows that residual top-K attention coverage alone does not
provide that grounding. It always returns three boxes and evaluates 436 pairwise
comparisons, but the third box adds no target-support row. The immediate
bottleneck is proposal purity and target coverage, not candidate quantity.
Iteration 12 shows that native target grounding materially improves proposal
purity by oracle IoU, but the full-canvas clear-region rendering wastes that
precision. Candidate acquisition on a zoomed object crop is the next bottleneck
test.
Iteration 13 shows that exact crops recover additional small targets but can
discard disambiguating context. Candidate acquisition must therefore operate
over a predefined object-to-context scale space, not one chosen crop.
Iteration 14 shows that candidate *names* cannot be required to stay identical
through that scale space. Cross-scale replication should apply to target
evidence; an alternative remains a separately verified one-view certificate.
Iteration 15 stage 1 supports this decomposition: repeated target-negative signs
plus one explicit alternative recover two historical false positives without a
real-target veto. Fresh coverage and precision are still unknown.
Stage 2 rejects that decomposition as a complete method: its only fresh action
crosses an unstable semantic boundary (`vase` versus `pot/flowerpot`). A future
grounded branch needs an independently validated equivalence/taxonomy layer;
spatial evidence alone has reached its limit on this benchmark.

## 4. Completed constructive audit — Visual Value Evidence

VVE tested **what visual attention carries**, rather than how much
or where it attends. For each layer, head, and object-generating row, decompose
the exact visual part of the attention output:

    c_vis^(l,h) = W_O^(l,h) sum_(j in V) alpha_j^(l,h) v_j^(l,h).

Project this residual contribution into the emitted object's unembedding
direction and measure both its direct alignment and its normalized logit-lens
gain. A grounded mention should receive target-specific positive visual support;
a hallucinated mention may allocate attention to real image regions while
transporting content that does not support the emitted target. This is the
candidate constructive counterpart to the paper thesis: allocation is not
verification, but value-bearing, target-aligned residual evidence might be.

The method first **removes RoPE from Q/K**, recomputes causal attention over the
complete prefix, and then applies those weights to the original values and
output projection. The residual stream and emitted target token remain fixed.
Raw post-RoPE VVE is an experimental ablation and the exact branch used for
causal intervention; it is not the headline detector. Position, response
length, repetition, and object identity are excluded from the strict detector;
training remains conditionally balanced within position bins and all evaluation
is image-grouped OOF.

Two variants are separated explicitly:

1. a label-free late-layer joint target-gain score;
2. a supervised external MLP over per-head value evidence and decoder confidence.

The LVLM stays frozen and generation is unchanged. The supervised probe is
training-free only with respect to the LVLM and must not be called an
unsupervised method. The locked first-iteration formulas, controls, and gates are
in [visual_value_evidence_iteration_1.md](iteration_protocols/visual_value_evidence_iteration_1.md).

The novelty claim is deliberately narrow. Direct logit attribution, logit lens,
and source decomposition are established primitives. HaloProbe does not inspect
value content; Owl/VTACR averages attention weights despite using the word
"contribution"; VISTA and Overthinking trace layerwise logits without isolating
their visual source. The tested unit was their conjunction: source-decomposed,
target-aligned visual residual evidence for position-controlled object
hallucination detection. The capacity result in §4.3 prevents claiming that
conjunction as an empirically distinct detector signal.

**Tested method center.** The core VVE object is the all-layer, all-head trajectory of
normalized target gain. Source mass, value norm, direct dot/cosine, confidence,
and attention shape are controls or diagnostic ceilings; they are not separate
method ingredients. The external MLP is a readout, not the conceptual
contribution. Iteration 2 was therefore designed to validate this fixed target-
gain trajectory rather than add features to chase a higher AUROC.

### 4.1 Iteration 1 result and decision

The full run covers 16,426 object mentions from 4,977 images under the locked
five-fold image-grouped protocol. The primary results are:

| Detector | Overall | Within-bin | Matched | OOF residual |
|---|---:|---:|---:|---:|
| confidence-only MLP | 0.7149 | 0.6724 | 0.6733 | 0.6325 |
| attention + confidence MLP | 0.8561 | 0.7988 | 0.7856 | 0.6679 |
| label-free no-RoPE VVE | 0.6020 | 0.5518 | 0.5623 | 0.5283 |
| supervised no-RoPE target-gain trajectory | 0.9119 | **0.8793** | 0.8654 | 0.7153 |
| supervised raw VVE | 0.9112 | 0.8821 | 0.8712 | 0.7138 |
| supervised no-RoPE VVE | 0.9142 | **0.8872** | 0.8749 | **0.7271** |
| supervised no-RoPE VVE + confidence | **0.9151** | **0.8873** | **0.8764** | 0.7219 |

The supervised route passes every pre-registered advance gate. Evidence-only
no-RoPE VVE improves over attention plus confidence by 0.0884 within-bin; the
image-bootstrap 95% interval is `[0.0763, 0.1026]`. It remains strong when only
the causal pre-object row is used (0.8747), so the result is not created solely
by consuming the target token. Inside the same position-bin and object category,
VVE reaches 0.8581 versus 0.7648 for attention plus confidence. When entire
object categories are excluded from detector training, VVE reaches 0.8040
versus 0.7539; the gain interval is `[0.0350, 0.0665]`.

Two limits are equally important. First, the fixed label-free score fails, so
iteration 1 yields a **supervised external detector over a frozen LVLM**, not an
unsupervised detector. Second, removing RoPE is a fixed position-prior control,
not the source of the empirical gain: removed-RoPE minus raw VVE is only +0.0052
and its interval crosses zero (`[-0.0009, 0.0124]`). The method still removes it
by definition; the novelty and predictive gain are assigned to target-aligned
visual values. Confidence adds effectively nothing to VVE.

The current +0.0884 comparison is not yet a fully capacity-matched attribution.
Full VVE supplies 10,240 inputs (about 2.62M MLP parameters), whereas the reused
two-layer attention-plus-confidence cache supplies 522 inputs (about 0.13M
parameters). The 2,048-dimensional target-gain-only detector already reaches
0.8793, so the result is not created by the four auxiliary VVE features. This
made an all-layer, equal-dimension attention control mandatory before assigning
the gain to value content; §4.3 reports that failed attribution test.

This iteration-1 decision was superseded by the registered Stage-A0 and first
Stage-A capacity result in §4.3. The locked audit is in
[visual_value_evidence_iteration_2.md](iteration_protocols/visual_value_evidence_iteration_2.md).

### 4.2 Same-data comparison with prior detectors

The following methods all use the same frozen LLaVA-1.5-7B captions, 16,426
COCO/CHAIR object mentions, hallucination labels, and position-bin metric. They
are separated by supervision; comparisons across the two blocks are descriptive,
not claims that label supervision is free.

| Label-free detector | Overall | Within-bin |
|---|---:|---:|
| PAS | 0.8349 | 0.5928 |
| SVAR | 0.8053 | 0.5678 |
| NLL | 0.7114 | 0.6360 |
| Entropy | 0.7206 | 0.6366 |
| IC | 0.7761 | **0.6856** |
| TVER | 0.8303 | 0.5655 |
| SAE (Mask2Former) | 0.7833 | 0.5747 |
| LaSCD raw layer energy | 0.5739 | 0.5000 |
| label-free VVE | 0.6020 | 0.5518 |

| Supervised external detector | Overall | Within-bin |
|---|---:|---:|
| ADS-only, 32-layer OOF | 0.8379 | 0.6660 |
| CGC-only, 32-layer OOF | 0.8372 | 0.6688 |
| ADS+CGC, 32-layer OOF | 0.8661 | 0.7185 |
| raw per-head attention probe | 0.8546 | 0.7405 |
| HaloProbe-style attention + confidence | 0.8561 | 0.7988 |
| **VVE-core, RoPE removed (ours)** | **0.9119** | **0.8793** |
| full VVE diagnostic | 0.9142 | 0.8872 |

This established a strong same-data feasibility result for supervised VVE-core,
not value-specific validation. The HaloProbe-style row is our local two-layer
probe, not HaloProbe's official 0.935 from a different evaluation distribution.
The equal-dimension control in §4.3 shows why this table cannot support a VVE
method claim by itself.

### 4.3 Position-uniform and capacity-matched outcome

The one registered position-uniform objective does not fix the late-bin
degradation. Against the same-seed size-weighted fit, VVE-core macro-bin AUROC
moves only 0.8298 to 0.8305 and full VVE falls from 0.8369 to 0.8306; their
80--139 AUROCs also fall from 0.7992 to 0.7938 and 0.8213 to 0.8123. The
optimization fails its locked +0.015 gates and is rejected. The original
size-weighted objective remains the reporting reference; no blended weighting,
position head, or threshold sweep is allowed on this dataset.

More decisively, the first registered Stage-A seed fails the capacity gate.
With identical 2,048-dimensional layer/head/row coverage, classifier, folds,
and initialization, visual allocation mass reaches **0.8812** within-bin versus
**0.8783** for VVE-core. The difference is -0.0029 with image-bootstrap 95% CI
`[-0.0093, 0.0036]`. Full VVE reaches 0.8855, only +0.0043 over allocation mass,
with CI `[-0.0002, 0.0094]`. Because Stage A required VVE-core to beat allocation
mass at every seed and by at least 0.05 on average, one registered losing seed
already makes the gate impossible to pass. Further seeds cannot rescue the
registered decision.

A post-hoc explicit balance audit gives the operational number hidden by the
within-bin summary. Downsampling each valid 10-token bin to equal grounded and
hallucinated counts retains 7,072 mentions per draw. Across 1,000 frozen-OOF
draws, all-layer allocation mass reaches **0.8373 AUROC / 0.7605 zero-threshold
accuracy**; VVE-core reaches 0.8267 / 0.7478 and full VVE 0.8398 / 0.7624. The
two-layer attention-plus-confidence probe reaches 0.7411 / 0.6775. Thus the
supervised per-head signal survives actual position balancing, but neither VVE
nor allocation approaches 0.9 on this local balanced test.

The pooled balanced result is not position-uniform. Allocation-mass AUROC drops
from 0.9404 at positions 0--9 to 0.6236 at 140--149, while zero-threshold
accuracy drops from 0.8187 to 0.5764. The decline is already visible in per-bin
AUROC, so it cannot be explained only by one globally miscalibrated threshold.
The detailed distribution is reported in `docs/experiment_results.md`.

**Decision:** close VVE as the constructive detector route. Its high OOF AUROC
is real, but the evidence does not identify target-aligned value content as the
cause; all-layer allocation plus a high-capacity supervised readout explains
essentially the same labels. Do not run the downstream target/source,
intervention, Qwen3-VL, or transfer stages as VVE confirmation. The extractor
and results remain useful for the paper's audit and for designing a genuinely
orthogonal next hypothesis.

### 4.4 Registered explanation audit

The strong No-RoPE all-layer allocation control creates a separate mechanistic
question: whether its performance comes from a few stable specialized heads, a
distributed per-head pattern, or a simple layerwise mean-mass trajectory. The
locked audit in
[allocation_probe_mechanism_audit.md](iteration_protocols/allocation_probe_mechanism_audit.md)
combines fold-stable model-parameter rankings, held-out gradient effects,
layer/row interventions, mean-versus-head refits, and train-fold-only sparse-head
refits. Every result is reported by position bin. Weight magnitude alone is not
accepted as evidence of a special head, and this explanation audit does not
reopen allocation as the constructive method.

**Outcome.** The refit exactly reproduces the retained OOF scores. The signal is
not a layerwise mean trajectory: mean-only allocation reaches 0.7966 within-bin,
whereas head deviations after removing every layer/row mean retain 0.8805 versus
0.8812 for full allocation and preserve at least 90% of the full gain in all 14
stable position bins. Seventeen layer-head pairs satisfy the strict fold/gradient
stability rule, led by `(L19,H13)`, `(L19,H29)`, `(L8,H25)`, `(L30,H9)`, and
`(L23,H28)` using zero-based indices. They are predictive candidates, not
individual causal bottlenecks: removing any entire layer lowers aggregate
within-bin AUROC by at most 0.004, and no layer or eight-layer band passes the
necessity/sufficiency gate.

The registered sparse-head claim also fails. Top-32 refits reach 0.8534 and
retain 90% of the full gain in only 8/14 stable bins, despite stable rankings and
a clear advantage over random-32. Top-64 reaches 0.8694 and retains the gain in
14/14 bins, indicating concentration across dozens of heads but substantial
redundancy rather than a small decisive head set. Removing the pre row leaves
0.8639, while removing the post row leaves only 0.7733; much of the learned
signal is post-target attention reorganization. The defensible mechanism is a
distributed cross-layer relative-head pattern, dominated by the post row, not
target-object verification localized to a few heads.

**2026-07-28 confirmatory factorial.** A separately registered,
capacity-controlled `layer breadth × head treatment` experiment on all 16,426
mentions confirms both main components of that interpretation. With identical
64-dimensional fold-local transforms and the same grouped five-fold detector,
all-layer/per-head reaches 0.8768 within-position-bin AUROC. It exceeds the
nested train-selected single layer (0.8404), every fixed eight-layer band
(0.8062--0.8682), and both fixed halves (0.8679/0.8697); all seven registered
paired cluster-bootstrap contrasts have Holm-corrected intervals above zero.
It also exceeds the all-layer head mean (0.8030) by 0.0736
(`95% CI [0.0642, 0.0836]`, Holm `p=0.0020`). Removing each layerwise head
mean and retaining only head deviations reaches 0.8797, slightly above the
full per-head representation, so the head result is specifically relative
head identity rather than more raw mass.

The stricter positive layer-by-head interaction gate fails: the registered
interaction is negative (`-0.1824`, `95% CI [-0.2035, -0.1615]`). Thus the
supported claim is two additive, partly redundant requirements -- broad
cross-layer coverage and within-layer head differences -- not a positive
synergy between them. The full-mechanism gate is therefore recorded as failed
even though both component gates pass. This narrows, rather than reopens, the
constructive claim.

### 4.5 Registered spatial-allocation audit

The current allocation probe discards the 24x24 visual layout by reducing every
layer/head/row map to total visual mass. The next local audit tests whether that
distributed fingerprint contains spatial organization that mass alone cannot
represent. RoPE removal is fixed by method definition. The two independently
controlled additions are (1) a top-10% 8-neighbor coherence statistic, compared
against equal-dimensional top-10% mass concentration, and (2) deletion of the
existing low-key-norm visual sink set. Both are evaluated alone and in matched
mass fusions over all layers, heads, and pre/post rows.

The locked protocol is
[no_rope_spatial_allocation_audit.md](iteration_protocols/no_rope_spatial_allocation_audit.md).
It treats connectivity, concentration, and sink deletion as separate factors,
uses the existing image-grouped position-conditional MLP, and reports every
position bin plus literal position-balanced tests. This remains a same-data
feasibility audit. It can reopen a spatial feature family only after beating
equal-dimensional concentration and mass controls; it cannot by itself turn an
allocation fingerprint into target-object verification.

**Outcome.** The registered spatial family fails. No-deletion
`mass+connectivity` reaches 0.8601 within-bin, below both mass alone (0.8812)
and the equal-dimensional `mass+top10` control (0.8774). Connectivity minus
top-10% concentration is -0.0173 with image-bootstrap CI
`[-0.0239, -0.0103]` and wins only 2/14 stable position bins. Sink deletion
leaves mass essentially unchanged (0.8808) and raises the weaker
mass+connectivity representation from 0.8601 to 0.8683, but this +0.0082 change
misses the +0.010 sink gate and the result remains -0.0129 below the original
mass probe. Literal position-balanced AUROC likewise falls from 0.8373 for mass
to 0.8115/0.8124 for no-deletion/sink-clean spatial fusion. Close this fixed
top-10% 8-neighbor and low-key-norm sink formulation; do not add either
component to the detector method.

### 4.6 Registered pre/post temporal-row audit

The full allocation representation mixes an anticipatory `pre` row, before the
object's first subtoken enters the sequence, with a `post` row that has already
consumed that subtoken. The earlier zero-at-mean intervention shows that the
trained full readout relies more heavily on post, but it does not measure the
best row-only detectors. The locked
[temporal-row audit](iteration_protocols/no_rope_temporal_row_audit.md) therefore
refits equal-dimensional 1,024-input pre-only and post-only No-RoPE mass probes
under the same five folds, seed, conditional objective, and MLP. The existing
2,048-input full probe is retained only to test whether the rows are
complementary. This is explanatory and cannot convert a predictive temporal
fingerprint into verification.

**Outcome.** Separately refitted post-only mass is the strongest representation:
0.8859 within-bin from 1,024 inputs, versus 0.8425 for equal-dimensional
pre-only and 0.8812 for 2,048-dimensional full pre+post. Post minus pre is
+0.0434 with bootstrap CI `[0.0353, 0.0529]` and 13/14 stable-bin wins, passing
the post-dominance gate. Full minus post is -0.0047 with CI
`[-0.0083, -0.0009]` and only 2/14 wins, so temporal complementarity fails:
adding pre slightly hurts rather than helps. Literal position-balanced AUROC is
0.7913 pre, 0.8426 post, and 0.8373 full. The strongest allocation fingerprint
is therefore primarily a post-target reorganization across layers and heads,
not an anticipatory object-verification signal. Pre remains predictive, but it
is neither dominant nor complementary in this readout.

**Direct VAR-MLP comparison.** On one user-requested fixed COCO image split
(4,000 train / 1,000 test), the same seed-17 width-256 readout gives the
published VAR feature (native post-RoPE L5--18, 448 inputs) 0.8869 overall and
0.8612 width-10 within-bin AUROC, versus 0.9119 and 0.8748 for our post-row
No-RoPE L0--31 representation (1,024 inputs). Accuracy is 0.8039 versus 0.8406.
Thus ours has a modest descriptive within-bin advantage of +0.0137 under a
shared split/readout. This minimal run has no PCA, resampling, bootstrap, or
multi-seed analysis and does not separate the effects of No-RoPE and layer
breadth; it supports a small empirical advantage, not a distinct verification
architecture claim.

### 4.7 Registered per-position detector comparison

The temporal result must be read against prior detectors on the same literal
position-balanced rows, not only against pooled or natural-distribution AUROC.
The locked
[comparison protocol](iteration_protocols/position_balanced_detector_comparison.md)
adds five representative label-free reproductions (PAS, SVAR, IC, TVER, SAE)
and two supervised external probes (ADS+CGC and local HaloProbe-style) to the
pre/post/full mass table. All methods share the same 1,000 draws inside each
10-token bin. The comparison reports AUROC only across methods because the
label-free scores do not share a calibrated zero threshold.

**Outcome.** Literal position balancing reverses the apparent ranking of the
prior attention scores. PAS and TVER fall from natural-distribution overall
AUROCs of 0.8349/0.8303 to 0.5228/0.5034, while IC is the strongest label-free
baseline at 0.6996. The supervised local HaloProbe-style reproduction reaches
0.7439. No-RoPE mass gives 0.7913 pre, 0.8426 post, and 0.8373 full; post is the
highest column in 12/15 position bins, with full leading in the other three
(twice by less than 0.006). Thus the post-target allocation fingerprint remains
strongest under an identical sampling distribution, but this is a supervised
diagnostic result, not a label-free detector. ADS+CGC is omitted from the
per-bin table because
its original OOF scores were not retained and the current reconstruction misses
the locked aggregate-reproduction tolerance; inserting reconstructed bins would
violate the protocol.

## 5. Contribution framing (intended)

The current defensible frame is an **analysis paper**; a constructive method can
be restored only by a new, independently motivated hypothesis that survives an
equal-capacity allocation control:

1. The **two-pronged critique** — attention-based detection and mitigation both
   fail under matched controls (length control for detection; answer-prior /
   related-vs-plain control for mitigation).
2. A controlled **semantic-neighbor / alternative-confusion** view that makes the
   mitigation failure measurable.
3. A **capacity-controlled detector audit**: a source-decomposed target-value
   probe appears strong against narrow baselines, but an equal-dimension
   all-layer allocation probe matches it. This prevents a false value-evidence
   contribution claim and sharpens what a future method must explain.

Honest guardrails carried over from the tear-down:
- Prong 1 is pre-empted by HaloProbe → background only.
- Prong 2 is scoped to attention *amplification* (PAI is inert).
- Any absolute-gain claim must be reported against the honest baselines here.
- VVE has failed its registered capacity confirmation gate. Keep it as an audit,
  remove the constructive claim, and do not revive XEV or the closed clean-
  attention detector.

## 6. Known risks / must-dos (rebuild backlog)
- **XEV is closed:** preserve code and append-only evidence for reproducibility,
  but do not tune aliases, candidates, boxes, scales, thresholds, or support
  guards. The 4/960 result is a sparse numerical fluctuation, not the method.
- **TCS is cancelled:** do not build the LaMa/inpainting pipeline or train the
  four-view objective. Its protocol is historical, not pending work.
- **NEA is cancelled:** do not implement null keys/values or run its Stage A.
  The retained protocol is historical, not pending work.
- **EEA is cancelled:** do not train the patch matcher or run its Stage A. The
  retained protocol is historical, not pending work.
- **PCPA is cancelled:** do not build manifests or train LoRA. The retained
  protocol is historical, not pending work.
- **No-RoPE is not deconfounding by itself:** it removes rotary geometry but not
  response-depth information in hidden states or labels. The learning and
  evaluation controls in §3.1 are mandatory.
- **Supervision relabeling:** an external probe is allowed but must be called
  supervised and compared to HaloProbe/DHCP/Beyond rather than to label-free
  methods as though supervision were free.
- **Feature-selection leakage:** do not select heads, layers, signs, temporal
  rows, or fusion weights on evaluation labels. VVE iteration 1 extracts all
  layers and freezes its label-free aggregation before the full run; all
  calibration is train-fold-only.
- **VVE supervision boundary:** the successful MLP uses CHAIR labels. Call it a
  supervised external detector, even though the LVLM is frozen and inference
  does not fine-tune generation. The registered label-free VVE score failed.
- **VVE route is closed:** the equal-dimension allocation-mass probe matches
  VVE-core, so do not attribute iteration-1 performance to values, select a
  favorable detector seed, or continue its causal/transfer stages as method
  confirmation.
- **Spatial-allocation route is closed:** fixed top-10% 8-neighbor coherence is
  significantly worse than equal-dimensional top-10% concentration, and sink
  deletion does not beat no-deletion mass. Do not sweep support fractions,
  neighborhoods, sink thresholds, or morphology on COCO to rescue it.
- **Position-uniform readout failed:** it does not improve macro or late-bin
  VVE metrics. Retain the original objective for historical reporting and do
  not sweep blended weights, position heads, or thresholds on COCO.
- **Uncertainty shortcut:** compare clean-head scores against IC, NLL, and
  entropy, and test whether a supervised fusion adds signal beyond them.
- **Transfer:** COCO/LLaVA feasibility cannot support a general detector claim.
  Reserve Qwen3-VL and independent datasets for a new route that first survives
  local equal-capacity controls; do not spend them validating closed VVE.
- **Compute:** cache compact per-head evidence only, never full hidden states or
  full attention tensors. Run the 16-image numerical gate before full sharding;
  do not run mitigation inference in this iteration.
- **Prong 2 generality:** current evidence is ClearSight/VisAttnSink/PAI/VCD on
  LLaVA-1.5. Extend the yes-rate / related-FPR decomposition to more mitigation
  baselines (OPERA/DAMRO/M3ID) and ≥1 modern model (Qwen-VL / InternVL).
- **Prong 1 boundary:** keep it as background; its only forward use is to
  motivate evidence criteria that survive confound controls. Do not claim all
  internal signals fail.
- **Object repetition control** (free, borrowed from HaloProbe) for detection.

## 7. Pointers
- Score/metric definitions and the position/length control: [design.md](design.md).
- Repo layout: [project_structure.md](project_structure.md).
- Reproduced baseline results: [experiment_results.md](experiment_results.md).
- Decision history: [iteration_log.md](iteration_log.md).
- Completed VVE falsification protocol:
  [visual_value_evidence_iteration_2.md](iteration_protocols/visual_value_evidence_iteration_2.md).
- Completed VVE iteration-1 protocol:
  [visual_value_evidence_iteration_1.md](iteration_protocols/visual_value_evidence_iteration_1.md).
- Closed position-free attention protocol:
  [position_free_detector_iteration_1.md](iteration_protocols/position_free_detector_iteration_1.md).
- Rejected PCPA design record:
  [pcpa_iteration_1.md](iteration_protocols/pcpa_iteration_1.md).
- Rejected EEA design record:
  [eea_iteration_0.md](iteration_protocols/eea_iteration_0.md).
- Rejected NEA design record:
  [nea_iteration_1.md](iteration_protocols/nea_iteration_1.md).
- Rejected TCS design record:
  [tcs_iteration_1.md](iteration_protocols/tcs_iteration_1.md).
