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

**Status after iteration 12 (2026-07-19):** the first usable constructive version
is confirmed, with deliberately narrow scope. On 960 untouched POPE-adversarial
rows, the precision-repaired certificate triggers 4 times: 3 false-positive
corrections and 1 true-positive veto (precision 0.75). FPR falls 0.0771 -> 0.0708,
TPR falls only 0.8417 -> 0.8396, balanced accuracy rises 0.8823 -> 0.8844, and
MCC rises 0.7671 -> 0.7719. It passes every pre-registered gate. The effect is
sparse and small; current evidence supports a high-precision correction
certificate, not a broad decoding replacement. That held-out result used raw
post-RoPE attention for localization. The current implementation defaults to a
pre-RoPE localization map, but its behavior is not yet confirmed and the v1
held-out claim must remain attributed to the raw map. A development attempt to
broaden the local relation gate from C to any non-supporting identity raised
local proposals from 1 to 21/120, but its complement-or-inventory support guard
left 3 actions and all 3 vetoed true positives. This variant is rejected: the
remaining problem is image-level coverage, not local-conflict recall alone.
Residual top-3 attention coverage improves support recall but still leaves 10
true-positive vetoes for 4 corrections on the 120-row development window. The
variant is rejected: coarse attention boxes are not object-pure enough to make
candidate-relative local comparisons globally consistent.
A target-grounding/re-identification audit then returns valid boxes for all 14
selected failures with high support precision but only 2/10 true-target recall.
This variant is also rejected; its coordinate proposals remain a useful lead,
but keeping a small grounded box on the full canvas does not provide visual zoom.

### 3.1 Principle and scope

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
- **XEV coverage:** preserve XEV v1 as the raw-attention held-out reference and
  report its sparse action rate. Treat no-RoPE as the current unconfirmed
  localization update. Iteration 10 already establishes a higher-coverage local
  conflict proposal but rejects free-form full-image inventories and one-region
  complements as its support check. Iteration 11 also rejects residual top-3
  attention rectangles: they improve coverage but remain multi-object and yield
  inconsistent local identities. Next test a target-focused or class-agnostic
  object proposal that remains proposal-only; do not use attention mass,
  proposal confidence, or generated inventory presence as decision evidence.
  Native target grounding passes the coordinate sanity audit but its full-canvas
  re-identification cycle has only 2/10 support recall. Next test exact box crops
  as zoomed views and keep grounding output itself proposal-only.
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
