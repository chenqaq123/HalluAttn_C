# XEV Iteration 12 Protocol: Ground-and-Reidentify Cycle

Date: 2026-07-19
Status: completed; stage-1 gate failed, stage 2 not run

## Motivation

Iteration 11 shows that residual attention rectangles are spatially diverse but
not object-pure. They can contain multiple valid identities, so pairwise answers
against different candidates contradict one another. The next proposal must be
target-focused before local verification, while its own output remains barred
from serving as evidence.

Qwen3-VL natively supports multi-target 2D grounding with normalized 0--1000
coordinates. Its official cookbook states that requested negative categories can
be skipped. This behavior is useful for proposing where a target might be, but a
returned or missing box is not an absence/presence certificate: the same model
may ground a hallucinated target onto a related object.

## Hypothesis

A target-conditioned grounding box will be tighter than an attention rectangle.
If the target is real, a second target-free recognition pass on that box should
recover the target identity. If the grounding request lands on a related object
for an absent target, target-free recognition should instead name that visible
alternative. This proposal-to-recognition cycle can provide positive target
support without treating grounding coordinates, labels, or confidence as
evidence.

## Frozen Cycle Rule

For a queried target `t`, use the official-style prompt:

    Locate every instance that belongs to the following category: "t".
    Report bbox coordinates in JSON format.

Parse at most three valid `bbox_2d` boxes in normalized 0--1000 coordinates.
Ignore every generated label and any non-coordinate field. A missing/invalid box
means only that no extra proposal is available; it is not negative evidence.

For every box, keep the exact box clear on the matched blurred canvas and:

1. without including `t` in the prompt, list at most six concrete object
   identities supported by the clear region;
2. compute the ordinary Yes/No margin `m(t,R)`;
3. compare `t` against up to three positive-keep open/closed C or N alternatives
   with the existing reversed pairwise questions.

The box has **cycle-consistent target support** only when all three conditions
hold:

    target alias appears in the target-free regional inventory
    m(t,R) > 0
    no candidate gives target-first < 0 and alternative-first > 0

All thresholds are natural zero. The grounding response itself, box count,
coordinates, generated label, attention mass, inventory size, and proposal
confidence never enter the final decision.

For the full development replay, retain iteration 11's frozen K=3 local-conflict
proposal. Veto baseline Yes only when at least one conflict exists and no
grounding box has cycle-consistent target support.

## Data and Stages

- **Stage 1 mechanism audit:** the 14 iteration-11 final actions, selected before
  this protocol: 4 baseline false positives and 10 true positives. This set is
  deliberately failure-enriched and cannot support an aggregate benchmark claim.
- **Stage 2 development:** only if stage 1 passes, lazily ground every one of the
  49 iteration-11 local-conflict rows on POPE-adversarial 2520--2639 and replay
  behavior on all 120 rows.
- Reuse saved iteration-11 conflict decisions; do not rerun or change attention
  localization.
- Rows 2640 onward remain untouched for a later confirmatory protocol.

## Gates

Stage 1 advances only if cycle support:

- suppresses at least 8 of the 10 prior true-positive vetoes;
- falsely suppresses at most 1 of the 4 prior false-positive corrections;
- has at least 0.80 support precision on these 14 selected cases.

Stage 2 advances to a separately committed holdout protocol only if the final
rule has at least 2 actions, correction precision at least 2/3, no more than one
true-positive veto, lower FPR, TPR loss at most 0.005, and non-decreasing balanced
accuracy and MCC versus baseline.

If either gate fails, audit box parse rate, target-free target recovery, target
margin, and conflict rejection separately. Do not tune box expansion, inventory
length, aliases, K, or non-zero margins on these rows.

## Result

All 14 selected cases return at least one valid box, 11 have a positive target
margin, 5 recover the target in target-free regional recognition, and 11 contain
a directional conflict. Only 2 boxes satisfy the full cycle; both are true
targets (`wine glass` and `umbrella`), so support precision is 1.0 and no prior
false-positive correction is suppressed. However, only 2/10 true-positive vetoes
are prevented instead of the required 8/10. Stage 1 fails and stage 2 is not run.

The failure is not simply poor coordinate generation. Among eight true targets
with a matching COCO detection annotation, maximum proposal-to-ground-truth IoU
has mean 0.607 and median 0.576. The current clear-box view preserves the full
canvas, so a precisely grounded small target remains small after preprocessing.
In addition, target-free recognition names real `bench` and `backpack` targets
but the cycle rejects them because a spatially overlapping `person` produces an
N-relation directional conflict. No rule was relaxed on this audit.
