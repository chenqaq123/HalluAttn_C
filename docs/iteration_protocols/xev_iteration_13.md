# XEV Iteration 13 Protocol: Grounded Crop-and-Zoom Verification

Date: 2026-07-19
Status: completed; stage-1 gate failed, variant retained only as a view ablation

## Motivation

Iteration 12 separates proposal quality from verification quality. Native
target-conditioned grounding returns a valid box for every selected failure and
has mean maximum IoU 0.607 on the eight real targets with matching COCO boxes,
but its full-canvas clear-box verifier recovers only 2/10 real targets. That view
preserves the original object scale after preprocessing, so even an accurate box
does not make a small target easier to inspect. It also treats an overlapping N
object such as `person` as an exclusive identity contradiction.

The paper's distinction remains unchanged: grounding may say where to inspect,
but neither its returned label, box presence, coordinates, nor confidence may
serve as target evidence. Evidence must come from a separate visual
re-identification pass.

## Hypothesis

Reusing the exact same grounding coordinates as iteration 12 but presenting each
box as an independent crop will enlarge the proposed object during model
preprocessing. Target-free recognition and a positive target margin should then
recover most real small targets. Within a target-focused crop, only a C relation
is a mutually exclusive identity alternative; an N relation may be a co-visible
object and cannot invalidate target support.

## Frozen Paired Rule

Use the 14 saved iteration-12 action rows and reuse their parsed pixel boxes
exactly. Do not rerun grounding, select boxes by downstream scores, expand boxes,
add context padding, or change box order. For each box:

1. crop the exact pixel rectangle and let the unchanged Qwen3-VL processor resize
   it normally;
2. without naming the queried target, list at most six concrete object identities
   visible in the crop;
3. compute the ordinary Yes/No target margin on that crop;
4. compare the target against up to three positive-margin C alternatives from
   the same frozen open/closed candidate sources, using both reversed pairwise
   questions.

A crop has **cycle-consistent target support** only when:

    a target alias appears in the target-free crop inventory
    crop target margin > 0
    no C alternative has target-first < 0 and alternative-first > 0

All thresholds remain natural zero. N relations are recorded diagnostically but
are neither compared nor used to accept or reject support. A missing saved box
means only that no proposal is available. Grounding output, target-conditioned
labels, box count, coordinates, proposal confidence, attention values, inventory
size, and candidate keep margins never directly enter the decision.

For the full development replay, retain iteration 11's frozen K=3 local-conflict
proposal. Veto baseline Yes only when at least one saved local conflict exists and
no grounded crop has cycle-consistent target support.

## Data and Stages

- **Stage 1 paired mechanism audit:** the same 14 iteration-11 final actions used
  in iteration 12: 10 true-positive vetoes and 4 false-positive corrections.
- Reuse `iter12_grounded_cycle_actions.jsonl` as the sole source of boxes and
  iteration-11 JSONL files as the sole source of local-conflict decisions.
- **Stage 2 development:** only if stage 1 passes, first generate and freeze boxes
  for the remaining iteration-11 local-conflict rows under a separate committed
  protocol, then apply this crop rule. Stage 1 cannot justify an aggregate claim.
- Rows 2640 onward remain untouched for a later confirmatory protocol.

## Gate

Stage 1 passes only if crop cycle support:

- suppresses at least 8 of the 10 prior true-positive vetoes;
- falsely suppresses at most 1 of the 4 prior false-positive corrections;
- has support precision at least 0.80 on these 14 selected cases.

Report paired per-condition recovery against iteration 12: target-free target
name, positive target margin, C-conflict rejection, and full cycle support. If
the gate fails, do not tune crop padding, box selection, inventory length,
aliases, candidate K, relation classes, or non-zero margins on these cases.

## Result

All 14 saved box sets are reused exactly. Relative to the full-canvas verifier,
target-free target recovery changes 5 -> 4 rows, positive target margin changes
11 -> 9, directional conflict changes 11 -> 3 under the pre-registered C-only
rule, and full cycle support changes 2 -> 4. The supported real targets are
`wine glass`, `bench`, `cell phone`, and `backpack`; no absent target receives
support, so precision remains 1.0. However, 4/10 true-target recovery is below
the required 8/10 and the gate fails.

The paired failures show that exact crop-and-zoom is complementary rather than
uniformly better. It newly recovers `bench`, `cell phone`, and `backpack`, but
loses the full-canvas `umbrella` support because target-free recognition changes
from `umbrella` to `cloud`. That box is only 19x7 source pixels; `surfboard` is
43x12. In contrast, a 15x12 `cell phone` box succeeds, so source resolution
alone does not explain recovery. The amount of surrounding visual context is a
second variable. No padding or scale was tuned after observing this result.
