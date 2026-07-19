# XEV Iteration 10 Protocol: Local Identity Conflict, Global Target Support

Date: 2026-07-19
Status: completed; development gate failed, variant rejected

## Motivation

XEV v1 acts on only 4/960 rows because its local proposal requires a closed-source
candidate labeled C. This applies an image-level semantic relation to a region-level
identity question. Two objects may be unrelated or coexist in the image (N) while
still being mutually exclusive identities for the same clear region.

The paper's central distinction must remain intact:

1. attention proposes a region but is not evidence;
2. a counterbalanced visual question identifies what the region contains;
3. an independent check asks whether target evidence exists outside that region.

## Hypothesis

A strict directional switch against any non-supporting local identity is a
higher-recall local conflict proposal than the closed+C gate. False proposals
caused by a real target elsewhere can be suppressed by positive target support
from either the region complement or two target-free full-image inventories.

## Frozen Development Rule

For a baseline Yes answer and no exact target mention in the target-free local
inventory, propose a local conflict when at least one positive-keep candidate:

- comes from open or closed acquisition;
- is not labeled S;
- gives target-first margin `< 0` and alternative-first margin `> 0`.

For the strongest such conflict, define global target support as:

    complement support: m(target, I_drop) >= 0
    inventory support: both target-free inventories match a target alias
    global support: complement support OR inventory support

Veto the baseline Yes only when a local conflict exists and global support is
false. All numeric thresholds are the natural zero; no fitted cutoff is allowed.

## Data and Comparisons

- Development: POPE-COCO adversarial rows 2520--2639, already development-only
  after iteration 9.
- Localization: no-RoPE, layers 18--29, one top-30%-mass proposal.
- Compare: XEV-v1 closed+C local gate; non-S/any-source plus inventory-only
  support; non-S/any-source plus complement-or-inventory support.
- Oracle annotations are not used by the method.

## Development Gate

Advance to a separately committed confirmatory protocol only if the 120-row
development run has:

- at least 3 local conflict proposals;
- at least 2 final actions;
- correction precision at least 2/3;
- no more than one true-positive veto;
- non-decreasing balanced accuracy and MCC versus the baseline.

If this fails, audit whether complement support or inventory support caused the
failure. Do not tune non-zero thresholds on these rows.

## Result

The broadened relation gate succeeded only at the proposal-recall objective:
there were 21 local conflicts instead of one under the XEV-v1 gate. The
complement-or-inventory union suppressed 18 and left 3 actions, all of which
vetoed true positives; it corrected none of the 5 baseline false positives.
Balanced accuracy fell 0.9083 -> 0.8833 and MCC fell 0.8168 -> 0.7684. The
development gate therefore fails.

The pre-registered source audit shows different but insufficient failure modes:
inventory-only gives 1 correction / 9 true-positive vetoes among 10 actions;
complement-only gives 2 corrections / 4 true-positive vetoes among 6 actions;
their union gives 0 corrections / 3 true-positive vetoes. No non-zero threshold
was fit. The local broadening is retained as a diagnostic capability, while the
support union is rejected.
