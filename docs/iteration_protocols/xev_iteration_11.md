# XEV Iteration 11 Protocol: Multi-Region Grounded Coverage

Date: 2026-07-19
Status: pre-registered exploratory development protocol

## Motivation

Iteration 10 recovers local-conflict coverage but shows that neither a generated
full-image inventory nor a one-region complement is reliable target support.
Both ask the model for another image-level judgment, so neither establishes
where the target evidence is. This violates the constructive side of the paper's
central distinction: attention may propose where to inspect, but evidence must
come from explicit visual identity verification.

The iteration-10 false actions are real small/accessory targets whose first
attention proposal contains another salient object. A single locally coherent
conflict therefore cannot establish image-level absence. The direct repair is
to cover multiple distinct regions and ask the same counterbalanced identity
question in every region.

## Hypothesis

For baseline-Yes cases, diverse attention proposals can provide image coverage
without treating attention magnitude as evidence. A real target missed by one
proposal will receive a strict target-over-alternative identity switch in another
proposal, whereas an absent target will produce local conflicts but no such
target-supporting region.

## Frozen Development Rule

Generate at most three no-RoPE proposals. Proposal 1 is the current top-30%-mass
connected region. Before extracting each later proposal, zero the previous
proposal's attention-grid box plus the existing one-cell dilation halo, then
repeat the same connected-region extraction on the residual map. Stop only when
no positive residual mass remains. Attention values rank proposals but never
enter the verification score or final decision.

For every proposal, obtain target-free open candidates and closed COCO-NPMI
candidates. Keep up to three candidates with positive region keep margin and
relation C or N. An exact target phrase in the open candidate list is not itself
evidence and does not stop pairwise verification.

For target `t`, alternative `a`, and region `R`, use the same two reversed
questions as XEV v1:

    conflict(t,a,R): m(t>a,R) < 0 and m(a>t,R) > 0
    support(t,a,R):  m(t>a,R) > 0 and m(a>t,R) < 0

Both use natural zero thresholds. A region is ambiguous when neither strict
direction holds; ambiguity is not support. For a baseline-Yes answer, veto Yes
only when:

1. at least one region has a conflict against any open/closed C or N candidate;
2. no region has target support against any open/closed C or N candidate.

No full-image inventory, generated S relation, complement margin, oracle
annotation, attention threshold, or fitted numeric threshold is used by the
decision.

## Data and Comparisons

- Development: POPE-COCO adversarial rows 2520--2639, already development-only.
- Localization: no-RoPE, layers 18--29, top-30% residual connected proposals.
- Primary comparison: K=3 diverse regional verification.
- Paired ablation from the same saved rows: use region 0 only (K=1) with the
  identical conflict/support aggregation rule.
- Diagnostics: proposal count and pairwise-call coverage; TP-conflict support
  rate; FP-conflict false-support rate; final corrections and TP vetoes.
- Rows 2640 onward remain untouched for a later confirmatory test.

## Development Gate

Advance to a separately committed confirmatory protocol only if K=3 has:

- at least 3 rows with a local conflict;
- at least 2 final actions;
- correction precision at least 2/3;
- no more than one true-positive veto;
- lower FPR, TPR loss at most 0.005, and non-decreasing balanced accuracy and
  MCC versus the baseline;
- correction precision no worse than the paired K=1 regional-support ablation.

If the gate fails, separate proposal-coverage failure from verifier failure. Do
not tune non-zero pairwise thresholds, proposal mass, proposal count, or region
size on these rows.
