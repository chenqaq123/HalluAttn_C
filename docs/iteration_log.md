# Iteration Log

> Append-only. One entry per completed iteration (a decision, pivot, or
> implementation milestone). Newest at the bottom. Each entry records the date,
> what changed, why, and what it implies for the proposal / next steps.

<!-- Template:
## YYYY-MM-DD — <title>
**What changed:** ...
**Why:** ...
**Evidence/refs:** ...
**Implications / next:** ...
-->

## 2026-06-23 — Repositioning after discovering HaloProbe

**What changed:**
- Established three living docs (`proposal.md`, `experiment_results.md`,
  `iteration_log.md`) + root `CLAUDE.md` as the steering documents.
- Demoted the position-confound chapter and the per-head diagnostic from
  headline to background.
- Promoted the **semantic-neighbor stress test + target-discriminative (TDEV)
  criterion** to the single headline.
- Decided to **remove the external OWLv2 detector** from the headline method and
  replace it with an internal, self-contrastive target-vs-neighbor readout.

**Why:**
- Literature check found **HaloProbe** (arXiv:2604.06165, Apr 2026; A./M.
  Rohrbach among authors — original CHAIR authors) already establishes
  position as a confounder for attention-based detection (framed via Simpson's
  paradox), uses a per-head attention+confidence probe (93.5 AUROC), adds object
  repetition as a second confounder, and evaluates on 5 models. This pre-empts
  two of our three pillars.
- PARALLAX (arXiv:2605.17028) similarly works on benchmark artifacts in
  hallucination detection.
- Our OWLv2-backed TDEV invites the "why not just use OWLv2 / this is
  Woodpecker-lite" critique and carries the main latency cost.

**Evidence/refs:**
- HaloProbe overlap analysis and full read recorded in this session.
- Semantic-neighbor results show all attention/VCD methods leave an unclosed
  related-minus-plain FPR gap (see `experiment_results.md` §D).

**Implications / next:**
1. Implement internal self-contrastive TDEV. Primary = **B (attention-region
   discriminability)**; cheap baseline = **A (IC/logit-lens target-vs-neighbor
   margin)**; supervised ceiling = **C (contrastive per-head probe)**. See
   `proposal.md` §3.
2. First validation experiment: replace OWLv2 with IC target-vs-neighbor margin
   on the related-present negative subset; check whether the FPR gap closes.
3. Run a HaloProbe-style internal probe on the related-present subset to show it
   leaves the gap open (turn the threat into a foil).
4. Add object-repetition control to detection eval (free, borrowed from HaloProbe).
5. Add ≥1 modern model (Qwen-VL / InternVL) for the semantic-neighbor headline.
6. Borrow HaloProbe's *explanation framing* (Simpson's paradox; internal vs.
   external factorization) for the intro — not its experiments.
