# CLAUDE.md

Guidance for any agent working in this repository.

## Three living documents — always read first, always keep updated

This project is steered by three documents under `docs/`. Before starting any
non-trivial task, read all three. After any change that affects them, update
them in the same session.

| Doc | Role | Update cadence |
|---|---|---|
| [docs/proposal.md](docs/proposal.md) | The current research proposal: thesis, positioning vs. prior work, the headline method design, and contribution framing. The single source of truth for *what we are trying to publish and why*. | Whenever the research direction, thesis, or method design changes. |
| [docs/experiment_results.md](docs/experiment_results.md) | Consolidated experiment results. Every new experimental result is appended here with date, setup, numbers, and interpretation. | Every time a new result is produced. Append, do not overwrite history. |
| [docs/iteration_log.md](docs/iteration_log.md) | Project iteration log. Each completed iteration (decision, pivot, implementation milestone) is recorded with date and rationale. | At the end of each iteration. |

Rules:

- These three docs override older planning docs when they conflict. Background
  references: `docs/design.md` (score rationale), `docs/aaai2027_paper_plan.md`
  and `docs/tdev_detector_positioning.md` (still-useful planning/lit-review).
  Superseded ICML-era plans and working audits live in `docs/archive/` — history
  only, not the current plan.
- `docs/experiment_results.md` and `docs/iteration_log.md` are append-only logs.
  Never delete past entries; add a new dated entry instead.
- If a task changes the proposal, record the change as an iteration-log entry too.

## Project in one paragraph

SinkDetect studies object hallucination in LVLMs (LLaVA-1.5-7B baseline). The
thesis: visual attention allocation is not the same as target-object evidence
verification. The headline contribution is a **semantic-neighbor stress test**
plus a **target-discriminative evidence verification (TDEV)** criterion. See
[docs/proposal.md](docs/proposal.md) for the full argument.

## Repository conventions

- Detection code under `detection/`, mitigation code under `mitigation/`.
  See [docs/project_structure.md](docs/project_structure.md).
- Generated artifacts (`experiments/`, `**/results/`) are git-ignored; do not
  commit them.
- The paper is managed in a separate repo under `paper_repo/` (ignored here).
- Score definitions and the *why* behind each score live in
  [docs/design.md](docs/design.md) and [detection/docs/scores.md](detection/docs/scores.md).
