# AGENTS.md

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

- These three docs are authoritative. Background reference: `docs/design.md`
  (score / position-length-control metric rationale) and
  `docs/project_structure.md` (repo layout).
- **The project was restarted on 2026-07-07 (branch `restart-two-pronged`).**
  Only the reproducible detection/mitigation baselines were kept; all prior
  method design (TDEV / semantic-neighbor verifier / no-external-detector family
  / CHAIR intervention) was torn down and is being rebuilt. That torn-down
  history — including the old planning/audit docs — lives on the `main` branch.
- `docs/experiment_results.md` and `docs/iteration_log.md` are append-only logs.
  Never delete past entries; add a new dated entry instead.
- If a task changes the proposal, record the change as an iteration-log entry too.

## Project in one paragraph

SinkDetect studies object hallucination in LVLMs (LLaVA-1.5-7B baseline). The
thesis: visual attention allocation is not the same as target-object evidence
verification, and this proxy fails in **both** operational directions —
**reading** attention (detection collapses to chance under length control) and
**pushing** attention (amplification only adds yes-bias / backfires). This
two-pronged critique is the durable motivation; the constructive method is being
rebuilt (see [docs/proposal.md](docs/proposal.md)).

## Repository conventions

- Detection code under `detection/`, mitigation code under `mitigation/`.
  See [docs/project_structure.md](docs/project_structure.md).
- Generated artifacts (`experiments/`, `**/results/`) are git-ignored; do not
  commit them.
- The paper is managed in a separate repo under `paper_repo/` (ignored here).
- Score definitions and the *why* behind each score live in
  [docs/design.md](docs/design.md) and [detection/docs/scores.md](detection/docs/scores.md).
