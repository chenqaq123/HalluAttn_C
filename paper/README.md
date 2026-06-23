# Looking Is Not Verifying Paper

This directory contains the current paper draft for:

> Looking Is Not Verifying: Target-Discriminative Evidence for LVLM Object Hallucination

The paper should be treated as a diagnostic-plus-verification submission, not as
a state-of-the-art standalone mitigation paper. The core motivation remains
`looking is not grounding`: visual routing can be meaningful while still failing
target-object verification.

## Main Files

| Path | Purpose |
|---|---|
| `acl_latex.tex` | Main LaTeX entry point for the current draft. |
| `sections/` | Paper sections included by the main file. |
| `tables/` | Audited paper tables. |
| `custom.bib` | Non-Anthology bibliography entries. |
| `scripts/check_paper_all.py` | Single verification entry point before committing paper changes. |

## Verification

Run this before committing paper edits:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python paper/scripts/check_paper_all.py
```

The unified checker runs:

- `paper/scripts/check_paper_static.py`: citation keys, refs, figure paths, TODO markers, and forbidden over-strong claims.
- `paper/scripts/check_audited_numbers.py`: table numbers against saved result artifacts.

LaTeX compilation is not currently part of the local gate because this runtime has
no `latexmk` or `pdflatex`. If a TeX environment is available, compile
`paper/acl_latex.tex` after the scripted checks.

## Current Paper Claim Gate

Supported framing:

- attention mass and simple attention-shape scores are unreliable grounding proxies under position and same-object controls;
- attention/decoding interventions can change routing, yes rate, or caption object richness without verifying target-object presence;
- semantic-neighbor negatives expose the key failure mode: associated evidence can be present while the queried target is absent;
- TDEV is a target-vs-neighbor verification criterion, currently instantiated with OWLv2 region evidence;
- the strongest POPE result is a modest hybrid gate-plus-rescue verifier: macro MCC `0.730 -> 0.763`, related FPR `0.114 -> 0.069`;
- LH-Shape/TDEV-lite evidence is supervised triage/readout evidence, not a standalone mitigation method.
- verified atomic caption detail is a scoped 100-image prototype, not a complete caption benchmark method.

Avoid claiming:

- that the method solves hallucination;
- state-of-the-art mitigation;
- that OWLv2/external detection is the contribution;
- that attention is useless;
- that LH-Shape is a standalone mitigator;
- that aggregate POPE MCC alone proves grounding.

## Main Evidence Artifacts

| Evidence | Artifact |
|---|---|
| Claim/evidence matrix | `../docs/icml_evidence_matrix.md` |
| Paper blueprint | `../docs/icml_paper_blueprint.md` |
| Paper coverage audit | `../docs/icml_paper_coverage_audit.md` |
| Claim gate | `../docs/claims_alignment_audit.md` |
| Baseline availability | `../docs/baseline_availability_refresh.md` |
| TDEV ablations | `../docs/tdev_ablation_summary.md` |
| Mechanism figure | `../mitigation/results/pope_mechanism_alignment_full/figure/pope_mechanism_alignment_contact_sheet.png` |

## Editing Notes

Keep the draft aligned with the current result boundary. If a new experiment
improves or contradicts the current TDEV story, update the relevant docs first,
then update the paper and rerun `check_paper_all.py`.
