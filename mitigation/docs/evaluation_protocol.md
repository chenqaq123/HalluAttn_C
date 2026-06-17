# Mitigation Evaluation Protocol

## Goal

Evaluate whether attention interventions genuinely alleviate object
hallucination or mainly shift output behavior.

## Methods

- `vanilla`: no intervention.
- `pai`: attention-only PAI, without its contrastive/logit-refinement branch.
- `clearsight`: Visual Amplification Fusion (VAF).
- `visattnsink`: Visual Attention Sink redistribution.

The common runner ports each intervention to the current HuggingFace
LLaVA-1.5-7B model stack and saves one output row per input sample. These are
controlled ports for mechanism testing, not bit-level official reproductions.
PAI is strictly attention-only here: the official CFG/logit-refinement branch
is not included.

## POPE

Run each intervention on the identical `random`, `popular`, and `adversarial`
COCO POPE question sets. Retain each sample's prediction and label.

Headline diagnostics:

| Metric | Purpose |
|---|---|
| `accuracy`, `f1`, `mcc`, `balanced_accuracy` | standard aggregate quality |
| `yes_rate` | detects answer-prior movement |
| `invalid` | counts non yes/no outputs; default evaluation aborts if nonzero |
| `recall_tpr` | gain on present-object questions |
| `fpr` | new false positives on absent-object questions |
| `delta_tpr_minus_delta_fpr` | distinguishes selective gain from yes bias |

POPE parsing is strict: only an explicit first answer token of `yes` or `no` is
accepted. By default, evaluation aborts on invalid outputs; use
`--invalid_policy as_wrong` or `--invalid_policy drop` only for explicitly
marked sensitivity analyses. An intervention that raises `TPR` and `FPR` by
similar amounts is not evidence of improved grounding, even if aggregate
accuracy changes on one split.

## CHAIR

Use the fixed COCO image manifest already used for detection experiments:
`experiments/coco_llava_7b/generation.json`. Each method regenerates captions
for the same images.

Report:

| Metric | Purpose |
|---|---|
| `CHAIRi`, `CHAIRs` | object hallucination rates |
| mean caption words | checks for shortened outputs |
| mean object mentions | checks for conservative under-description |
| mean hallucinated mentions | absolute hallucination output tendency |

A lower CHAIR score together with fewer object mentions or much shorter
captions must be interpreted as a trade-off, not automatically as stronger
visual grounding.

## Outputs

For each task/split/method, the runner writes `predictions.jsonl` and
`metrics.json`. A task-level `comparison.json` contains deltas from vanilla.
Results are placed under `mitigation/results/<experiment>/`.
