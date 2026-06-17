# Mitigation Baselines

This directory contains attention-intervention baselines for controlled
hallucination mitigation analysis on LLaVA-1.5-7B.

## Implemented Ports

| Method key | Paper | Port scope | Default component settings |
|---|---|---|---|
| `vanilla` | none | greedy generation without intervention | comparison anchor |
| `pai` | Paying More Attention to Image | attention manipulation only; CFG/logit refinement deliberately excluded | layers `[2, 32)`, `alpha=0.2` |
| `clearsight` | ClearSight / Visual Amplification Fusion | VAF attention intervention port with dynamic image bounds | layers `[9, 15)`, visual `1.15`, system/prefix `0.95` |
| `visattnsink` | See What You Are Told / Visual Attention Sink | sink identification, head filtering, visual attention redistribution port | layers `[2, 32)`, `tau=20`, `rho=0.5`, `summ=0.2`, `p=0.6` |

Implementation lives in
[`mitigation/src/interventions.py`](../src/interventions.py). It ports the
intervention rules onto the HuggingFace `LlavaForConditionalGeneration` stack
already used in this project, rather than vendoring three incompatible LLaVA
forks. These are controlled ports for mechanism testing, not bit-level official
reproductions.

## Evaluation Question

This track is not satisfied with aggregate gains. For POPE, every method is
reported with:

- accuracy, F1, balanced accuracy, and MCC;
- yes-rate, TPR, FPR, TNR, and invalid yes/no outputs;
- change from vanilla in yes-rate, TPR, and FPR;
- `Delta TPR - Delta FPR`, which is positive only when positive-class gains
  exceed the increased false-positive tendency.

For CHAIR caption generation, every method is reported with:

- `CHAIRi` and `CHAIRs`;
- average caption word count;
- average object mentions;
- average hallucinated object mentions.

This permits checking whether lower CHAIR comes from genuinely better grounding
or simply shorter, less object-rich captions.

## Method Notes

- [`pai/README.md`](pai/README.md)
- [`clearsight/README.md`](clearsight/README.md)
- [`visattnsink/README.md`](visattnsink/README.md)
