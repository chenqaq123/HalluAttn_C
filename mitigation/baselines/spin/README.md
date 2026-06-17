# SPIN Controlled Port

Source method: *Mitigating Hallucinations in Vision-Language Models through
Image-Guided Head Suppression* (SPIN). The official release evaluates LLaVA-1.5
on POPE and CHAIR and suppresses non-routed attention heads according to
image-token attention.

## Local Scope

This repository implements a HuggingFace LLaVA controlled port rather than
vendoring the official LLaVA fork. For each active layer and text-query token,
the port:

1. sums each head's normalized attention mass over the dynamic image-token span;
2. keeps the top `routed_heads` fraction of heads;
3. multiplies non-routed head outputs by `small_num_mask` before the output
   projection.

The default POPE setting is `routed_heads=0.8` and `small_num_mask=0.1` over
layers `[0, 32)`, matching the official POPE hyperparameter scale while keeping
the same greedy decoding stack used by the other local mitigation baselines.
For CHAIR, the official configuration uses a larger routed-head ratio; run
`SPIN_ROUTED_HEADS=0.95` when making a direct CHAIR-oriented comparison.

## Run

```bash
METHODS=vanilla,spin \
RUN_CHAIR=0 \
POPE_SPLITS=random \
LIMIT=100 \
MITIGATION_EXP_NAME=pope_spin_smoke \
bash mitigation/scripts/run_parallel_mitigation.sh
```

After generation, summarize with:

```bash
python mitigation/scripts/audit_results.py \
  --result_root mitigation/results/pope_spin_smoke \
  --methods vanilla,spin
```

Use the same semantic-neighbor subset audit as the other baselines before
counting SPIN as evidence in the paper tables.

## Current Local Audit

A POPE-adversarial 120-row subset run is available under
`mitigation/results/pope_spin_adversarial_120/`, with semantic-neighbor metrics
under `mitigation/results/semantic_neighbor_audit/spin_adversarial_120_subset_eval/`.
It is an early negative signal rather than a full baseline result: SPIN raises
TPR from 0.850 to 1.000, but FPR rises from 0.183 to 0.983, yes-rate rises from
0.517 to 0.992, and related-present negative FPR rises from 0.204 to 1.000.
