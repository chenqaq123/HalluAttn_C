# DAMRO Controlled Port

Source method: *DAMRO: Dive into the Attention Mechanism of LVLM to Reduce
Object Hallucination*. The official code selects top-k outlier visual tokens
with the CLIP/Vision-Transformer CLS attention map, then contrasts original
image logits against logits from the outlier-token visual branch.

## Local Scope

This repository implements a HuggingFace LLaVA controlled port rather than
vendoring DAMRO's modified `transformers` fork. The port:

1. hooks the selected CLIP vision layer's query/key projections;
2. ranks spatial visual tokens by CLS-to-patch attention;
3. replaces the full image-token block in the negative branch with the top-k
   projected outlier-token features;
4. applies the DAMRO contrastive logit rule with the same greedy decoding policy
   used by the local VCD port.

For LLaVA-1.5 POPE, the paper uses `alpha=2`, `beta=0.1`, and `topk=10`; these
are the local defaults. CHAIR uses a different `alpha` in the paper and should
be run separately if needed.

## Run

```bash
METHODS=vanilla,damro \
RUN_CHAIR=0 \
POPE_SPLITS=random \
LIMIT=100 \
MITIGATION_EXP_NAME=pope_damro_smoke \
bash mitigation/scripts/run_parallel_mitigation.sh
```

After generation, summarize with:

```bash
python mitigation/scripts/audit_results.py \
  --result_root mitigation/results/pope_damro_smoke \
  --methods vanilla,damro
```

Count DAMRO as paper evidence only after POPE behavior and semantic-neighbor FPR
are audited under matched sample IDs.

## Current Local Audit

A 4-row POPE-random smoke run is available under
`mitigation/results/pope_damro_smoke/`. It is only a runtime validation: both
vanilla and DAMRO have `invalid=0`, and each DAMRO prediction records the
selected outlier indices. The run is too small for behavioral claims; the next
required step is an adversarial subset plus semantic-neighbor FPR audit.
