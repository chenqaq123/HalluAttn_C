# OPERA Official-Hook Interface

Source method: *OPERA: Alleviating Hallucination in Multi-Modal Large Language
Models via Over-Trust Penalty and Retrospection-Allocation* (CVPR 2024).

The official OPERA release modifies `transformers.generate` and adds an
`opera_decoding=True` beam-search path. It requires:

- `num_beams=5` by default;
- `output_attentions=True`;
- `key_position` with `image_start`, `image_end`, and `response_start`;
- `scale_factor`, `threshold`, `num_attn_candidates`, and `penalty_weights`.

This repository now exposes a guarded `opera` method key, but deliberately does
not implement a greedy approximation. If the active Python environment does not
contain the official modified transformers hook, `METHODS=opera` fails before
writing predictions.

Check the active environment with:

```bash
python mitigation/scripts/check_opera_support.py
```

Expected paper use:

1. Count OPERA only when `ready_for_official_opera` is `true`.
2. Run a small adversarial/semantic-neighbor subset first.
3. Promote to full POPE only if the subset improves target-discriminative FPR,
   not merely aggregate F1 or recall.

Example subset command after installing the OPERA transformers fork:

```bash
RUN_CHAIR=0 \
POPE_SPLITS=adversarial \
LIMIT=120 \
METHODS=vanilla,opera \
MITIGATION_EXP_NAME=pope_opera_adversarial_120 \
bash mitigation/scripts/run_parallel_mitigation.sh
```
