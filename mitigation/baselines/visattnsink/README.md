# Visual Attention Sink Port

Paper: *See What You Are Told: Visual Attention Sink in Large Multimodal
Models* (ICLR 2025).

Official code consulted: `seilk/VisAttnSink`, especially
`src/logic/logic.py`, `src/logic/constants.py`, and `src/model/llava.py`.

For LLaVA-1.5-7B, the official intervention:

1. Detects prompt sink tokens using hidden dimensions `2533` and `1415` with
   threshold `tau=20`.
2. Selects heads/queries whose visual attention is sufficiently strong
   (`summ=0.2`) and not dominated by sink locations (`rho=0.5`).
3. Retains fraction `p=0.6` of sink attention and redistributes the removed
   mass over non-sink visual tokens in proportion to their existing attention.

This port preserves that procedure while using the current HuggingFace LLaVA
layer interface and dynamic visual-token bounds. The port follows the effective
official execution path on layers `[2, 32)`; although the example YAML exposes
an `except_last_layer` option, that flag is not wired into the official
redistribution path.
