# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-07-28 | bootstrap | Registered a three-part expansion: matched image-caption mitigation on CHAIR/THRONE-Objects365/AMBER-generative, matched video-caption mitigation on Vript-HAL, and a position-controlled layer/head detector ablation. Audited the repository, existing result caches, model snapshots, datasets, and eight available RTX 4090 GPUs. Identified a comparability risk: PAI, DIML, and AllPath official code is architecture-specific, while SmartSight has no released code and under-specifies its segment threshold; protocols must distinguish official-native reproduction, disclosed-core reimplementation, and cross-architecture ports. |
| 2 | 2026-07-28 | protocol | Froze three pre-result protocols: the 3×3×5 image-caption matrix, the 3×3 Vript-HAL matrix, and the layer-breadth × head-treatment detector factorial. Locked prompts, exact model variants, split isolation, method parameters, dynamic visual-token bounds, capacity controls, paired confidence intervals, and support criteria. Architecture-unsupported image baselines are explicitly formula-faithful ports; SmartSight is explicitly a disclosed-core reimplementation with a label-free completion of its unpublished segment threshold. |
