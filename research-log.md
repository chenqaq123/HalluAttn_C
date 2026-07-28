# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-07-28 | bootstrap | Registered a three-part expansion: matched image-caption mitigation on CHAIR/THRONE-Objects365/AMBER-generative, matched video-caption mitigation on Vript-HAL, and a position-controlled layer/head detector ablation. Audited the repository, existing result caches, model snapshots, datasets, and eight available RTX 4090 GPUs. Identified a comparability risk: PAI, DIML, and AllPath official code is architecture-specific, while SmartSight has no released code and under-specifies its segment threshold; protocols must distinguish official-native reproduction, disclosed-core reimplementation, and cross-architecture ports. |

