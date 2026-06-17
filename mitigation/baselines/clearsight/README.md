# ClearSight VAF Port

Paper: *ClearSight: Visual Signal Enhancement for Object Hallucination
Mitigation in Multimodal Large Language Models* (CVPR 2025).

Official code consulted: `ustc-hyin/ClearSight`, especially
`visaug/inference/AttnAdapter.py` and `visaug/inference/infer_pope.py`.

The Visual Amplification Fusion intervention modifies attention logits in
middle decoder layers, where the paper identifies modality fusion:

```text
A_logit(text -> image)  <- 1.15 * A_logit(text -> image)
A_logit(text -> system) <- 0.95 * A_logit(text -> system)
```

The official LLaVA path applies this at layers `9..14` inclusive. This port
uses dynamically detected image-token bounds instead of the official fixed
system/image token offsets.
