# PAI Attention-Only Port

Paper: *Paying More Attention to Image: A Training-Free Method for Alleviating
Hallucination in LVLMs* (ECCV 2024).

Official code consulted: `LALBJ/PAI`, especially `attention.py`,
`pope_eval.py`, and `chair_eval.py`.

The original method contains two parts:

1. Attention intervention: increase logits toward image tokens at each
   generation step.
2. CFG-style logits refinement using a text-only comparison branch.

This project intentionally implements only part 1, because the question is
whether attention manipulation itself produces the observed behavior shift.
For layers `[2, 32)`, the official modification on the latest query is:

```text
A_logit(image) <- A_logit(image) + alpha * abs(A_logit(image))
```

Default: `alpha=0.2`.
