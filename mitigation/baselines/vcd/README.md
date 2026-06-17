# Visual Contrastive Decoding Port

This port implements the core VCD contrastive decoding rule for the local
HuggingFace LLaVA-1.5 runner.

## Scope

Official VCD contrasts the next-token logits from the original image and a
Gaussian-diffusion-noised image:

```text
logits_vcd = (1 + alpha) * logits(image) - alpha * logits(noisy_image)
```

It then applies the adaptive plausibility constraint from the original-image
logits. The official implementation samples from the adjusted distribution. For
controlled comparison with this repository's greedy POPE/CHAIR baselines, the
local method key `vcd` uses greedy argmax after the same contrastive adjustment.

## Defaults

| Parameter | Value |
|---|---:|
| `vcd_alpha` | 0.5 |
| `vcd_beta` | 0.1 |
| `vcd_noise_step` | 500 |

The image corruption follows the official VCD `add_diffusion_noise` utility:
a 1000-step sigmoid beta schedule and direct sampling from `q(x_t | x_0)` on
the processed LLaVA image tensor.

## Interpretation

This is a controlled VCD-greedy port, not a bit-level reproduction of the
sampling-based official code. It is useful for testing whether visual
contrastive logits reduce POPE false positives under the same strict parser,
semantic-neighbor subsets, and CHAIR caption diagnostics used for the other
mitigation baselines.
