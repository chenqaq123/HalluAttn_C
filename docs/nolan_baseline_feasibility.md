# NoLan Baseline Feasibility Note

Date: 2026-06-19

## Why This Matters

NoLan is now the closest public-code decoding baseline for the current ICML plan.
It tests a different hypothesis from TDEV: object hallucination can be reduced by
suppressing language priors when multimodal and text-only next-token
distributions are too similar. This is relevant to our motivation because it may
reduce generic hallucination, but it still does not explicitly verify the queried
target against semantic neighbors.

## Source Status

- Paper: `https://arxiv.org/abs/2602.22144`
- Official repository: `https://github.com/lingfengren/NoLan`
- Repository README says code is released and supports LLaVA-1.5, InstructBLIP,
  and Qwen-VL.
- The repository README describes integration by importing
  `nolan_utils.nolan_sample.evolve_nolan_sampling()` and adding
  `prepare_inputs_for_generation_cd` plus `input_ids_cd`, `cd_alpha`, and
  `cd_beta` plumbing to the model.

## Compatibility Audit

Current local environment:

| Component | Local env | NoLan requirement |
|---|---|---|
| Python env | `/home/chenguanxu/miniconda3/envs/latentGuard` | separate `nolan` env suggested |
| transformers | `4.57.6` | `4.31.0` |
| torch | `2.10.0+cu128` | `2.0.1` |

Important integration details from the official code:

1. NoLan monkey-patches `transformers.generation.utils.GenerationMixin.sample`.
2. The current repository mostly uses deterministic `model.generate(...,
   do_sample=False)` for POPE/CHAIR comparability, while NoLan's README example
   uses `do_sample=True`.
3. The NoLan sampling code expects a contrastive/text-only input path through
   `prepare_inputs_for_generation_cd`. HuggingFace `LlavaForConditionalGeneration`
   in our stack does not expose that method by default.
4. Directly downgrading transformers/torch inside `latentGuard` would risk
   breaking already-audited LLaVA, Qwen2.5-VL, OWLv2, and CHAIR scripts.

## Decision

Treat NoLan as a **P0 runnable candidate**, not as an already integrated
baseline. Do not patch the main environment globally.

The safer path is a guarded local compatibility port for the adversarial
semantic-neighbor subset:

1. Keep the existing `latentGuard` environment unchanged.
2. Implement or isolate a NoLan runner that computes multimodal and text-only
   logits per generation step for the same LLaVA-HF model path.
3. Preserve our normal strict yes/no parsing and semantic-neighbor audit metrics:
   MCC, TPR, FPR, yes rate, related-present FPR, plain FPR, and gap.
4. Report the run as `NoLan-compatible port` unless it uses the official monkey
   patch and model modifications exactly.
5. If the port diverges materially from official code, keep it as a diagnostic
   ablation and do not claim official NoLan reproduction.

## First Success Gate

Run only the POPE-adversarial semantic-neighbor subset first. The result is worth
scaling only if it reduces related-present FPR without a yes-rate shortcut and
without collapsing TPR. This matches the paper's core claim that better decoding
or stronger visual influence is not enough unless the target itself is verified.
