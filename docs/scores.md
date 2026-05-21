# Score Reference

Exhaustive list of every key written into new `experiments/<exp>/raw_scores.npz`
and `metrics.json::roc_auc` runs. Sign convention everywhere: **higher value
⇒ more likely hallucination**.

For score families that have both per-layer and global variants, only the
template is shown. `{l}` runs `0..n_layers-1` (32 for LLaVA-1.5-7B).

For where each is computed, see [scoring.py](../src/sinkdetect/scoring.py)
and [grounding.py](../src/sinkdetect/grounding.py).

---

## Notation

- `A`              — layer's original attention matrix `(seq, seq)`, mean over heads.
- `A^S`            — sink-removal-only attention, without top-mass masking.
- `A^T`            — top-mass-only attention, without sink removal.
- `A'`             — fully purified attention after sink removal and top-mass masking.
- `A^R`            — no-RoPE attention recomputed from pre-RoPE Q/K.
- `q`               — `token_pos`, absolute index of the token preceding the object word.
- `V`               — visual token span `[vis_start, vis_end)`, dynamically detected per image.
- `S^{(l)}`         — sink positions detected in layer `l` (subset of `V`).
- `instr_mask`      — bool mask over instruction tokens (system + user + ASSISTANT:).
- `ã_q`             — `normalize( strip_sinks( A[q, vis_start:vis_end] ) )`.
- `p̃_null`         — `normalize( strip_sinks( mean over q' ∈ instr  A[q', vis_start:vis_end] ) )`.

---

## Branches

`A^S` is the sink-removal-only ablation: sink columns zeroed for text queries,
then row-renormalized.

`A^T` is the top-mass-only ablation: top-mass-`ratio` visual masking applied,
then row-renormalized.

`A'` is the full post-purification matrix: sink columns zeroed for text queries
and top-mass-`ratio` visual mask applied, then row-renormalized. Comparing
`A^S`, `A^T`, and `A'` tests whether sink removal, top-mass masking, or their
combination adds detection signal.

New runs intentionally do **not** emit PAS-style scalar attention-mass keys
(`*_prelim_attn`, `*_image_attn`, `*_bos_attn`) or their differential shift
variants. PAS is now treated as an external baseline, not part of the active
score output.

If `--compute_no_rope_attention` is enabled, the same shape families are also
computed on no-RoPE branches:

- `no_rope_*` on `A^R`;
- `no_rope_sink_only_*` on sink-removed `A^R`;
- `no_rope_topmass_only_*` on top-mass-only `A^R`;
- `no_rope_purified_*` on sink-removed + top-mass `A^R`.

---

# CVG / CLC / Concentration family

## CVG — distance from null

| key | formula |
|---|---|
| `cvg_kl_instr_layer_{l}`         | `−KL( ã^{(l)}_q  \|\|  p̃^{(l)}_null )` — close to instruction/language-prior null = hallu |
| `cvg_kl_uniform_layer_{l}`       | `−KL( ã^{(l)}_q  \|\|  uniform(V \ S^{(l)}) )` |
| `cvg_jsd_instr_layer_{l}`        | `−JSD( ã^{(l)}_q,  p̃^{(l)}_null )` |
| `cvg_kl_local_nonobj_layer_{l}`  | `−KL( ã^{(l)}_q  \|\|  p̃^{(l)}_local )`, where local null averages nearby generated non-object token rows |
| `cvg_jsd_local_nonobj_layer_{l}` | `−JSD( ã^{(l)}_q,  p̃^{(l)}_local )` |
| `global_cvg_kl_instr`            | `−KL( ã_avg  \|\|  p̃_null_avg )` |
| `global_cvg_jsd_instr`           | `−JSD( ã_avg,  p̃_null_avg )` |
| `global_cvg_kl_local_nonobj`     | `−KL( ã_avg  \|\|  p̃_local_avg )` |
| `global_cvg_jsd_local_nonobj`    | `−JSD( ã_avg,  p̃_local_avg )` |

`ã_avg = normalize( mean_l ã^{(l)}_q )`, likewise `p̃_null_avg`. We average
**the distributions** then re-score, not the scores. KL/JSD all in nats.

Every CVG / concentration / CLC key also has:

- a `sink_only_*` variant, computed on `A^S`.
- a `topmass_only_*` variant, computed on `A^T` without stripping sink
  positions from the score distributions.
- a `purified_*` variant, computed on `A'` after sink removal.

This gives the main ablation: sink removal only vs. top-mass only vs. both.

## Concentration — peakedness of the test distribution

| key | formula |
|---|---|
| `conc_entropy_layer_{l}`             | `H( ã^{(l)}_q )` (high = hallu, no negation) |
| `conc_top1_mass_layer_{l}`           | `−max(ã^{(l)}_q)` |
| `conc_top5_mass_layer_{l}`           | `−sum of top-5 entries` |
| `conc_top10_mass_layer_{l}`          | `−sum of top-10 entries` |
| `conc_max_over_mean_layer_{l}`       | `−( max / mean )` |
| `global_conc_entropy`                | `H( ã_avg )` |
| `global_conc_top{5,10}_mass`         | `−` top-k mass of `ã_avg` |
| `global_conc_max_over_mean`          | `−( max / mean )` of `ã_avg` |

## CLC — cross-layer consistency

Let `P ∈ R^{L × |V|}` be the per-layer sink-stripped object distributions.

| key | formula |
|---|---|
| `clc_gen_jsd`              | `H( mean of rows of P ) − mean H(row)` — generalised JSD |
| `clc_mean_pairwise_jsd`    | `mean_{i<j} JSD(P_i, P_j)` |
| `clc_gen_jsd_midlate`      | same as `clc_gen_jsd` but restricted to layers `[L/2−2, L−4]` |
| `clc_argmax_agree`         | `−` fraction of layer-pairs with same argmax |

CLC is a single number per mention (not per layer). High generalised JSD ⇒
layers disagree ⇒ hallucination.

## CVG — shuffled null

A permuted version of `ã_q` used as a content-free null. If the original
distribution is close to its shuffle, it is weakly structured (diffuse =
hallucination).

| key | formula |
|---|---|
| `cvg_kl_shuffled_layer_{l}`       | `−KL( ã^{(l)}_q  \|\|  permute(ã^{(l)}_q) )` — close to shuffle = unstructured = hallu |
| `cvg_jsd_shuffled_layer_{l}`      | `−JSD( ã^{(l)}_q,  permute(ã^{(l)}_q) )` |
| `global_cvg_kl_shuffled`          | `−KL( ã_avg  \|\|  permute(ã_avg) )` |
| `global_cvg_jsd_shuffled`         | `−JSD( ã_avg,  permute(ã_avg) )` |

## CVG — cross-image null

Instruction null distributions from other images, averaged, provide an
image-independent "what does the model typically look at" baseline. If the
object query is far from cross-image nulls, it has image-specific grounding
(not hallucination), so we negate.

| key | formula |
|---|---|
| `cvg_kl_crossimg_layer_{l}`       | `−KL( ã^{(l)}_q  \|\|  mean_{other images} p̃_null^{(l)} )` |
| `cvg_jsd_crossimg_layer_{l}`      | `−JSD( ã^{(l)}_q,  mean_{other images} p̃_null^{(l)} )` |
| `global_cvg_kl_crossimg`          | `−KL( ã_avg  \|\|  cross-image null avg )` |
| `global_cvg_jsd_crossimg`         | `−JSD( ã_avg,  cross-image null avg )` |

Cross-image nulls are collected from a rolling buffer of the 20 most recently
processed images.

## Per-head analysis (key layers only)

Computed only for layers in `PER_HEAD_LAYERS = [0, 1, 10, 22, 31]`. Instead of
averaging over 32 attention heads, we compute per-head statistics.

| key | formula |
|---|---|
| `ph_vis_mass_std_layer_{l}`           | `std_{heads}( sum A_h[q, V\S] )` — high = heads disagree = hallu |
| `ph_vis_mass_mean_layer_{l}`          | `−mean_{heads}( sum A_h[q, V\S] )` — negated, same as mean image attn |
| `ph_argmax_agreement_layer_{l}`       | `−fraction of heads with same argmax visual position` — high agreement = grounded → negate |
| `ph_entropy_std_layer_{l}`            | `std_{heads}( H(A_h[q, V\S]) )` — high = heads disagree on peakedness |
| `ph_vis_mass_max_layer_{l}`           | `−max_{heads}( visual mass )` — negated |
| `ph_vis_mass_min_layer_{l}`           | `−min_{heads}( visual mass )` — negated |

---

## File layout

- `metrics.json::roc_auc` — every key above with its AUROC (sorted desc).
- `metrics.json::objects` — `{total, hallucinated, non_hallucinated}`.
- `metrics.json::chair_metrics` — CHAIRi pooled across shards (and per-shard
  CHAIRi/CHAIRs in the `shards` list).
- `raw_scores.npz[key]` — `float32` 1-D array, one entry per object mention
  in evaluation order.
- `raw_scores.npz["labels"]` — `int32` 1-D array of `{0, 1}` labels.
