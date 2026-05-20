# SinkDetect — Method Design

This document is the source of truth for *why* we compute the scores we compute.
It's written for a future-me reading the repo cold, and as the seed of the
methodology section of a paper.

> Status: current implementation design. PAS is treated as the comparison
> baseline; SinkDetect focuses on de-biased visual-attention shape signals and
> explicit ablations for sink removal and top-mass visual masking.

---

## 1. Phenomenon and gap

Object hallucination in LVLMs is the case where the model emits an object
word at decoding step *t* without genuine visual support in the image. Two
families of detection signals exist in the literature:

- **Logit-based**: NLL, Entropy, DoLa-style contrastive logits. Captures
  cases where the model is internally uncertain; fails for confident
  hallucinations.
- **Attention summary**: PAS (`A[q, T].sum()`), Image-Attention (IC,
  `A[q, V].sum()`), GLSim. Reduces a `|V|`-dim attention distribution to a
  single scalar, throwing away the *shape*.

Two parallel architectural observations motivate the de-biasing step:

1. **Attention sinks** (Xiao et al. ICLR'24; Sun et al. '24 Massive
   Activations). A small set of token positions act as architectural sinks
   regardless of input content. In LVLMs the visual stream also exhibits
   sink positions.
2. **RoPE position bias**. RoPE produces an attention component that depends
   only on relative position, not content. In a long visual sequence, this
   biases queries toward positionally adjacent visual tokens.

These two are **confounders**: they contribute mass to `A[q, V]` without
representing content-driven grounding. Existing summary statistics conflate
the confounders with the signal.

**Gap statement.** Existing detectors measure *how much* a query attends to
visual tokens, not *whether that attention has the shape of real grounding*.
The shape carries information that is invariant to sink/RoPE bias.

---

## 2. Core hypothesis

> **A grounded object generation produces a visual attention distribution
> that is (i) concentrated on a small set of content-relevant tokens,
> (ii) different from a content-independent null distribution for the same
> image, and (iii) consistent across layers.**
>
> Hallucinated object generation fails at least one of these properties.

This reframes detection as testing the shape of a distribution against a
null, rather than measuring the mass of a single summary.

---

## 3. De-biasing model and ablations

Let `a^{(l)}_q ∈ Δ^{|V|}` be the row of layer-*l* attention from query
position *q* (the object's preceding token) restricted to the visual span
`V`. We posit

    a^{(l)}_q  =  π_S · uniform(S^{(l)})              # sink null
                + π_R · p^{(l)}_RoPE(q, ·)            # position null
                + π_C · p^{(l)}_content(q, ·)         # signal
                + ε,
    π_S + π_R + π_C = 1.

- `S^{(l)}` are sink positions estimated per layer from pre-RoPE key norms.
- `p_RoPE` depends only on the query/key *positions*, not on image content;
  it is approximately the same for all images.
- `p_content` is the genuine content distribution — the regions the model
  uses as evidence.

Hallucination hypothesis: π_C → 0.

Detection objective: estimate π_C (or any monotone proxy).

The implementation keeps four attention branches for each layer:

    A       = original attention
    A^S     = sink-only: zero sink columns for text queries, no top-mass mask
    A^T     = topmass-only: apply visual top-mass mask, keep sinks
    A'      = purified: zero sink columns + apply visual top-mass mask

This gives a 2×2 ablation:

| branch | remove sinks | top-mass mask | purpose |
|---|---:|---:|---|
| `orig_*` | no | no | PAS-compatible baseline and raw shape scores |
| `sink_only_*` | yes | no | isolate sink removal |
| `topmass_only_*` | no | yes | isolate top-mass / visual-bias filtering |
| `purified_*` | yes | yes | full SinkDetect shape signal |

The headline method should compare `purified_*` against `topmass_only_*` and
`sink_only_*`: if full purification wins, sink removal and top-mass filtering
are complementary; if one ablation dominates, the method can be simplified.

---

## 4. Estimating the null without extra forward passes

The cleanest estimator of `p_RoPE` would be to run forward passes on
**noise images** (or all-mean-token replacements), averaging the attention
rows for the same query position across many noise samples. The resulting
distribution carries no content because no specific image is shown. This is
a few hundred extra forwards once, then cached. But it **changes the
experimental design**, so for v0 we take a cheaper, weaker null:

**Instruction-token null.** The user instruction string is fixed across
all images:

    "Please help me describe the image in detail."

The attention rows at instruction-token positions have

- **identical queries** (Q comes from the fixed string),
- **image-conditioned keys** (K comes from the visual encoder).

Their visual-side attention captures sink + RoPE + "generic prompt's gaze
over this specific image". It does *not* carry signal from any specific
object word — instruction queries don't know what the model will generate.

We define the per-layer instruction null as the row-average:

    p̂^{(l)}_null(j)  =  mean over q' ∈ Instr of  a^{(l)}_{q'}[j],   j ∈ V.

This is computable from the same attention matrix that produces the test
distribution, at zero additional FLOPs.

> Caveat. The instruction null is a *weaker* null than the noise-image null:
> it still contains "what's salient in this image regardless of the object
> word". So `KL(a_obj || p̂_null)` measures *the marginal grounding effort
> contributed by the object word itself*, not the total visual evidence.
> This is still a discriminative signal between hallucinated and grounded
> mentions, just on a tighter axis.

A noise-image null is a planned extension (`docs/roadmap.md` once written).

---

## 5. Scoring families

All distributions below are visual-token distributions extracted from one of
the four branches above. For branches that remove sinks (`sink_only_*` and
`purified_*`), sink positions are also stripped from the scoring distribution
before renormalization. For `topmass_only_*`, sinks are intentionally kept so
the branch remains a clean "top-mass without sink removal" ablation.

Sink positions per layer are detected via `mean − 1.5·std` thresholding on
pre-RoPE key L2 norms within the visual span, then clamped to `[3, 30]`.

For a single object mention at query position *q*:

### 5.1 CVG — Counterfactual Visual Grounding

Closeness between the object query's visual distribution and the
instruction null:

    CVG^{(l)}_KL  =  KL( ã^{(l)}_q  ||  p̃^{(l)}_null )
    CVG^{(l)}_JS  =  JSD( ã^{(l)}_q,    p̃^{(l)}_null )

The instruction null is treated as a language/prompt-prior visual gaze.
If the object query stays close to that null, it has not developed
object-specific visual grounding. Therefore the hallucination score uses
the negative divergence:

    S_CVG^{(l)}  =  −D( ã^{(l)}_q, p̃^{(l)}_null )

Large `S_CVG` means the object query is close to the language-prior null,
so it is more likely hallucinated.

Also computed against a `uniform-on-non-sink-visual` null for an architectural
floor:

    CVG^{(l)}_KL_uniform  =  KL( ã^{(l)}_q  ||  uniform(V \ S^{(l)}) )

The fixed instruction null is the main paper-facing null because it is shared
across images and does not depend on the generated object word. Local
generated-token nulls are implementation-level exploratory features and should
not be part of the main method unless a later ablation justifies them.

### 5.2 Concentration

Shape statistics of the test distribution, no null required:

    entropy^{(l)}     =  H( ã^{(l)}_q )                                  # high = hallu
    top_k_mass^{(l)}  =  sum of the largest k entries of ã^{(l)}_q       # low = hallu (negated)
    max_over_mean^{(l)} = max / mean                                      # low = hallu (negated)

These test the "concentrated on a small region" property directly.

### 5.3 CLC — Cross-Layer Consistency

Stack the per-layer sink-stripped distributions into a matrix
`P ∈ R^{L × |V|}` (one row per layer). Compute:

    clc_gen_jsd     =  H( mean of rows of P ) − mean H(row)
                       (generalised Jensen-Shannon divergence across L layers)

    clc_mean_pwjsd  =  mean_{i<j} JSD( P_i, P_j )

    clc_argmax_agree = fraction of layer-pairs (i,j) with argmax(P_i) == argmax(P_j)
                       (negated; high agreement = grounded)

A "mid-late layers only" version restricts to `[L/2−2, L−4]` because
grounding is typically cleanest in mid-late layers in the literature
(empirical, layer-range is a hyperparameter).

### 5.4 Branch prefixes and global versions

For every shape score, the branch name is encoded in the key:

    cvg_* / conc_* / clc_*                 # original A
    sink_only_cvg_* / sink_only_conc_* / sink_only_clc_*
    topmass_only_cvg_* / topmass_only_conc_* / topmass_only_clc_*
    purified_cvg_* / purified_conc_* / purified_clc_*

For each per-layer score, a `global_*` version averages **the distributions
first** (not the scores), then applies the same formula. That is:

    obj_avg   =  normalize( mean_l ã^{(l)}_q )
    null_avg  =  normalize( mean_l p̃^{(l)}_null )
    global_cvg_kl_instr  =  -KL( obj_avg || null_avg )

This is structurally different from "average the per-layer scores" and the
two should *not* be expected to give the same answer.

---

## 6. Fast metric iteration cache

Changing CVG signs, KL/JSD variants, concentration formulas, CLC aggregation,
or fusion groups should not require another LLaVA forward pass. Stage 2 can
therefore save a compact shape cache:

    SAVE_SHAPE_CACHE=1 bash scripts/run_parallel.sh

The cache stores, for each object mention and layer:

- object-query visual row;
- instruction-null visual row;
- sink mask;
- label, image id, token position, and object word;
- all four branches: `orig`, `sink_only`, `topmass_only`, `purified`.

Then shape metrics can be recomputed with:

    python scripts/recompute_shape_from_cache.py \
        --cache experiments/<exp>/shape_cache.npz

This cache is valid for formula-level iteration. It is **not** valid if the
model, prompt, sink detector, top-mass ratio, or purification operation itself
changes, because those change the cached rows.

---

## 7. Efficiency design

The main computational cost is not CHAIR or score aggregation. It is the
white-box attention extraction pass. Stage 2 currently runs a teacher-forced
forward on `image + prompt + generated caption` and asks the model to expose
attention. This is intrinsically expensive because explicit attention analysis
conflicts with FlashAttention-style kernels: FlashAttention is fast precisely
because it avoids materializing the full `QK^T` attention matrix, while
SinkDetect needs attention weights as data.

The current implementation therefore follows two efficiency principles:

1. **Keep attention-resident computation on GPU.** The adapter keeps the
   mean-over-head attention matrices and purified branches on the model
   device. CVG, concentration, CLC, and sink-mass scoring operate on those
   device tensors. Only scalar scores and optional cache rows are moved to
   CPU. This avoids the earlier bottleneck where every layer's full attention
   matrix was copied to CPU before scoring.
2. **Cache only formula-relevant rows for iteration.** When
   `SAVE_SHAPE_CACHE=1`, the saved artifact contains object-query and
   instruction-null visual rows, not arbitrary intermediate model state.
   This makes later metric changes cheap while keeping the cache tied to the
   exact prompt/model/purification configuration.

This still leaves a structural bottleneck: the implementation currently
materializes full `(seq_len × seq_len)` attention matrices even though the
shape scores mostly use only a small set of query rows:

- object mention positions;
- instruction-token positions for the null;
- optional nearby non-object positions;
- optional per-head rows for selected layers.

The next implementation target is therefore **row-level attention extraction**:
compute or retain only selected query rows over the visual span `V`, rather
than storing the full matrix. The method definition does not require full
attention; it requires distributions of the form

    a_q^V = softmax(Q_q K^T)[V].

A row-level implementation could recover these distributions from selected
`Q` rows and visual `K` columns, preserving the same scores while reducing
memory traffic and making the detector closer to modern efficient attention
kernels. This is the main engineering path to make SinkDetect practical at
larger scale.

---

## 8. Model-agnostic extension for commercial VLMs

Commercial VLMs usually do not expose attention maps, hidden states, logits,
or decoding internals. A naive black-box extension based on image
perturbation would overlap heavily with VCD/contrastive decoding-style
methods, because it would again ask whether output changes under visual
counterfactuals.

SinkDetect should instead be framed as a **post-hoc grounding verifier**:

1. Any generator, including a commercial VLM, produces a caption or answer.
2. A fixed white-box verifier VLM receives the original image and the
   generated text under teacher forcing.
3. SinkDetect extracts verifier-side visual attention for each object
   mention in that text.
4. CVG, concentration, and CLC estimate whether the text is visually
   supported.

In this extension, the commercial model is not introspected. Its output is
treated as a claim set to be verified. The attention distribution belongs to
the verifier, not to the generator.

This keeps the conceptual contribution intact:

> Hallucinated object mentions are visually unsupported claims whose
> verifier-side object-token attention remains close to a language/prompt
> prior rather than becoming object-specific.

It also gives a clean distinction from VCD and related decoding methods:

- VCD is a **generation-time intervention** on the model being decoded.
- SinkDetect is a **generation-after detector/verifier**.
- VCD needs access to decoding logits or at least generation control.
- Model-agnostic SinkDetect only needs the final text and the image.
- VCD tries to improve the output; SinkDetect estimates token-level support
  for an already produced output.

The verifier can be a single open-source VLM or an ensemble of verifiers
(e.g. LLaVA/Qwen-VL/InternVL). Agreement across verifiers would strengthen
claims that the score measures visual support rather than idiosyncrasies of
one architecture. The main paper can present the current attention-based
implementation as the white-box setting, and a follow-up section can evaluate
surrogate-verifier transfer to commercial VLM outputs.

---

## 9. Theoretical hook (for a paper)

Under the generative model in §3:

> **Proposition (informal).** Suppose hallucinated mentions satisfy
> `π_C = 0` and grounded mentions satisfy `π_C ≥ c_0 > 0`, and that
> `p_content` is distinguishable from the instruction-prior null. Then the
> divergence `D(a_q^V || p_null^V)` is larger in expectation for grounded
> mentions than for hallucinated mentions:
>
>   `E[ D | grounded ] − E[ D | hallu ]  ≥  c_0 · D( p_content || null )`,
>
> where `null = (π_S/(π_S+π_R))·uniform(S) + (π_R/(π_S+π_R))·p_RoPE`.
> The implemented hallucination score is `S_CVG = −D`, so the direction is
> reversed: larger `S_CVG` means closer to the null and therefore more likely
> hallucinated.

The point isn't the constants — it's that CVG is a consistent estimator of
the `π_C` gap, while PAS-style summary statistics conflate `π_C` with `π_S`
and `π_R`. A rigorous proof needs Pinsker plus a tail bound; not done.

---

## 10. What this is **not**

- Not a learned probe — every score is closed-form on the attention matrix.
- Not a decoding intervention — purely a detector/verifier. (Future: feed
  CVG into decoding-time early stopping or reranking; would be a follow-up
  paper.)
- Not single-attention-head — heads are averaged before any of this. Per-head
  analysis is a planned ablation.
- Not multi-pass — single forward pass per image.
- Not a black-box perturbation detector — the model-agnostic extension uses a
  white-box verifier, rather than perturbing the commercial generator.

---

## 11. Open questions to validate empirically

These are the questions the AUROC table in `experiments/<exp>/metrics.json`
will answer:

1. Does `purified_cvg_*` beat raw `cvg_*` and PAS image/prelim attention?
2. Does `purified_shape` beat both `sink_only_shape` and `topmass_only_shape`?
3. Which component matters more: sink removal or top-mass visual filtering?
4. Does `clc_gen_jsd` carry signal without any null reference?
5. Which layers contribute most to CVG / Concentration / CLC?
6. Does concentration already match CVG, or does the instruction-prior null
   add real signal?
7. Are CVG, Concentration, CLC, and PAS complementary under simple fusion?

Answer pattern that would support the design:

- (1) yes: shape > magnitude.
- (2) yes: full de-biasing is better than either ablation alone.
- (3) either sink or top-mass dominates: simplify the method accordingly.
- (4) yes: cross-layer consistency is a free signal.
- (6) CVG > concentration: the instruction-prior null does useful work.
- (7) yes: families are complementary, suggesting a fused detector.

If any of these fail, the design needs revision (see §12).

---

## 12. Likely failure modes and revision paths

- **Instruction null is too close to object null.** Mitigation: switch to
  noise-image null. Cost: small one-time calibration pass; clean theoretical
  story.
- **Sink detection is unstable across layers.** Mitigation: use a fixed,
  globally-shared sink set estimated from a calibration pass.
- **Top-mass dominates everything.** Mitigation: drop explicit sink removal
  from the main method, keep it as an analysis.
- **Sink removal dominates everything.** Mitigation: simplify purification to
  sink removal only and treat top-mass as an unnecessary truncation.
- **CLC dominates everything (results too good).** Possible because layer
  consistency might be a function of "the model is confident" rather than
  "grounded". Mitigation: compare against an Entropy/NLL baseline at the
  same operating point; show CLC carries information *beyond* uncertainty.
- **`p_content` isn't peaked in practice.** Mitigation: rely more on CVG
  null-distance and layer consistency, and treat top-k concentration as an
  auxiliary family rather than a core assumption.
- **Verifier transfer is weak for commercial VLM outputs.** Mitigation:
  calibrate the verifier on open-source generator outputs first, then test
  whether ranking transfers; consider verifier ensembles or object-detector
  agreement as auxiliary evidence.
- **Full attention extraction is too slow.** Mitigation: implement row-level
  visual attention extraction for object/instruction/null query rows, and
  restrict expensive per-head analysis to a small layer set.

---

## 13. References (from memory; verify before submission)

- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*,
  ICLR 2024.
- Sun et al., *Massive Activations in Large Language Models*, 2024.
- Hoang-Xuan et al., *PAS: Prelim Attention Score …*, CVPR 2026
  (sibling repo).
- Huang et al., *OPERA: Alleviating Hallucination in MLLM via Over-Trust
  Penalty*, CVPR 2024.
- Leng et al., *Mitigating Object Hallucinations in LVLMs through Visual
  Contrastive Decoding (VCD)*, CVPR 2024.
- Liu et al., *Paying More Attention to Image (PAI)*, 2024.
- Chuang et al., *DoLa*, ICLR 2024.
- Abnar & Zuidema, *Attention Rollout*, 2020.

A thorough literature review and citation-checking pass is still to do.
