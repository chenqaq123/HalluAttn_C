# SinkDetect — Method Design

This document is the source of truth for *why* we compute the scores we compute.
It's written for a future-me reading the repo cold, and as the seed of the
methodology section of a paper.

> Status: current research framing. The strongest verified result is not yet a
> new training-free detector, but a controlled re-evaluation: global AUROC for
> attention-based hallucination detection is strongly confounded by object
> generation position, and mean-over-head attention aggregation hides much of
> the signal that survives in late-layer per-head features. Mitigation-side
> experiments now show that attention interventions can shift output behavior
> (e.g. yes-prior on POPE or object richness in captions) without improving
> object-level hallucination.

---

## 1. Phenomenon and gap

Object hallucination in LVLMs is the case where the model emits an object
word at decoding step *t* without genuine visual support in the image. Two
families of detection signals exist in the literature:

- **Logit-based**: NLL, Entropy, DoLa-style contrastive logits. Captures
  cases where the model is internally uncertain; fails for confident
  hallucinations.
- **Representation / logit-lens**: IC, GLSim. Captures whether internal visual
  states support the object, often more robust than attention mass.
- **Attention summary**: PAS (`A[q, T].sum()`), SVAR/image-attention
  (`A[q, V].sum()`), and related visual-mass scores. These reduce a
  high-dimensional attention pattern to a scalar, throwing away head/layer
  structure and spatial shape.

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

The latest experiments reveal a third confounder:

3. **Object generation position prior.** In LLaVA captions, objects generated
   later are much more likely to be CHAIR hallucinations. The generated-token
   position `gen_pos = token_pos − prompt_end_idx` alone reaches AUROC
   0.8304 on the current COCO/LLaVA run. Any score that drifts monotonically
   with `gen_pos` can obtain high global AUROC without detecting visual
   grounding within a fixed position range.

**Gap statement.** Existing attention detectors often measure *how much* a
query attends to visual/text tokens, not whether the attention evidence is
position-controlled, head-specific, and genuinely grounded. Existing
attention-based mitigation methods often make the symmetric move: they increase
or re-route visual attention, assuming this will improve grounding. Aggregate
metrics then make both sides worse: global AUROC can reward generation
position, and POPE accuracy/F1 can reward a shifted yes/no answer prior.

The current project therefore treats evaluation confounds, attention
aggregation, and attention intervention as one connected problem:

> **Visual attention allocation is not equivalent to object-level visual
> evidence verification.**

Attention reveals where computation is routed. Hallucination detection and
mitigation require a harder property: whether the current object claim is
supported by image evidence. Existing methods often optimize visibility rather
than verifiability.

A more concrete failure mode is **associated-evidence grounding**. The model
may attend to a real visual region that is semantically or contextually related
to the queried object, while the queried object itself is absent. For example,
when asked whether a *sink* is present in an image containing a *toilet*, the
model can route attention to the toilet/bathroom region, activate a
sink-related language prior through object co-occurrence and embedding
similarity, and answer "yes". The attention map then looks visually grounded,
but the evidence is only associated with the claim, not discriminative for the
target object.

This mechanism refines the thesis:

> Attention may be selective for visually or semantically associated evidence,
> but hallucination detection and mitigation require target-discriminative
> evidence.

The newly added reference
`ref/adavib_adaptively_constraining_information_flow_2502.20750.pdf`
supports this direction. Bai et al. argue that object hallucination can arise
from overconfidence in irrelevant visual features when soft visual tokens map
into the LLM word-embedding space, and they analyze semantic similarity between
visual tokens and word embeddings. LURE and POPE provide complementary support
from the data side: co-occurring objects are more likely to be hallucinated.
Together, these works support our explanation that attention can correctly
route to visual context while still failing object-level verification.

---

## 2. Unifying Thesis

Detection and mitigation look like separate tasks, but attention-based methods
for both usually share the same proxy:

    visual attention quality  ≈  visual grounding quality

Detection methods instantiate this proxy by scoring attention distributions:
PAS and SVAR measure text/visual mass, CVG and concentration measure visual
shape, and related methods summarize a high-dimensional attention pattern into
a scalar hallucination score.

Mitigation methods instantiate the same proxy operationally: PAI-attention-only
pushes visual attention logits upward, ClearSight amplifies mid-layer
text-to-image attention while suppressing system-prefix attention, and Visual
Attention Sink redistributes sink mass toward non-sink visual tokens.

Our results suggest the proxy breaks in both directions:

- **Detection:** coarse attention scores can look strong globally while
  exploiting generation-position confounds rather than object grounding.
- **Mitigation:** increasing visual influence can increase positive responses
  or object-rich captions without improving object-presence discrimination.
- **Diagnosis:** attention is not empty; late-layer per-head features contain
  signal. The failure is the assumption that simple attention mass, shape, or
  amplification is already evidence verification.

The associated-evidence mechanism explains why the proxy can fail even when the
attention map is not obviously wrong. A visual-attention intervention can
increase attention on a real object or scene region that is correlated with the
target word. That increased attention can strengthen a plausible but false
object claim, producing a yes-prior shift or richer caption, while leaving the
model unable to distinguish "the image contains a toilet" from "the image
contains a sink". Therefore the relevant question is not only whether visual
attention increased, but whether the attended evidence is
**target-discriminative**.

This gives the paper-level story:

    attention as proxy
      -> proxy breaks under controlled detection evaluation
      -> interventions based on the proxy also shift behavior without
         reliable hallucination reduction
      -> future methods need verification-aware attention analysis

One-sentence thesis:

> Attention reveals where a model looks, but not whether it verifies what it
> says.

---

## 3. Current Claim Structure

We separate claims into what is already supported by experiments and what
remains a hypothesis.

### 3.1 Verified Claims

1. **Detection scores are position-confounded.** On the current
   LLaVA-1.5-7B / COCO run, object generation position alone reaches AUROC
   0.8304. PAS and SVAR obtain high global AUROC but collapse under
   within-bin, matched-pair, and residual evaluation. The same conclusion
   holds under stronger post-hoc controls: nonlinear residualization and
   same-object-word matched AUROC.
2. **Mean-over-head attention is too coarse.** Closed-form shape scores on
   mean-over-head attention mostly fail under position control.
3. **Attention is not empty.** A supervised diagnostic probe over late-layer
   per-head attention-shape features recovers substantial position-controlled
   signal. This suggests that the failure is not attention itself, but coarse
   aggregation and uncontrolled evaluation.
4. **Attention interventions do not guarantee grounding improvements.** In the
   current mitigation run, PAI-attention-only gives no POPE/CHAIR benefit,
   ClearSight primarily shifts POPE responses toward "yes" with comparable or
   larger FPR increases, and Visual Attention Sink increases caption object
   richness together with hallucinated object mentions.
5. **Mitigation claims must be component-scoped.** PAI is currently evaluated
   as attention-only, excluding its contrastive/CFG branch. ClearSight and
   Visual Attention Sink are ported attention interventions. Paper claims
   should name this scope explicitly.
6. **Mechanism audit is the next decisive mitigation experiment.** The
   implemented attention-shift audit records pre/post visual, prefix, and sink
   attention mass for each POPE sample. This tests whether the interventions
   change attention as intended and whether those changes align with TP rather
   than FP behavior.
7. **Associated visual evidence is a plausible mechanism, not a side note.**
   The newly added AdaVIB reference analyzes hallucination as overconfidence in
   irrelevant visual features after visual tokens enter the LLM embedding
   space. Together with POPE/LURE-style co-occurrence findings and our
   mitigation audit, this supports a concrete failure mode: attention can route
   to a real, semantically related region while the answer still lacks
   target-discriminative evidence. The toilet/sink case is therefore not just
   anecdotal; it illustrates how attention to a related object can support a
   false "yes" when the model does not verify the queried object itself.

### 3.2 Hypotheses Still Under Test

1. **Head-selection mitigation may be a positive counterexample.** Papers that
   select or enhance specific heads might align with the per-head diagnostic
   finding. They must be reproduced before we claim that mitigation methods are
   generally flawed.
2. **A fair training-free per-head detector is not implemented yet.** The
   supervised per-head probe is an evidence test / upper bound, not an
   apples-to-apples detector against PAS, SVAR, IC, or GLSim.

---

## 4. Original Shape Hypothesis

> **A grounded object generation produces a visual attention distribution
> that is (i) concentrated on a small set of content-relevant tokens,
> (ii) different from a content-independent null distribution for the same
> image, and (iii) consistent across layers.**
>
> Hallucinated object generation fails at least one of these properties.

This reframes detection as testing the shape of a distribution against a
null, rather than measuring the mass of a single summary.

The row-cache experiments only weakly support this hypothesis for
mean-over-head attention. The per-head diagnostic revises it: if shape carries
grounding information, it appears to live in specific late-layer heads and is
washed out by averaging.

---

## 5. De-biasing model and ablations

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

The implementation keeps four standard attention branches for each layer:

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

For the explicit RoPE ablation, Stage 2 can additionally compute

    A^R = softmax(Q_preRoPE K_preRoPE^T / sqrt(d) + causal_mask)

where `Q_preRoPE` and `K_preRoPE` are captured inside the attention adapter
before applying rotary position embeddings. This branch tests whether the
shape scores improve when the positional rotation is removed from attention
weights. When enabled, the same sink/top-mass variants are emitted:

    no_rope_*,
    no_rope_sink_only_*,
    no_rope_topmass_only_*,
    no_rope_purified_*.

This no-RoPE branch is more expensive because it materializes an additional
attention matrix per layer. It should be used for targeted ablations and
shape-cache generation, not as the default full-run setting.

Important: removing RoPE from attention weights is **not** enough to remove
the generation-position prior. RoPE is an attention-level relative-position
mechanism; `gen_pos` is a decoding/data-distribution confounder. The current
results show that no-RoPE scores can still be position-correlated, and raw
scores can remain high under global AUROC only because later object mentions
are more often hallucinated.

---

## 6. Estimating the null without extra forward passes

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

## 7. Scoring families

All distributions below are visual-token distributions extracted from one of
the four branches above. For branches that remove sinks (`sink_only_*` and
`purified_*`), sink positions are also stripped from the scoring distribution
before renormalization. For `topmass_only_*`, sinks are intentionally kept so
the branch remains a clean "top-mass without sink removal" ablation.

Sink positions per layer are detected via `mean − 1.5·std` thresholding on
pre-RoPE key L2 norms within the visual span, then clamped to `[3, 30]`.

For a single object mention at query position *q*:

### 7.1 CVG — Counterfactual Visual Grounding

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

### 7.1.1 Position-Calibrated CVG

The row-cache experiments show that instruction-token and uniform nulls can be
dominated by generation position. In particular, `sink_only_cvg_kl_uniform`
reaches high global AUROC but falls to roughly random under position bins. The
next design target is therefore a position-calibrated CVG:

    PC-CVG(t) = S_CVG(t) − E[ S_CVG | gen_pos=t ]

or a local contrastive version:

    PC-CVG_local(t) = −D( a_obj_noRoPE(t) || a_local_null_noRoPE(t) )

where `a_local_null` is the average visual attention row of nearby generated
non-object tokens. The local null is intended to share the same image, similar
context length, and similar residual position bias as the object token. A
strong object-specific grounding signal should appear as a difference between
the object row and this local non-object baseline.

The current local-null implementation is deliberately conservative and weak:
it only takes nearby non-object generated tokens. Refinements to test:

- exclude punctuation, stopwords, subword fragments, and generic function
  tokens from the local null;
- weight local null rows by distance from the object token;
- compare deltas in concentration and peak overlap, not only KL/JSD;
- evaluate only with position-controlled metrics (see §7.5).

### 7.2 Concentration

Shape statistics of the test distribution, no null required:

    entropy^{(l)}     =  H( ã^{(l)}_q )                                  # high = hallu
    top_k_mass^{(l)}  =  sum of the largest k entries of ã^{(l)}_q       # low = hallu (negated)
    max_over_mean^{(l)} = max / mean                                      # low = hallu (negated)

These test the "concentrated on a small region" property directly.

### 7.3 CLC — Cross-Layer Consistency

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

### 7.4 Branch prefixes and global versions

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

### 7.5 Evaluation metrics: global vs position-controlled

Global AUROC is still useful as a diagnostic, but it is insufficient for the
main claim. AUROC means the probability that a random hallucinated object
receives a higher score than a random grounded object. If hallucinated objects
occur later and a score increases with generation position, global AUROC will
be high even when the score has no within-position discriminative power.

For the current COCO/LLaVA row-cache run:

    AUROC(gen_pos only) = 0.8304

For `sink_only_conc_top10_mass_layer_0`:

    global AUROC     = 0.8099
    same-bin AUROC   = 0.5087
    cross-bin AUROC  = 0.8267
    same-bin pairs   = 5.29%
    cross-bin pairs  = 94.71%

Thus the apparent global performance is dominated by cross-bin comparisons,
mostly late hallucinated objects versus early grounded objects.

Paper-facing evaluation should report:

- **Overall AUROC**: legacy comparability only.
- **Position-only AUROC**: `gen_pos` as a baseline confounder.
- **Within-bin AUROC**: AUROC inside generation-position bins, then weighted
  by valid sample count.
- **Matched-pair AUROC**: AUROC over positive/negative pairs with
  `|gen_pos_pos − gen_pos_neg| ≤ δ`.
- **Residual AUROC**: regress or smooth out `E[score | gen_pos]`, then compute
  AUROC on the residual.

A useful attention score must beat the position-only baseline in a meaningful
way or retain above-random performance under within-bin/matched-pair/residual
evaluation.

---

## 8. Fast metric iteration cache

Changing CVG signs, KL/JSD variants, concentration formulas, CLC aggregation,
or fusion groups should not require another LLaVA forward pass. Stage 2 can
therefore save compact caches.

### 8.1 Legacy branch cache

    SAVE_SHAPE_CACHE=1 bash detection/scripts/legacy/run_parallel.sh

The cache stores, for each object mention and layer:

- object-query visual row;
- instruction-null visual row;
- sink mask;
- label, image id, token position, and object word;
- all four branches: `orig`, `sink_only`, `topmass_only`, `purified`.

Then shape metrics can be recomputed with:

    python detection/scripts/legacy/recompute_shape_from_cache.py \
        --cache experiments/<exp>/shape_cache.npz

This cache is valid for formula-level iteration. It is **not** valid if the
model, prompt, sink detector, top-mass ratio, or purification operation itself
changes, because those change the cached rows.

### 8.2 Attention-row cache for current ablations

The current experimental path is an explicit two-stage attention-row cache:

    python detection/scripts/cache_attention_rows.py \
        --generation_json experiments/<exp>/generation.json \
        --output_dir experiments/<exp>_rows \
        --cache_layers 0,1,2,3,4

This stage caches the minimal visual rows needed by all shape metrics:

- object-query visual row;
- instruction-null visual row after filtering likely object words from the
  instruction span;
- local generated non-object visual row within a small window around the
  object mention;
- sink mask per cached layer;
- the same rows for both original attention and true no-RoPE attention
  recomputed from pre-RoPE Q/K.

It intentionally does **not** store `sink_only`, `topmass_only`, or
`purified` rows. Those are derived in the recompute stage by zeroing cached
sink columns and/or applying top-mass masks to the cached object row:

    python detection/scripts/recompute_from_row_cache.py \
        --cache experiments/<exp>_rows/attention_row_cache.npz \
        --ratio 0.5

This means `ratio`, CVG null choice, KL/JSD sign, concentration formulas, CLC
aggregation, and sink-removal ablations can be changed without another model
forward. The cache remains invalid if the model, prompt, generated captions,
cached layer set, RoPE-removal definition, or sink detector changes.

The row cache is also shardable:

    python detection/scripts/cache_attention_rows.py ... --shard_idx 0 --num_shards 4
    python detection/scripts/merge_shards.py --mode row_cache --output_dir experiments/<exp>_rows --num_shards 4

---

## 9. Efficiency design

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

This still leaves a structural bottleneck: the current row-cache worker still
runs the model in a mode that exposes attention, then copies only selected
visual rows to CPU. That is much cheaper for iteration, but the forward itself
still materializes full `(seq_len × seq_len)` attention matrices even though
the shape scores mostly use only a small set of query rows:

- object mention positions;
- instruction-token positions for the null;
- optional nearby non-object positions;
- optional per-head rows for selected layers.

The next implementation target is therefore **true row-level attention extraction**:
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

## 10. Model-agnostic extension for commercial VLMs

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

## 11. Mitigation-side diagnostic

The same evaluation concern applies to hallucination **mitigation** benchmarks.
In POPE-style object-existence QA, a method can improve aggregate accuracy or
F1 by changing the answer prior, not necessarily by improving grounding. In
particular, an intervention that makes the model answer "yes" more often will
tend to:

- increase recall / TPR on positive object queries;
- increase false positives / FPR on negative object queries;
- sometimes improve F1 while worsening hallucination on absent-object cases.

The mitigation-side diagnostic should therefore report:

- **Yes rate**: fraction of all responses judged as yes.
- **TPR / Recall**: yes rate on present-object queries.
- **FPR**: yes rate on absent-object queries.
- **TNR / Specificity**: no rate on absent-object queries.
- **Balanced accuracy**: `(TPR + TNR) / 2`.
- **MCC**: correlation-style metric less sensitive to yes/no skew.
- **Asymmetric gain**: `ΔTPR − ΔFPR` relative to the base model.

Interpretation:

| pattern | interpretation |
|---|---|
| `ΔTPR > 0`, `ΔFPR <= 0` | genuine mitigation candidate |
| `ΔTPR > ΔFPR > 0` | useful but answer-prior shift exists |
| `ΔFPR >= ΔTPR > 0` | likely yes-bias / threshold shift |
| `ΔTPR ≈ ΔFPR` | mostly global answer-prior movement |

This section is intentionally conservative. The `mitigation/` track now
implements controlled generation for:

- PAI attention manipulation only, deliberately excluding its contrastive
  decoding / CFG logits branch;
- ClearSight Visual Amplification Fusion (VAF);
- Visual Attention Sink redistribution from *See What You Are Told*;
- vanilla greedy generation as the matched comparison anchor.

The implementation runs the same method on POPE and CHAIR. On POPE it retains
per-question outputs and reports yes-rate, TPR, FPR, balanced accuracy, MCC,
and `Delta TPR - Delta FPR` against vanilla. On CHAIR it regenerates captions
for the fixed COCO manifest and reports CHAIR scores together with caption
length, object-mention count, and hallucinated-mention count.

Current mitigation result:

- **PAI-attention-only** gives essentially no benefit. It slightly decreases
  POPE accuracy/F1 and does not reduce CHAIR.
- **ClearSight** increases POPE yes-rate by about 3.6 points on average. TPR
  rises by about 3.5 points, but FPR rises by about 3.8 points, so the net
  `Delta TPR - Delta FPR` is negative. On adversarial POPE, FPR increases more
  than TPR and accuracy drops.
- **Visual Attention Sink** mildly increases yes-rate and FPR on POPE, and on
  CHAIR it makes captions longer, object mentions more numerous, and
  hallucinated mentions more frequent.

These results support the unified thesis: increasing or re-routing visual
attention can change output behavior without improving object-level evidence
verification. Head-selection / head-enhancement methods remain outside the
implemented set and may be positive counterexamples.

---

## 12. Theoretical hook (for a paper)

Under the generative model in §5:

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

## 13. What this is **not**

- Not currently a learned detector — the main closed-form SinkDetect scores
  are training-free. The per-head logistic model is a **diagnostic probe** and
  must be reported separately from training-free baselines.
- Not a decoding intervention — purely a detector/verifier. (Future: feed
  CVG into decoding-time early stopping or reranking; would be a follow-up
  paper.)
- Not yet a final head-selection method — per-head analysis shows where signal
  lives, but a fair training-free head-selection detector is still open.
- Not multi-pass — single forward pass per image.
- Not a black-box perturbation detector — the model-agnostic extension uses a
  white-box verifier, rather than perturbing the commercial generator.

---

## 14. Open questions to validate empirically

These questions must now be answered under position-controlled evaluation, not
only global AUROC:

1. Which scores retain signal under within-bin or matched-pair AUROC?
2. Does no-RoPE improve position-controlled performance, not just global AUROC?
3. Does a local non-object null outperform the instruction null after
   controlling for `gen_pos`?
4. Can PC-CVG or local delta-concentration beat the position-only baseline?
5. Which layers contribute after position control? Current evidence points to
   no-RoPE layer 1 concentration and CLC, while global layer 0 concentration is
   likely position-confounded.
6. Is top-mass helpful after position control, or only under global AUROC?
7. Do CVG, Concentration, and CLC remain complementary after residualizing
   position?
8. Can a training-free late-layer head subset match the supervised per-head
   probe without using labels at test time?
9. Do mitigation methods improve grounding or merely shift the yes/no answer
   prior on POPE-style benchmarks?
10. When a queried object is absent, does attention concentrate on
    semantically related objects or scene context more often in false positives
    than in true negatives?

Answer pattern that would support the design:

- A score has within-bin or matched-pair AUROC clearly above 0.5, not merely
  high global AUROC.
- The score's advantage over `gen_pos` remains after residualization.
- Local-null CVG or local delta-concentration outperforms instruction-null
  CVG, showing that same-position baselines matter.
- CLC or no-RoPE concentration keeps modest but stable signal across position
  bins.

If any of these fail, the design needs revision (see §15).

---

## 15. Likely failure modes and revision paths

- **Instruction null is too close to object null.** Mitigation: switch to
  noise-image null. Cost: small one-time calibration pass; clean theoretical
  story.
- **Instruction null is position-mismatched.** The instruction tokens occur
  before generation, while object tokens can appear much later. This can make
  CVG reflect query position and context length more than grounding.
  Mitigation: prefer local non-object nulls near the object token and report
  PC-CVG.
- **Global AUROC is position-confounded.** Observed in the current row-cache
  run: `gen_pos` alone reaches AUROC 0.8304, and the strongest layer-0
  concentration scores collapse to roughly random within position bins.
  Mitigation: make within-bin, matched-pair, and residual AUROC mandatory.
- **No-RoPE does not remove generation-position bias.** Pre-RoPE attention
  removes the rotary component from attention weights but not the decoding
  prior that later objects hallucinate more often. Mitigation: combine no-RoPE
  with local-position-controlled contrasts.
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
- **Associated-evidence grounding can masquerade as visual grounding.** A
  method may attend to a real object or scene region that is correlated with the
  queried object, such as toilet evidence for a sink query, and still answer
  incorrectly. This failure mode is especially dangerous for both detection and
  mitigation because the attention map looks visually meaningful. Mitigation:
  add semantic-neighbor controls, absent-object POPE slices with co-occurring
  objects, and TP/FP attention audits that distinguish target evidence from
  merely related evidence.
- **Mitigation-side critique overreaches.** If head-selection or
  visual-enhancement methods improve TPR without increasing FPR, the story
  should not be "mitigation methods are flawed." It should be "aggregate POPE
  metrics need answer-prior controls, and some methods pass them while others
  do not."

---

## 16. Current empirical notes

Current row-cache run:

    experiments/coco_llava_7b_rows
    objects: 16426
    hallucinated: 4009
    layers cached: [0, 1, 2, 3, 4]

Global AUROC findings:

- Best global score: `sink_only_conc_top10_mass_layer_0` /
  `purified_conc_top10_mass_layer_0`, AUROC 0.8099.
- Best CVG-uniform: `sink_only_cvg_kl_uniform_layer_0`, AUROC 0.7954.
- Best no-RoPE instruction CVG: `no_rope_purified_cvg_jsd_instr_layer_4`,
  AUROC 0.7246.

Position-controlled findings:

- `gen_pos` alone: AUROC 0.8304.
- `sink_only_conc_top10_mass_layer_0`: same-bin AUROC 0.5087; cross-bin AUROC
  0.8267.
- Best within-bin scores found so far:
  - `no_rope_topmass_only_conc_top1_mass_layer_1`: weighted within-bin AUROC
    0.5751.
  - `clc_gen_jsd`: weighted within-bin AUROC 0.5670.
  - `no_rope_topmass_only_cvg_jsd_local_nonobj_layer_2`: weighted within-bin
    AUROC 0.5565.

Interpretation: high global AUROC from early-layer concentration and
uniform-null CVG is not yet evidence of visual grounding detection. The most
promising direction is no-RoPE plus local position-controlled contrasts, with
CLC as a complementary weak signal.

### 16.1 Per-head diagnostic: the signal is washed out, not absent

The mean-over-heads shape scores above collapse to chance under position
control. To test whether this is because (a) attention carries no
position-independent grounding signal, or (b) it does but head/layer averaging
destroys it, we cached **per-head** attention-shape features at layers
`[0, 1, 10, 22, 31]` (script `detection/scripts/cache_per_head_rows.py`) and ran a
position-controlled probe (`detection/scripts/diagnose_per_head.py`). Per head we take
the L1-normalized visual-attention row and compute four shape features:
entropy, `-top1_mass`, `-top5_mass`, `-max_over_mean`. Sinks are **not**
stripped (the question here is head granularity, not sinks).

Results on LLaVA-1.5-7B / MSCOCO (same 16426 mentions; IC reference within-bin
0.686; mean-over-heads SinkDetect within-bin ~0.50):

- **Single feature** (best of layer × head × feature): within-bin AUROC 0.595
  (`layer 31, head 9`). Above PAS/SVAR (~0.58) but below 0.60 — no single head
  suffices.
- **Multivariate linear probe** (logistic regression, all per-head features,
  5-fold CV): within-bin AUROC 0.672; stable across L2 ∈ [0.1, 50] (0.664–0.672),
  so not an overfitting artifact.
- **Signal increases monotonically with depth.** Per-layer probe within-bin
  AUROC: layer 1 → 0.523, layer 0 → 0.603, layer 10 → 0.645, layer 22 → 0.690,
  **layer 31 → 0.730**. Layer 31 alone (0.730) beats the full 5-layer
  combination (0.672): early layers are noise that dilutes the probe.
- **No image-level leakage.** Grouping the 5-fold CV by `image_id` (4977 images,
  3.3 mentions/image) leaves the numbers essentially unchanged: layer 31
  within-bin 0.7313 (vs 0.7300 mention-split); all-layers 0.6704.

**Conclusion.** Position-independent grounding signal *is present* in attention
but lives in **individual late-layer heads** and is destroyed by mean-over-heads
aggregation and by mixing in early layers. The culprit in SinkDetect's negative
result is the readout (head averaging + early/all layers), not attention per se.

Caveat for the paper: the 0.73 probe is **supervised** whereas IC/PAS/SVAR are
training-free, so "beats IC (0.686)" is not yet an apples-to-apples claim. The
*scientific* claim (signal exists, deepens with layer, is washed out by
averaging) is established by the within-probe layer comparison; a fair
detector comparison requires a training-free variant (§16.2) or giving IC the
same probe treatment.

### 16.2 LH-Shape: late-layer per-head attention-shape detector (proposed)

Motivated by §16.1, the constructive method for the paper. **LH-Shape** scores
the *shape* of each late-layer head's visual-attention distribution and
aggregates, rather than averaging heads first.

- **Features.** For the object query at the last layer (and optionally a small
  late-layer set, e.g. `{22, 31}`), per head `h`: the L1-normalized visual row's
  entropy, top-`k` mass, and `max_over_mean` (the §16.1 features). Sink removal
  optional as an ablation.
- **Supervised variant (signal ceiling).** Linear probe over the per-head
  feature vector; report position-controlled within-bin / matched-pair /
  residual AUROC. Establishes the upper bound (~0.73 within-bin) and the
  depth/aggregation story.
- **Training-free variant (fair baseline comparison).** No per-mention labels
  used at scoring time. Options to evaluate: (i) mean over heads of a single
  shape feature at layer 31 (tests whether *late-layer* mean already beats
  *all-layer* mean); (ii) unsupervised head weighting (e.g. weight heads by
  dispersion/peakedness consistency); (iii) a fixed head subset selected on a
  disjoint split. This is what gets compared head-to-head with PAS, SVAR, IC,
  GLSim under the position-controlled protocol.
- **Why it should help.** It keeps the per-head, late-layer information that the
  diagnostic shows is discriminative, instead of collapsing it by averaging.

**Open items before this is a paper claim:**

1. Build the training-free variant and compare fairly to IC (no supervision
   advantage).
2. Replicate the "late-layer per-head recovers signal" trend on ≥1 more LVLM
   (InstructBLIP / Qwen-VL) and ≥1 more dataset (Objects365).
3. Sweep which late layers / how many heads are needed (cost vs signal).
4. Test sink removal and no-RoPE as ablations on the per-head features.

Reproduce: `detection/scripts/cache_per_head_rows.py` (GPU, dumps per-head shape
features) then `detection/scripts/diagnose_per_head.py` (CPU, position-controlled probe).

---

## 17. References (from memory; verify before submission)

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
- Bai et al., *Mitigating Hallucinations in Large Vision-Language Models by
  Adaptively Constraining Information Flow (AdaVIB)*, arXiv 2025.
- Chuang et al., *DoLa*, ICLR 2024.
- Abnar & Zuidema, *Attention Rollout*, 2020.

A thorough literature review and citation-checking pass is still to do.
