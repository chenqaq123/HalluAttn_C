# EEA Iteration 0 Protocol: Absolute Set Evidence Before Tuning

Date: 2026-07-19
Status: pre-registered; not started

## Question

Do native Qwen3-VL visual states support an absolute, target-conditioned
existence score that separates a visible target from a related absent target,
or does every patch matcher collapse to the same relative scene prior as
attention?

This iteration tests only the mechanism premise of Existential Evidence
Aggregation (EEA). It does not change model answers.

## Frozen model contract

- Model: `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`.
- Runtime: Transformers 4.57.6, bf16, eager attention, deterministic
  200704-pixel image preprocessing.
- Prompt: the existing direct POPE yes/no format.
- Backbone, vision encoder, projector, decoder, and LM head are frozen.
- No LoRA, answer-logit correction, crop, image edit, box/mask supervision,
  external vision model, candidate generation, or second model call is allowed.

Cache visual rows at four predeclared taps: multimodal-projector output and
decoder layers 18, 24, and 30. Decoder visual rows occur before the question in
the causal sequence and therefore cannot read the target. The layer tap is
selected by grouped cross-validation on training categories and frozen before
held-out outcomes are opened.

Form target representation `x_t` by averaging the model's **input embeddings**
over the known target-phrase token span. Do not use a decoder target state,
answer state, or any representation that has already read the image.

## Shared matcher

For tap-specific visual states `v_i`, train one shared matcher:

    u_t = normalize(W_t x_t)
    u_i = normalize(W_v RMS(v_i))
    s_i = <u_t, u_i> / 0.07 + w_b^T u_t + b_0.

`W_t` and `W_v` map to 128 dimensions. All targets share every parameter. The
target-conditioned bias is shared and linear; per-category biases or thresholds
are forbidden.

The EEA logit is

    E_max = 0.1 * logmeanexp_i(s_i / 0.1).

The `logmeanexp` normalization makes the score exactly invariant to duplicating
the complete patch set. Verify this numerically to `1e-6` on 100 cached rows.

## Matched aggregation controls

Train each aggregator independently with the same `W_t`, `W_v`, target bias,
optimizer, and parameter count:

1. **EEA smooth max:** `E_max` above.
2. **Relative softmax pool:** `E_soft = sum_i softmax_i(s_i) * s_i`.
3. **Global mean pool:** `E_mean = mean_i(s_i)`.
4. **Top-attention patch:** select one native visual patch using Qwen3-VL's
   final target-phrase token row, averaged over heads and layers 18--29, then
   use that patch's `s_i`. Attention chooses only the patch; its magnitude is
   not a feature.

Also report non-matched diagnostics:

- query-only logistic regression on `x_t`;
- image-only logistic regression on the global visual mean;
- frozen vanilla first-token Yes/No margin;
- native maximum pre-RoPE attention logit and post-RoPE attention mass.

No aggregator may receive COCO category IDs, labels of visible neighbors, or
bounding boxes.

## Natural-image manifests

Use COCO train2017 instance annotations only. Sort categories by category ID and
hold out every fourth category as a queried target: 20 held-out targets and 60
training targets. A held-out object may appear as scene context but is never a
queried training label.

Compute positive NPMI neighbors from train2017 image-level category presence.
Require at least 50 co-occurring images and retain up to five neighbors per
target. For each target define:

1. `present`: target annotated in the image;
2. `related_absent`: target absent and at least one retained neighbor present;
3. `random_absent`: target and all retained neighbors absent.

Freeze two image-disjoint manifests before model inference:

- `matcher_train`: 1,200 rows, 20 per training target, with 10 present, 5
  related-absent, and 5 random-absent;
- `matcher_transfer`: 800 rows, 40 per held-out target, with 20 present, 10
  related-absent, and 10 random-absent.

Sample at most one queried target per image in each manifest. Store image ID,
target phrase/token span, label, subtype, visible category IDs, and prompt in
the committed manifest. Local image paths stay in ignored storage.

## Same-image target swaps

Every related-absent transfer row contains at least one visible NPMI neighbor.
For each such image, form a pair without rerunning the image encoder:

- absent query: the held-out target `t_abs`;
- present query: the annotated neighbor `t_vis` with highest positive NPMI.

The full patch set is identical. The registered success event is

    E(t_vis, I) > E(t_abs, I).

Report pair accuracy overall and by absent target. This control directly tests
target identity rather than generic scene or image quality.

## Training and freezing

Use seed 17, AdamW, learning rate `1e-3`, no weight decay, batch size 128, and
balanced binary cross-entropy. Run image-grouped five-fold cross-validation on
`matcher_train` for at most 100 epochs with patience 10. Choose the tap and epoch
from mean category-macro present-versus-related-absent AUROC, then freeze them.

All four matched aggregators use identical folds. Open `matcher_transfer` only
after the selected tap, epoch, and all model states are saved. There is no
held-out threshold fitting and no second seed unless the registered run fails
for a documented software error before outcomes are inspected.

## Controls on the held-out transfer set

Report category-macro AUROC for present versus related absent as primary. Also
report present versus random absent, balanced accuracy at the training-only
threshold, per-category AUROC, score distributions, and image-grouped paired
bootstrap intervals.

Run two destructive controls with the frozen EEA head:

1. exchange images between present and related-absent rows within held-out
   target category, recompute visual states, and retain the original labels;
2. permute patch states across images within target category while retaining
   target embeddings and labels.

The first is the registered image-swap gate. The second distinguishes real
target-patch matching from a target-only bias plus aggregate score calibration.

## Advance/kill gate

EEA advances only if all conditions hold on `matcher_transfer`:

1. present-versus-related-absent category-macro AUROC is at least 0.75;
2. EEA exceeds each of relative softmax, global mean, and top-attention-patch
   pooling by at least 0.05 AUROC, with image-grouped paired-bootstrap 95%
   intervals above zero;
3. same-image target-swap pair accuracy is at least 0.65;
4. image exchange lowers EEA AUROC by at least 0.15 and patch permutation lowers
   it by at least 0.10;
5. no more than five of the twenty held-out targets have AUROC below 0.60.

Failure of any condition closes EEA. Do not add box supervision, a CLIP model,
class-specific calibration, more layers, or LoRA to rescue Stage A.

If all conditions pass, append the result and pre-register a new behavior
iteration. That later protocol must compare answer-only LoRA, EEA-LoRA, and
parameter-matched relative/global auxiliary losses. No behavior training is
authorized by this protocol.

## Required reporting

Commit manifests and implementation before inference. Append all Stage-A
results, including failed controls and per-category dispersion, to
`docs/experiment_results.md`; append the route decision to
`docs/iteration_log.md`. Cached states, weights, raw predictions, and bootstrap
artifacts remain under ignored experiment storage.
