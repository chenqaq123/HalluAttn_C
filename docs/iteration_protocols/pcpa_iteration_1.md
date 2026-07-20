# PCPA Iteration 1 Protocol: Common-Mode-Free Pairwise Alignment

Date: 2026-07-20
Status: cancelled before start on 2026-07-20; retained as a historical record

> Superseded by the training-free position-free detector audit. No manifest,
> reference-margin cache, LoRA adapter, or trained checkpoint was produced. The
> active protocol is
> [position_free_detector_iteration_1.md](position_free_detector_iteration_1.md).

## Question

Can Qwen3-VL learn to distinguish a visible object from a semantically related
but absent object when both questions refer to the exact same natural image,
without obtaining the gain from a global `Yes` or `No` shift?

This iteration trains a usable mitigation model. It does not use attention,
image edits, an auxiliary detector, or a second inference distribution.

## Frozen model and inference contract

- Base/reference model: `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`.
- Runtime: Transformers 4.57.6, bf16, eager attention, deterministic
  200704-pixel image preprocessing.
- Prompt: the existing direct POPE existence template.
- Answers used for training scores: exact `Yes` and `No` strings under the
  model chat template, excluding EOS. Record their tokenization before training.
- Freeze the vision encoder, multimodal projector, embedding table, and LM head.
- Train one LoRA adapter on decoder `q/k/v/o` and MLP `gate/up/down` projections.
- At evaluation, merge or load only that adapter and generate normally. No
  reference model, paired query, candidate set, score correction, or threshold
  is present at inference.

## Pair construction

Use COCO train2017 instance annotations only. Convert annotations to image-level
category presence and compute positive NPMI for every category pair. A relation
is eligible only with at least 50 co-occurring images.

For one image `I`, a directed pair contains:

- `t+`: an annotated visible category;
- `t-`: an absent category for which `t+` is an eligible positive-NPMI neighbor;
- `q+`: `Is there a <t+> in the image?`;
- `q-`: `Is there a <t-> in the image?`.

Choose the highest-NPMI eligible visible neighbor, breaking ties by COCO category
ID and image ID. Boxes, masks, crops, detector scores, captions, and model
outputs are forbidden. The pair categories are supervision metadata only and
are not inference-time candidates.

Sort COCO categories by ID and reserve every fourth category as 20 held-out
query targets. None of those names may occur as either `t+` or `t-` in training.
Among the remaining 60 categories, solve a deterministic capacity-constrained
assignment so each category appears in the positive and absent role equally up
to two pairs. If this is infeasible, stop before model inference and amend only
the annotation-level sampling counts; do not relax category holdout or role
balance after seeing model behavior.

Freeze image-disjoint manifests before computing model scores:

- `pcpa_train`: 1,200 pairs, 20 absent-role pairs per training category;
- `pcpa_dev`: 300 pairs, 5 absent-role pairs per training category;
- `pcpa_transfer_pair`: 400 pairs, 20 per held-out absent target, with `t+`
  chosen from the 60 training categories;
- `pcpa_transfer_binary`: 800 single queries, 40 per held-out target: 20
  present, 10 related absent, and 10 random absent.

Use at most one pair or query per image across all four manifests. The transfer
binary manifest tests held-out targets in both answer directions; the pair
manifest tests related-absence transfer while holding every pixel fixed.

## Pairwise objective

For answer string `a`, let `log pi_theta(a | q, I)` be its teacher-forced token
log-probability sum. Define

    m_theta(t, I) = log pi_theta(Yes | q_t, I)
                    - log pi_theta(No | q_t, I).

Cache `m_0` from the frozen reference model. For each pair:

    Delta+ = m_theta(t+, I) - m_0(t+, I)
    Delta- = m_theta(t-, I) - m_0(t-, I)
    d_pair = Delta+ - Delta-
    c_pair = (Delta+ + Delta-) / 2.

The preferred joint outcome is `(Yes to q+, No to q-)`; the rejected outcome is
`(No to q+, Yes to q-)`. Their reference-centered joint log-ratio is `d_pair`.
Use

    L_pair = softplus(-beta * d_pair)
    L_center = c_pair^2
    L_anchor = softplus(gamma - m_theta(t+, I))
             + softplus(gamma + m_theta(t-, I))
    L_PCPA = L_pair + lambda_c * L_center + lambda_a * L_anchor.

Freeze `beta=0.1`, `gamma=0.5`, `lambda_c=0.1`, and `lambda_a=0.1` before the
transfer manifests are opened. `L_center` is the method-defining constraint: it
requires separation to come from opposite target-specific updates rather than a
common answer-prior movement. `L_anchor` prevents both final margins from
remaining on the same side of zero.

## Matched training arms

Train all arms on the same pairs, LoRA modules, optimizer, steps, and seed:

1. **Frozen base:** no training.
2. **Balanced answer SFT:**
   `L_SFT = -(log pi(Yes|q+,I) + log pi(No|q-,I)) / 2`.
3. **Independent DPO:** apply standard reference-centered DPO independently to
   `Yes>No` on `q+` and `No>Yes` on `q-`.
4. **Anchor-matched independent DPO:** arm 3 plus `lambda_a * L_anchor`.
5. **Joint pair without centering:** `L_pair + lambda_a * L_anchor`.
6. **Full PCPA:** `L_pair + lambda_c * L_center + lambda_a * L_anchor`.

Arms 3 versus 5 isolate independent from joint preference construction; arms 5
versus 6 isolate common-mode centering; arms 2 and 4 test whether absolute
answer supervision or anchoring already explains the gain.

## Optimization

- Seed: 17.
- LoRA rank 16, alpha 32, dropout 0.05.
- AdamW, learning rate `5e-5`, weight decay 0, cosine schedule, warmup ratio
  0.03, gradient clipping 1.0.
- Pair batch size 8, gradient accumulation 4, effective batch size 32.
- Train for at most 3 epochs. Evaluate every 100 optimizer steps.
- Select one checkpoint per arm by `pcpa_dev` pair accuracy, breaking ties by
  lower related-absent FPR and then earlier step.

Do not tune learning rate, rank, pair count, loss weights, or epochs from
transfer or POPE outcomes. A software or OOM amendment must be recorded before
any affected arm's evaluation is inspected and applied identically to all arms.

## Evaluation

Open transfer manifests only after all selected checkpoints and reference
margins are frozen. Report for every arm:

- same-image pair accuracy `m(t+,I) > m(t-,I)`, overall and by held-out `t-`;
- binary TPR, FPR, related-absent FPR, random-absent FPR, balanced accuracy,
  MCC, accuracy, yes-rate, and per-category dispersion;
- `mean(Delta+)`, `mean(Delta-)`, separation update, common-mode update, and
  the fraction of pairs where both margins have the same sign;
- POPE-popular official metrics and net answer changes relative to the frozen
  model, with changes split into corrections and new errors;
- image-grouped paired-bootstrap 95% intervals for all between-arm primary
  differences.

Run two destructive controls on the frozen transfer checkpoints:

1. remove the image using the model's supported text-only conversation path;
2. permute images across pairs within absent-target strata while retaining
   questions and labels.

If text-only input is unsupported, replace it with one fixed mid-gray image and
record the substitution before inspecting results. Neither control is used for
training or checkpoint selection.

## Advance/kill gate

PCPA advances only if all conditions hold:

1. it beats both balanced SFT and independent DPO by at least 0.03 transfer-pair
   accuracy, with image-grouped paired-bootstrap intervals above zero;
2. versus the frozen model, held-out related-absent FPR falls by at least 0.03,
   TPR falls by no more than 0.005, and both MCC and balanced accuracy increase;
3. on POPE-popular it makes at least ten net corrections, increases MCC and
   balanced accuracy, and changes yes-rate by at most 0.01;
4. `abs(mean(c_pair)) <= 0.1 * mean(abs(Delta+ - Delta-))`, and arm 5 violates
   this bound or is at least 0.03 worse in pair accuracy or related-absent FPR;
5. image removal or permutation eliminates at least half of PCPA's pair-accuracy
   gain over the frozen model;
6. no more than five of twenty held-out absent targets are below frozen pair
   accuracy, and no target loses more than 0.10 absolute TPR.

Failure closes this exact route. Do not rescue it by changing the category
split, mining model-dependent pairs, adding answer thresholds, or opening
POPE-random. If it passes, append all results and pre-register confirmation on
POPE-random plus LVIS-on-COCO before scaling to another model.

## Required reporting

Commit manifests and implementation before training. Append every arm and
destructive-control result to `docs/experiment_results.md`, including failures.
Append the route decision to `docs/iteration_log.md`. Checkpoints, cached
reference margins, raw predictions, and bootstrap artifacts remain in ignored
experiment storage.
