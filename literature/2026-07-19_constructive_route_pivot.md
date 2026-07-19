# Constructive Route Pivot: From Self-Verification to Target Selectivity

Date: 2026-07-19

## Question

After XEV's full inference-time route failed, what constructive direction can
still fit the paper's thesis, avoid repeating prior work, and produce a useful
Qwen3-VL method?

The route audit used three filters:

1. the method must introduce supervision or evidence that is not another vote
   from the same LVLM over an attention-selected, grounded, or cropped view;
2. it must address the related-present failure mode rather than only generic
   visual reliance or aggregate yes/no calibration;
3. it must have a clean ablation against the nearest published method.

## Route-level conclusion from our experiments

XEV is not a low-threshold method waiting for a better candidate set. It is a
closed information loop. Attention proposals, native coordinates, local crops,
target-free inventories, pairwise prompts, and scale replication all feed image
variants back into the same Qwen3-VL semantic judge. The model's co-occurrence
prior and category boundary therefore remain shared across every stage.

The nominal 960-row XEV-v1 result changed four decisions and removed two net
errors. Its coverage expansions did not generalize, and the final untouched
240-row window produced zero corrections and one new error. Treating this as an
effective constructive method is not defensible.

## Literature map

### Same-model self-verification: reject as our next route

- **R-CoV: Region-Aware Chain-of-Verification for Alleviating Object
  Hallucinations in LVLMs** (Xie et al., 2026),
  <https://arxiv.org/abs/2604.20696>. It generates an initial answer, extracts
  entities, asks the LVLM for coordinates and region descriptions, verifies the
  descriptions, and revises the answer. This is the published version of the
  general route our XEV iterations stress-tested. Its reported POPE protocol
  samples only 300 questions per split, whereas our failure appears when the
  rule is frozen and exercised over full disjoint windows.
- **Logical Closed Loop / LogicCheckGPT** (Wu et al., 2024),
  <https://arxiv.org/abs/2402.11622>. It uses object-to-attribute and
  attribute-to-object consistency from the same LVLM. It does not provide an
  independent visual identity boundary.

Conclusion: another prompt, crop, coordinate format, or voting rule would be an
incremental reproduction of an occupied route and would not address our measured
failure.

### External expert verification: useful control, weak contribution

- **Pelican** (Sahu et al., EMNLP 2024),
  <https://aclanthology.org/2024.emnlp-main.470/>. It decomposes claims and
  composes external tools for verification.
- **Look on Demand / CSMR** (Zhang et al., 2026),
  <https://arxiv.org/abs/2605.28160>. It schedules calls to an independent visual
  perception module when reasoning needs evidence.

Our previous `main`-branch OWLv2 positive control already established that an
external region detector can improve POPE more than the internal verifier. A new
detector gate would likely work, but the contribution would be vulnerable to
"why not just use the detector?" and overlaps established tool pipelines.

Conclusion: keep independent perception as a positive control, not the new
headline method.

### Generic attention/feature intervention: crowded and story-inconsistent

- **CausalLens** (Ji et al., CVPR 2026),
  <https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html>.
  It identifies visually reliable heads by sensitivity and steers hidden states.
- **Locate-then-Sparsify** (Dang et al., CVPR 2026),
  <https://openaccess.thecvf.com/content/CVPR2026/html/Dang_Locate-then-Sparsify_Attribution_Guided_Sparse_Strategy_for_Visual_Hallucination_Mitigation_CVPR_2026_paper.html>.
  It uses causal layer attribution to drive sparse feature steering.
- **One Token, Two Fates** (Fa et al., CVPR 2026),
  <https://openaccess.thecvf.com/content/CVPR2026/html/Fa_One_Token_Two_Fates_A_Unified_Framework_via_Vision_Token_CVPR_2026_paper.html>.
  It augments and prunes latent visual tokens for calibration.
- **CIPHER** (Dastmalchi et al., 2026),
  <https://arxiv.org/abs/2603.10470>. It constructs diffusion-edited
  counterfactual images and projects hidden states away from a learned
  hallucination subspace.
- **Causal-LLaVA** (Hu et al., 2025),
  <https://arxiv.org/abs/2505.19474>. It uses category feature dictionaries and
  causal modules to disentangle co-occurrence-biased representations.

Conclusion: a generic causal/feature-steering proposal is no longer a clear
novelty claim, and attention reinforcement conflicts with our mitigation-side
evidence unless it explicitly distinguishes target evidence from related visual
evidence.

### Grounding or target-focused training alone: insufficient boundary

- **Does Object Grounding Really Reduce Hallucination of Large Vision-Language
  Models?** (Geigle et al., EMNLP 2024),
  <https://aclanthology.org/2024.emnlp-main.159/>. Under open-caption evaluation,
  fine-grained grounding objectives have little or no effect on object
  hallucination.
- **TL-DPO** (Yoon et al., CVPR 2025),
  <https://openaccess.thecvf.com/content/CVPR2025/html/Yoon_Stop_Learning_it_all_to_Mitigate_Visual_Hallucination_Focus_on_CVPR_2025_paper.html>.
  It applies preference learning specifically to hallucinated response spans.

Conclusion: target localization or target-span weighting does not by itself
specify which visual change should alter the decision.

### Counterfactual visual learning: closest family and remaining gap

- **oDPO / Can They Still See Removed Objects?** (He et al., NAACL 2025),
  <https://aclanthology.org/2025.naacl-long.349/>. ROHE removes an existing object
  with LaMa; oDPO prefers the original image over the target-removed image.
  Therefore, target removal and object-based preference optimization are already
  prior art.
- **VES-RFT** (Hou et al., CVPR 2026),
  <https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html>.
  It rewards uncertainty change between image and no-image branches. This is
  global visual sensitivity, not target specificity.
- **VISE** (Venkatraman et al., 2026),
  <https://arxiv.org/abs/2606.27373>. It uses geometric invariance and penalizes
  evidence-agnostic generation under predicted-region perturbation, but does not
  pair target removal with invariance to a specific co-occurring distractor.

The unoccupied mechanism-level question is:

> Does the decision react to the queried object's pixels specifically, or to a
> correlated object/context that happens to be visual?

Neither an image/no-image comparison nor original/target-removed preference can
answer the second half. The missing control is a matched `neighbor removed,
target retained` intervention.

## Selected direction

**Target-Causal Selectivity (TCS)** trains a model on a triple:

- original image `I`, containing target `t` and related neighbor `n`;
- `I^{-t}`, target removed and neighbor preserved;
- `I^{-n}`, neighbor removed and target preserved.

For the target existence margin `m(t,I)`:

    target sensitivity: Delta_t = m(t,I) - m(t,I^{-t})
    neighbor leakage:   Delta_n = m(t,I) - m(t,I^{-n})
    selectivity:        TCS = Delta_t - beta * |Delta_n|

Training increases `Delta_t`, suppresses `Delta_n`, and also presents natural
related-present negatives containing `n` but no `t`. At inference, the adapted
Qwen3-VL uses one standard forward pass. No masks, detector, attention map,
candidate list, or verifier is required.

This direction fits the paper because it replaces the failed proxy with an
operational definition of target evidence. Attention allocation and generic
visual dependence can both be high for a related object; target selectivity
requires the correct asymmetric counterfactual behavior.

## Required comparisons

1. Frozen base Qwen3-VL.
2. Binary SFT on exactly the same natural examples.
3. Target-removal-only objective, the nearest oDPO-like mechanism ablation.
4. Full target-sensitivity plus neighbor-invariance objective.
5. Same-area/background removal control for edit-artifact sensitivity.

The method is supported only if the full objective improves related-present FPR
and held-out-category selectivity beyond target-removal-only without obtaining
the gain through a global reduction in yes rate.

## Evaluation boundary

- Development: POPE-popular plus a held-out COCO intervention-pair set.
- Confirmation: POPE-random and MME existence through `lmms-eval`.
- Auxiliary mechanism benchmark: ROHE.
- Closed diagnostic only: POPE-adversarial, because every row has already been
  used during XEV development/confirmation.
- Generalization: at least 20 COCO categories held out as intervention targets;
  no open-vocabulary claim until a category-disjoint result succeeds.

## Main risks

- Inpainting artifacts can become a shortcut. Use natural absent images,
  same-area removal controls, all-instance masks, and removal-quality filters.
- TCS may improve binary existence questions but not free generation. CHAIR or
  AMBER generation is a later stage, not part of the first pilot.
- Qwen3-VL-8B LoRA may exceed two-4090 memory with three image branches. Start
  with cached/frozen vision features or QLoRA and a small mechanism set before
  any full training run.
- The novelty is the paired selectivity objective. If target-removal-only is
  equally good, the proposed mechanism and paper method fail.
