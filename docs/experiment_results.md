# Experiment Results — Consolidated Log

> Append-only. Every new experimental result gets a dated entry at the bottom of
> section E. Do not delete or rewrite past numbers; if a result is superseded,
> add a new entry that says so. Sign convention for detection scores:
> **higher ⇒ more likely hallucination**.
>
> **Restart baseline: 2026-07-07.** On the `restart-two-pronged` branch this log
> was reset to hold only the reproducible baseline evidence for the two-pronged
> critique (§A detection baselines, §B detection shape/per-head reproduction,
> §C mitigation baseline behavior). All prior TDEV / semantic-neighbor /
> no-external-detector result entries were removed here; they remain on `main`.
> New results for the rebuilt method are appended under §E.

## Standing setup

- Model: LLaVA-1.5-7B (HuggingFace), fp16, greedy decoding.
- Detection data: COCO val2014 captions, CHAIR object labels.
  - object mentions: 16426 (hallucinated 4009 / non-hallucinated 12417).
  - **position-only AUROC (gen_pos): 0.8304** — the confound baseline to beat.
- Mitigation data: POPE (random/popular/adversarial, 3000 each = 9000 rows) and
  CHAIR caption regeneration on a fixed COCO manifest.
- Length/position control (detection): within-bin, matched-pair, and residual
  AUROC over generation-position bins. Definitions in [design.md](design.md).

---

## A. Detection — baselines under position control (Prong 1)

Source: `detection/baselines/results/coco_llava_7b_baselines/`,
[detection/docs/results_summary.md](../detection/docs/results_summary.md).

| Score | Overall | Within-bin | Matched-pair | Residual | Note |
|---|---:|---:|---:|---:|---|
| PAS | 0.835 | 0.593 | 0.583 | 0.569 | high overall, almost entirely position-correlated |
| SVAR (fixed, layers 5–18) | 0.805 | 0.568 | 0.566 | 0.548 | faithful middle-layer SVAR; **still collapses under length control** |
| IC | 0.776 | 0.686 | 0.703 | 0.633 | strongest *unsupervised attention/logit* controlled baseline |
| GLSim-local | 0.772 | 0.564 | 0.593 | 0.601 | some residual signal |
| Entropy | 0.721 | 0.637 | 0.655 | 0.634 | reliable uncertainty baseline |
| NLL | 0.711 | 0.636 | 0.652 | 0.630 | ~Entropy |
| **TVER** (ICCV'25 ONLY, layer 0) | 0.830 | 0.566 | 0.544 | 0.519 | H(text attn)/H(vis attn); unsup; **overall ≈ position-only 0.830**, collapses |
| **SAE — true** (Mask2Former, best L10) | 0.783 | 0.575 | 0.579 | 0.529 | 2026 unsup attention×segmentation; reproduces overall≈paper 0.766, **collapses** |
| SAE — seg-free skeleton (best L16) | 0.792 | 0.608 | 0.589 | 0.539 | dispersion proxy, **not** the paper's SAE (no segmenter) |
| Beyond ADS (1-layer, L16) | 0.688 | 0.555 | 0.559 | 0.538 | attention-dispersion; 1-layer is *diagnostic only* (paper's ADS = supervised vector) |
| Beyond CGC (faithful, best L31) | 0.728 | 0.574 | 0.605 | 0.600 | repr grounding; **collapses** — old 0.872/0.752 was a `token_pos+1` indexing bug |
| **LaSCD** (norm shape-energy, mean L11–28) | 0.559 | 0.522 | — | — | attention 2D-Laplacian shape; near-chance (best layer 0.589/0.569). Raw energy is a *mass proxy* (anti-correlated); see note. |
| ADS-only **supervised** (32-layer OOF) | 0.840 | 0.664 | 0.657 | 0.619 | multi-layer *attention* pooling partly survives |
| CGC-only **supervised** (32-layer OOF) | 0.837 | 0.669 | 0.676 | 0.656 | multi-layer *repr* pooling |
| Beyond ADS+CGC **supervised** (OOF) | 0.866 | 0.719 | 0.724 | 0.674 | suite ceiling; barely exceeds single-score IC (0.686) |

Takeaway (v3, all faithful to official repos — see §E 2026-07-08 repo-audit entry):
**every unsupervised *single-score attention* detector collapses under length
control** — PAS, SVAR, ADS, **the faithfully-reproduced 2026 SAE (with its real
Mask2Former segmenter)**, **LaSCD (near-chance, see note)**, and the newest
**TVER (ICCV 2025)**, whose overall AUROC (0.830) sits *exactly on the position-only
baseline* (0.8304).

> **LaSCD is a special case (reviewed against `macovaseas/LaSCD` + paper):** its
> energy is a *within-token cross-layer selector* (argmax=premature/argmin=GT layer,
> feeding contrastive decoding), never a cross-token detector — the repo/paper report
> **no per-sample AUROC**. The implemented *raw* energy is mass-weighted (Eq. 3), so
> cross-token it degenerates into a visual-mass proxy that *anti-correlates* with
> hallucination (grounded > hallu, overall < 0.5); Theorem 1 (fixed mass) is about the
> mass-*normalized* shape energy, which has the right sign but is near-chance (mean
> L11–28 0.559/0.522, best layer 0.589/0.569). So LaSCD's weakness is an **off-label
> mismatch, not a reproduction bug** — our `lascd_raw` reproduces `_deco_layer_energy`
> verbatim.

The earlier "CGC survives (0.752)" was an indexing artifact
(reading the object token's own hidden state at `token_pos+1` instead of the
generating row `token_pos` the official code uses); the faithful CGC also collapses
(0.574). What retains moderate within-bin signal is **not any single attention
readout**: the single logit-lens **IC** (0.686) and **supervised 32-layer pooling**
of ADS/CGC (0.664/0.669), ceiling at supervised **ADS+CGC 0.719**.

> **Reproduction fidelity (verified 2026-07-08 against official GitHub repos, see §E):**
> ADS/CGC diffed vs `tuandung2812alt3/token-grounding-detector` (CVPR/ICLR 2026):
> ADS math faithful, single-layer ~0.5 is *by design* (paper's ADS is the supervised
> per-layer vector; our supervised overall 0.865 ≈ paper 0.88). CGC had a one-off
> position bug, now fixed to the repo's generating-row readout. LaSCD `lascd_raw`
> reproduces the official `_deco_layer_energy` verbatim (`macovaseas/LaSCD`). SAE has
> no repo; the **true** version adds Mask2Former-Swin-L segmentation-region entropy
> per the paper. TVER reproduces `zifuwan/ONLY` layer-0 `H(text)/H(vis)` exactly.

## B. Detection — SinkDetect shape scores & per-head diagnostic (Prong 1)

Source: `experiments/coco_llava_7b_rows`, [design.md §16](design.md).

- Mean-over-head shape scores collapse under position control (best global
  `sink_only_conc_top10_mass_layer_0` = 0.8099 overall but same-bin **0.5087**).
- Best within-bin shape scores are weak: `no_rope_topmass_only_conc_top1_mass_layer_1`
  0.5751; `clc_gen_jsd` 0.5670.
- **Per-head supervised probe** (logistic, position-controlled, image_id-grouped
  CV, no leakage): within-bin AUROC rises monotonically with depth —
  layer 1 → 0.523, layer 0 → 0.603, layer 10 → 0.645, layer 22 → 0.690,
  **layer 31 → 0.730**. Layer 31 alone beats the all-layer combination (0.672).
- Conclusion: grounding signal exists but lives in late-layer *individual heads*
  and is washed out by mean-over-head aggregation.
- ⚠️ Positioning: this is **supervised** and **largely pre-empted by HaloProbe**
  (per-head + confidence probe, 93.5 AUROC). Report as background, not headline.

## C. Mitigation — attention interventions (POPE/CHAIR behavior) (Prong 2)

Scope: PAI (attention-only, no CFG branch), ClearSight VAF, Visual Attention
Sink. Vanilla greedy is the anchor.

- **PAI-attention-only:** essentially no benefit; slight POPE acc/F1 drop, no
  CHAIR reduction. (Attention-only reweight is inert — not a backfire.)
- **ClearSight:** POPE yes-rate +~3.6pts; TPR +~3.5 but FPR +~3.8 ⇒ net
  ΔTPR−ΔFPR negative; adversarial accuracy drops; Macro MCC 0.730 → 0.723.
- **Visual Attention Sink:** mild yes-rate/FPR increase on POPE; on CHAIR
  longer captions, more object mentions, more hallucinated mentions.

Takeaway: attention *amplification* changes routing/output style and raises the
yes-rate without improving target-object verification — the Prong-2 reproduction.
The related-present vs. plain-absent FPR decomposition that makes this measurable
is part of the rebuild backlog (was on `main`; to be re-derived here).

> **Reproduction fidelity (verified 2026-07-08, see §E):** the amplification
> family that carries Prong 2 — PAI (faithful, inert), ClearSight and VisAttnSink
> (faithful; both raise yes-rate/FPR, drop MCC) — is sound, and VCD is faithful.
> Caveats: ClearSight's POPE suppression collapses to ~1 token (enhancement-only
> on POPE). **SPIN and DAMRO were buggy; both were fixed 2026-07-08 and now need a
> re-run** (their old smoke numbers are invalid). **OPERA was dropped** (fork-only,
> not reproducible in this stack). Full detail in §E.

---

## E. New results (append below, newest last)

<!-- Template for each new entry:
### YYYY-MM-DD — <short title>
- Setup: model / data / script / commit
- Result: key numbers
- Interpretation: 1-3 sentences, tie back to proposal claim
- Supersedes: (if applicable)
-->

### 2026-07-08 — Baseline reproduction verification against source papers

- Setup: audited every retained detection and mitigation baseline against its
  source paper / official code (6 parallel reviews). Verified our code in
  `detection/baselines/methods/extract_model_signals.py` and
  `mitigation/src/{interventions,decoding}.py`. Reproduced numbers read from
  `mitigation/results/coco_llava_7b_attention_only/audit_pope_macro.csv`,
  `audit_chair.csv`, and `pope_full_vcd_greedy_audit/audit.json`.

- **Verdicts (fidelity of the port to the paper):**

| Baseline | Paper / code | Verdict | Note |
|---|---|---|---|
| NLL, Entropy | standard | ✅ faithful | correct teacher-forcing alignment (hard-abort guard) |
| IC | arXiv:2410.02762 | ✅ faithful (conservative) | final-layer logit-lens only vs paper's max-over-layers → within-bin 0.686 is a **lower bound** |
| GLSim | arXiv:2508.19972 | ✅ faithful | formula + LLaVA-1.5 hyperparams (layers 31/32, k=32, w=0.6) match |
| PAS | arXiv:2511.11502 (`lanl/pas`) | ✅ faithful | matches official line-for-line; **the 0.835→0.593 collapse is a genuine property of PAS, not a bug** |
| SVAR | arXiv:2411.16724 | ⚠️ relabel | ours is **layer-0** visual-attention mass; paper's SVAR sums middle layers 5–18. Mechanism + position-confound conclusion hold, but the row is not the paper's SVAR — describe as "layer-0 visual attention mass" or fix. |
| Beyond-ADS/CGC/ADS+CGC | arXiv:2604.04863 | ⚠️ simplified proxy | single-layer, unsupervised; drops blob-mass (ADS), selects top-k by attention not similarity (CGC), fixed 0.5/0.5 avg vs paper's supervised 2L classifier. Near-random 0.503 reflects the proxy, **not** the original method. Already tagged `paper_reimplementation`. |
| PAI (attn-only) | ECCV 2024 (`LALBJ/PAI`) | ✅ faithful-but-scoped | CFG/logit branch intentionally dropped (documented); attention half matches official exactly |
| ClearSight/VAF | CVPR 2025 (`ustc-hyin/ClearSight`) | ⚠️ POPE caveat | enhancement half faithful to official code; **on POPE the suppression region collapses to ~1 token** (POPE prompt has no system block) → effectively enhancement-only; no visual-head selection (inherited from official code); suppress=0.95 (code) not 0.9 (paper) |
| VisAttnSink | ICLR 2025 (`seilk/VisAttnSink`) | ✅ faithful | sink dims 2533/1415, τ=20, ρ=0.5, retain p=0.6 all match; minor layer window 2–31 vs 0–30 |
| VCD | CVPR 2024 (`DAMO-NLP-SG/VCD`) | ✅ faithful-but-scoped | contrast formula + β-constraint + diffusion noise bit-faithful; greedy (not sampling) and α=0.5 are documented choices |
| DAMRO | EMNLP 2024 | ❌ likely bug | outlier selection at `decoding.py:285` collapses CLIP's 16 heads into one full-dim head scaled by 1024^-0.5 (should be per-head 64^-0.5); ranks wrong tokens → plausibly explains the negative signal |
| SPIN | EMNLP 2025 (`YUECHE77/SPIN`) | ❌ bug | `interventions.py:168-189` masks **all** prompt tokens in prefill; official gates only the last position → corrupts KV cache → degenerate always-yes (FPR→0.98) |
| OPERA | CVPR 2024 (`shikiw/OPERA`) | ❌ not reproduced | pure delegation wrapper (no algorithm code); hard-fails on stock transformers; **no recorded results**; needs the official fork + verified `key_position` |

- **Reproduced numbers we can stand behind (full POPE 9000 + CHAIR 4977 imgs):**

| Method | POPE MCC | yes-rate | TPR | FPR | ΔMCC | CHAIRi | CHAIRs | CHAIR Δwords |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.731 | 0.450 | 0.813 | 0.087 | — | 0.134 | 0.490 | — |
| PAI (attn-only) | 0.729 | 0.446 | 0.808 | 0.084 | −0.002 | 0.134 | 0.500 | +0.72 |
| ClearSight | 0.725 | 0.486 | 0.848 | 0.125 | −0.006 | 0.135 | 0.493 | −1.79 |
| VisAttnSink | 0.723 | 0.455 | 0.815 | 0.096 | −0.008 | 0.143 | 0.522 | +1.47 |

  (VCD, adversarial-split full audit: MCC 0.668→0.654, FPR 0.145→0.162, yes-rate
  0.479→0.489 — mild harm, no help. DAMRO/SPIN only have adversarial-120 smokes
  and are affected by the bugs above; OPERA has none.)

- Interpretation: **Prong 1 is solidly reproduced.** PAS is a line-for-line
  faithful port and its collapse to chance under length control is a genuine
  property of the score; IC/GLSim/NLL/Entropy are faithful; the sink shape score
  0.5087 (§B) is our own. **Prong 2 holds on the faithful amplification family:**
  PAI is inert, ClearSight and VisAttnSink both raise yes-rate/FPR and drop MCC
  without CHAIR benefit — exactly the answer-prior over-fit the proposal claims.
  Caveat to fix before paper: SVAR is layer-0 (relabel), ClearSight's POPE
  suppression is near-empty (mechanism note), and SPIN/DAMRO/OPERA are not
  currently valid reproductions.
- Supersedes: none (first fidelity audit of the retained baselines).

### 2026-07-08 — Fixed SPIN + DAMRO, dropped OPERA (code; re-run pending)

- Setup: acted on the fidelity audit above. Code changes only; no re-run yet.
- **SPIN** (`mitigation/src/interventions.py` `_spin_head_mask`): now gates heads
  only at the **last query position** in prefill (+ each decoded token), leaving
  earlier prompt positions untouched so the KV cache is not corrupted; head count
  uses truncation `int(routed·H)` (=25 at 0.8) to match official. Unit-tested
  with dummy tensors: only the last row is masked, 25/32 heads kept, earlier rows
  all-ones. Fixes the always-yes collapse.
- **DAMRO** (`mitigation/src/decoding.py` `_select_damro_outlier_features`): CLS
  attention now uses CLIP's real 16-head form (reshape to heads, per-head scale
  `head_dim^-0.5`, average CLS→patch attention across heads) instead of one
  full-dim head scaled by `hidden^-0.5`. Unit-tested: top-10 outlier selection
  overlaps the old buggy version only 3/10, i.e. it now ranks materially
  different tokens.
- **OPERA**: removed from the runnable set (deleted `generate_opera_beam`,
  `check_opera_support.py`, `mitigation/baselines/opera/`, and all wiring in
  `run_task.py` / `run_parallel_mitigation.sh`). It is a `transformers`-fork
  beam-search hook with no local algorithm code and cannot be reproduced in this
  HF stack without vendoring the fork.
- Interpretation: SPIN and DAMRO are now **faithful in code** but their previous
  (buggy) smoke numbers are void — **valid numbers require re-running** them on
  POPE/CHAIR. Until then, only the faithful set with recorded numbers (vanilla,
  PAI, ClearSight, VisAttnSink, VCD) should be reported.
- Supersedes: the "SPIN/DAMRO not valid" status in the 2026-07-08 audit entry (now fixed).

### 2026-07-08 — Fixed SVAR + Beyond-ADS/CGC per the papers, re-run on full COCO

- Setup: LLaVA-1.5-7B, full COCO detection set (16426 mentions, 4009 hallucinated).
  New scripts `detection/scripts/extract_beyond_svar_features.py` (GPU feature
  extraction, all 32 layers, 3 shards) + `detection/scripts/evaluate_beyond_svar.py`
  (offline scoring vs. the same position/length-controlled metrics). Faithful
  formulas from arXiv:2411.16724 (SVAR) and arXiv:2604.04863 (Beyond).
  **Pipeline validated:** layer-0 SVAR reproduces the committed reference exactly
  (overall 0.8336). NOTE: COCO images now live at
  `/srv/common_dataset/coco-2014-dataset` (the `.env` `/data/...` path is stale/empty).
- Fixes applied: SVAR now sums VAR over layers **5–18** (was layer-0 only); ADS now
  uses top-10% foreground + 8-connected **blob mass** `(1−m)·Ĥ_bg`, entropy ÷ log576
  (was plain single-layer entropy); CGC now ranks top-5% patches **by cosine
  similarity** at the **same** token/patch layer (was top-k by attention, layer
  offset); ADS+CGC is now the paper's **supervised** detector — image-grouped OOF
  over the 64-dim `[per-layer ADS ‖ per-layer CGC]` vector (sklearn HistGradientBoosting;
  paper uses XGBoost).
- Result (overall / within-bin / matched / residual AUROC):

| Score | Before (broken) | After (fixed) |
|---|---|---|
| SVAR | 0.834 / 0.575 / 0.576 / 0.557 (layer-0) | **0.805 / 0.568 / 0.566 / 0.548** (layers 5–18) |
| ADS | 0.503 / 0.511 / 0.515 / 0.501 | **0.688 / 0.554 / 0.559 / 0.537** (blob-mass, L16) |
| CGC | 0.692 / 0.521 / 0.520 / 0.518 | **0.872 / 0.752 / 0.750 / 0.679** (sim-topk, best L18; L16: 0.859/0.723) |
| ADS+CGC | 0.693 / 0.523 / 0.524 / 0.521 (fixed avg) | **0.890 / 0.793 / 0.784 / 0.748** (supervised OOF; logreg 0.892/0.792) |

- Interpretation: **the two attention-based fixes do not change the Prong-1
  story** — faithful middle-layer SVAR (within-bin 0.568) and blob-mass ADS
  (0.554) still collapse under length control, confirming the collapse is a
  property of attention-mass/shape detection, not our earlier approximations.
  **The Beyond fixes that matter are the semantic ones:** CGC (token↔patch
  hidden-state similarity) jumps 0.521→**0.752** within-bin — it *survives* the
  control — and the supervised ADS+CGC reaches within-bin **0.793**, the strongest
  controlled detector in the suite. This sharpens Prong 1: what survives length
  control is grounding/semantic consistency (CGC, IC), not attention allocation.
- Artifacts: `experiments/beyond_svar_fixed/` (git-ignored: shard{0,1,2}.npz,
  beyond_svar_metrics.json).
- Supersedes: the §A Beyond-ADS/CGC "simplified proxy" rows and the SVAR "layer-0"
  row (now the faithful fixed numbers).

### 2026-07-08 — Official-repo audit + three new 2026 attention detectors (SAE, LaSCD, TVER)

- **Motivation:** verify the fixed scores line-by-line against the official GitHub
  repos (not just the papers), and broaden the attention-detector comparison with
  recently-published, unsupervised, attention-based metrics. A 4-venue proceedings
  sweep (CVPR/ICCV/ECCV, ACL/EMNLP/NAACL, NeurIPS/ICML/ICLR, surveys+citation graph)
  found the field **saturated**: the only clean new unsupervised attention detector
  not already catalogued is **TVER** (ICCV 2025). Meta-finding: the 2026 detection
  wave is pivoting *away* from attention toward hidden-state/uncertainty probes,
  several explicitly citing attention-sink unreliability — independent support for Prong 1.
- **Repo audit verdicts** (repos cloned & read):
  - **Beyond ADS/CGC** (`tuandung2812alt3/token-grounding-detector`, = CVPR/ICLR-2026
    "The Devil is in the Tokens"): ADS math faithful. **Single-layer ADS ≈ 0.5 is by
    design** — the paper's ADS is a *supervised classifier over the full per-layer ADS
    vector*, reported without length control; our supervised overall (0.865) ≈ paper 0.88.
    **CGC had a one-off bug:** we read the object hidden state at `token_pos+1`; the repo
    builds a prefix ending *before* the object and reads the generating row (= our
    `token_pos`, causal-equivalent). Fixed → CGC within-bin **0.752 → 0.574** (the old
    0.752 "survives" was the artifact).
  - **LaSCD** (`macovaseas/LaSCD`): our `lascd_raw` reproduces `_deco_layer_energy`
    verbatim (per-head 4-neighbour Laplacian, zero-pad, Frobenius→mean, raw 24×24 slice).
    It is a decoding/layer-selector, so detector use is a labeled adaptation.
  - **SAE** (arXiv:2603.16558): **no official repo.** Added the **true** version with
    **Mask2Former-Swin-L** (COCO-panoptic) segmentation-region entropy ÷ log|C| per the
    paper; kept the seg-free raw-patch version as a labeled dispersion ablation.
  - **TVER** (`zifuwan/ONLY`, ICCV 2025): reproduced layer-0 per-head
    `H(text attn)/H(visual attn)` with outlier-zeroing + renorm, mean over heads,
    exactly per the code. Sign confirmed empirically (higher = hallucinated).
- **Setup:** LLaVA-1.5-7B, full 16426-mention set, 2 GPU shards →
  `experiments/beyond_svar_v3/` (git-ignored). Same position/length-controlled metrics.
  Also added **supervised ADS-only / CGC-only** (32-layer OOF) as the faithful test of
  the paper's multi-layer *attention* detector under length control.
- **Result (overall / within-bin AUROC), all faithful:**

| Score | Type | Overall | Within-bin |
|---|---|---|---|
| TVER (ICCV'25, layer 0) | attn, unsup | 0.830 | 0.566 |
| SAE — true (Mask2Former, best L10) | attn×seg, unsup | 0.783 | 0.575 |
| SAE — seg-free skeleton (best L16) | attn disp., unsup | 0.792 | 0.608 |
| LaSCD-raw (max L8–28) | attn, unsup | 0.574 | 0.500 |
| CGC (faithful, best L31) | repr, unsup | 0.728 | 0.574 |
| ADS-only supervised (32-layer) | attn | 0.840 | 0.664 |
| CGC-only supervised (32-layer) | repr | 0.837 | 0.669 |
| ADS+CGC supervised | attn+repr | 0.866 | 0.719 |

- **Interpretation:** Prong 1 is now airtight and reproduction-faithful. Every
  unsupervised **single-score attention** detector collapses under length control —
  including the faithfully-reproduced 2026 **SAE** (with its real segmenter; overall
  0.783 ≈ paper's 0.766, within-bin 0.556–0.575), **LaSCD** (chance, 0.500), and the
  newest **TVER**, whose overall AUROC (0.830) equals the position-only baseline
  (0.8304). Semantic segmentation does *not* rescue SAE. The corrected **CGC** also
  collapses (0.574), so the surviving signal is not "grounding vs attention" per se —
  it is **supervised multi-layer pooling** (ADS/CGC 0.664/0.669, combined 0.719) and
  the single logit-lens **IC** (0.686); the ceiling (0.719) barely exceeds IC.
- Supersedes the §A Beyond-CGC/ADS+CGC within-bin numbers (0.752/0.793 → faithful
  0.574/0.719) and adds the SAE/LaSCD/TVER rows.

### 2026-07-13 — H3 qualitative attention-case figures (object co-occurrence spec §5.4)

- Setup: offline rendering from the saved H3 artifacts
  (`mitigation/results/object_cooccurrence_prior/qualitative_case_{1,2}_attention.npz`,
  LLaVA-1.5-7B, POPE adversarial, first answer-token decision row, layers 9–14
  mean over heads) via the new `mitigation/scripts/visualize_qualitative_attention.py`.
  No model re-run. Overlay style adapted from
  `MM2026/Hallu_Sink_Correlation/show_attention_map.ipynb` (24×24 patch map,
  nearest upsampling to 336², heatmap over the processed image), with `magma`,
  value-modulated opacity, and p99.5-capped color scales instead of jet +
  constant alpha. Outputs: `qualitative_cases_attention.{pdf,png}` (spec §5.4
  five-panel layout: image+bbox | vanilla localization | ClearSight localization,
  shared scale | positive matched residual | measurements), per-layer supplements
  `qualitative_case_{1,2}_layers.{pdf,png}`, exact-values sidecar
  `qualitative_cases_attention_data.json`, and the previously-missing result
  `README.md` (spec §6).
- Result (from the saved measurements, now visible in the figure):
  - Case 2 (truck absent / car present, vanilla no → ClearSight yes): **97.3%**
    of the positive matched residual falls inside the related-object (*car*)
    region — consistent with the H3 associated-evidence mechanism.
  - Case 1 (chair absent / dining table present, vanilla no → ClearSight yes):
    only **2.3%** of the positive residual is in-region, and ClearSight
    *decreases* the related-region absolute mass (0.0399 → 0.0297). Total
    visual mass drops under ClearSight in both cases (0.0882→0.0736,
    0.0709→0.0546).
- Interpretation: the answer flip to "yes" is reproduced in both cases, but the
  "intervention concentrates extra attention on the related object" narrative
  (spec §5.1 criterion 4) holds only for case 2. Case 1 must be re-selected from
  the saved 72-candidate pool or presented as a general associated-evidence
  false positive (spec §5.1 fallback). H2 — the paired
  difference-in-differences quantitative half — is still pending.
- Supersedes: `qualitative_cases.{pdf,png}` (residual-only layout) as the
  paper-facing H3 figure; the old files are kept for history.

### 2026-07-14 — Clean-map qualitative figure (position prior + sink tokens removed)

- Motivation: visualize the H3 cases with a **clean** attention map — RoPE
  position prior removed *and* detected sink tokens deleted — following the sink
  machinery of `MM2026/Hallu_Sink_Correlation/utils/AttnAdapter.py`.
- Setup: `mitigation/scripts/extract_qualitative_sink_attention.py` reproduces the
  vanilla first-answer-token decision for the two saved cases (same image/POPE
  prompt), LLaVA-1.5-7B, layers 9–14. Sink detection = `auto_detect_sinks` on
  pre-RoPE key L2 norm; purification / no-RoPE reuse the pure-torch
  `detection/src/sinkdetect/sink_utils.py`. Runs in the `latentGuard` env
  (transformers 4.48.1) via `output_attentions` + a forward pre-hook, so it does
  **not** import the tfm-5.x `DetectionAdapter`/`interventions.py`. Model cache is
  `/srv/common_model/huggingface` (the `.env` HF path is stale like the COCO path).
- **Faithfulness gate PASSED**: reproduced `original` visual attention vs the saved
  `vanilla_attention` — max|Δ| = 1.5e-3 (case 1), 6.4e-4 (case 2), both < 5e-3.
  This also confirms tfm 4.48.1 reproduces the run that produced the H3 npz.
- Result (in-region attention-mass fraction, raw → no-RoPE → clean):
  - Case 2 (truck absent / car present): **0.318 → 0.519 → 0.573** — removing the
    position prior and sinks *concentrates* attention onto the related object,
    consistent with the H3 associated-evidence mechanism.
  - Case 1 (chair absent / dining table present): 0.472 → 0.458 → 0.449 (flat) —
    same selection caveat as the 2026-07-13 entry (case 1 fails §5.1 criterion 4).
  - Removing RoPE raises total visual mass ~0.08 → ~0.55; detected sinks are
    low-key-norm border/background patches (14–19 flagged in ≥4/6 band layers).
- Artifacts: `qualitative_case_{1,2}_sink_attention.npz`,
  `qualitative_cases_clean_attention.{pdf,png}`,
  `qualitative_cases_clean_attention_data.json` (all under
  `mitigation/results/object_cooccurrence_prior/`). New scripts:
  `mitigation/scripts/extract_qualitative_sink_attention.py` and the extended
  `visualize_qualitative_attention.py` (`render_clean_figure`). Minor additive
  change: `DetectionAdapter` now stashes `last_key_norms` (unused here; for a
  future tfm-5.x DetectionAdapter run).

### 2026-07-14 — HCVR/LCVR attention-amplification on case 1268 (paper Fig. 4 reproduction)

- Question: does amplifying attention on **high-credibility** vs **low-credibility**
  visual regions move the answer symmetrically (paper's claim: HCVR↑→"yes",
  LCVR↑→"no")? Single case: POPE adversarial qid 1268, "Is there a traffic light in
  the image?" (COCO_val2014_000000244455; traffic light ABSENT). LLaVA-1.5-7B,
  band layers 9–14, first answer-token decision row.
- Setup: `mitigation/scripts/hcvr_lcvr_intervention.py`. Regions are defined on the
  **CLEAN** map (no-RoPE position prior removed + detected sink tokens deleted,
  sinks *excluded* from the ranking): HCVR = top-q, LCVR = bottom-q of the 558
  eligible non-sink visual tokens (18 sink tokens dropped). Intervention =
  **post-softmax** multiply of the region's attention columns by a factor, for all
  post-image query rows, in every band layer, **no row renormalization** (renorm
  saturates at ~10–20%). Gated monkeypatch of tfm-4.48 `eager_attention_forward`;
  runs in `latentGuard` env. Baseline P(yes)=P(no)=0.500 (argmax "No"). Cache +
  render split (GPU once, restyle with no GPU). Dense sweep factors 1.0→20; the two
  line plots share one x-axis (1.00–2.25, step 0.25) scaled to HCVR's +30% point.
- Result (probability change over the displayed window, factor 1.0→2.25):

  | config | HCVR amplify → P(yes) | LCVR amplify → P(no) |
  |---|---|---|
  | top 25% / bottom 25% (140/140 tokens) | 0.500 → **0.804**  (+30.4%) | 0.500 → 0.542  (+4.2%) |
  | top 10% / bottom 50% (56/279 tokens)  | 0.500 → **0.811**  (+31.1%) | 0.500 → 0.533  (+3.3%) |

  - **Strong asymmetry, both configs.** Amplifying HCVR drives P(yes) up hard
    (~+30%, robust, monotonic to a ~0.88 plateau around factor 3). Amplifying LCVR
    moves P(no) only ~3–4% and, per the dense sweep, **reverses** at higher factors
    (LCVR P(yes) climbs back above 0.5 past factor ~6–8). So *both* regions, pushed
    hard enough, add **yes**-bias — the paper's symmetric HCVR-yes / LCVR-no does
    not hold in magnitude. This is direct evidence for Prong-2 (pushing attention
    injects answer bias, it is not target-object evidence verification).
  - **Which quantile split is better: top 25% / bottom 25%.** HCVR effect is
    essentially identical to the 10/50 split (~+30%), but the LCVR→no trend is
    cleaner and slightly larger (+4.2% vs +3.3%), the regions are balanced (equal
    140/140 token counts), and it matches the paper's own 25% choice. The 10/50
    split's LCVR is a diffuse half-image region with a weaker, flatter early
    response. Note the two line plots use independent y-axes (fig4 is zoomed to
    0.46–0.54), so the ~4% LCVR effect *looks* as steep as the 30% HCVR effect —
    the magnitudes differ ~7×.
- Artifacts (git-ignored `mitigation/results/object_cooccurrence_prior/`):
  `hcvr_lcvr_1268_t25_b25/` and `hcvr_lcvr_1268_t10_b50/`, each with 4 bare figures
  (no titles / tight margins): `fig1_HCVR_top`, `fig2_LCVR_bottom` (binary region
  highlights on the image), `fig3_amplify_top`, `fig4_amplify_bottom` (line plots,
  shared round x-axis), plus `cache.npz`, `sweep_dense.csv`, `sweep_{HCVR,LCVR}_*.csv`,
  `attention_distribution.csv`, `summary.json`.

### 2026-07-14 — HCVR/LCVR region definition: by token-count vs by cumulative score (mass)

- Follow-up to the same-day HCVR/LCVR entry. Question (raised by user): should HCVR/LCVR
  be the top/bottom fraction of **tokens** (count) or of the **cumulative clean-attention
  score** (mass)? Added `--select_by {count,mass}` to `hcvr_lcvr_intervention.py`; same
  case 1268, same post-softmax no-renorm intervention, top_q=bottom_q=0.25.
- The clean map is only moderately concentrated (top-10 eligible tokens hold 7.3% of
  mass; top-25%-of-tokens hold 47.7%). Under the two definitions the regions differ a lot:

  | 25% split | HCVR tokens / mass | LCVR tokens / mass | HCVR amplify→P(yes) @2.25 | LCVR amplify→P(no) @2.25 |
  |---|---|---|---|---|
  | by count | 140 / **47.7%** | 140 / **10.8%** | 0.500→0.804 (+30.4%) | 0.500→0.542 (+4.2%) |
  | by mass  | 55 / 25.1% | 261 / 25.1% | 0.500→0.814 (+31.4%) | 0.500→0.539 (+3.9%) |

- **The HCVR≫LCVR asymmetry is robust to the region definition** (HCVR ~+31% toward yes,
  LCVR ~+4% toward no then reverses, under both). But **mass-based is the more appropriate
  (and more rigorous) definition**, for two reasons:
  1. It matches the "credibility = evidence mass" semantics — HCVR becomes a tight 55-token
     high-score set (visually concentrated on the salient riders/vehicles), LCVR a broad
     261-token low-score field.
  2. **It removes a confound.** Count-based HCVR already holds 47.7% of the eligible mass
     vs LCVR's 10.8%, so multiplying both by the same factor injects ~4.4× more *absolute*
     attention mass into HCVR — the bigger HCVR effect is partly just "more mass moved."
     Mass-based equalizes both regions at 25.1% mass, i.e. the *same* absolute injection,
     and HCVR **still** moves the answer ~8× more than LCVR (+31% yes vs +4% no). So the
     asymmetry is genuinely about *where* attention lands (credibility), not how much mass
     is amplified — a cleaner Prong-2 result.
- Recommendation: use `--select_by mass` for the paper figure. Artifacts:
  `hcvr_lcvr_1268_t25_b25_mass/` (55/261 tokens), alongside the renamed
  `hcvr_lcvr_1268_{t25_b25,t10_b50}_count/`.

### 2026-07-15 — ClearSight paired prediction-distribution shift on POPE adversarial

- Setup: paired all 3000 rows by `question_id` between
  `mitigation/results/coco_llava_7b_attention_only/pope/adversarial/{vanilla,clearsight}/predictions.jsonl`.
  The set is balanced (1500 positive / 1500 negative); all IDs matched and every
  output normalized exactly to yes/no.
- Prediction distribution: vanilla = 1438 yes / 1562 no (47.93% / 52.07%);
  ClearSight = 1574 yes / 1426 no (52.47% / 47.53%), a **+136 / +4.53-point
  shift toward yes**.
- Paired transitions: 1438 yes→yes, 1426 no→no, **136 no→yes, 0 yes→no**.
  Of the 136 flips, 53 correct vanilla false negatives, but 83 convert true
  negatives into false positives. Thus TP +53, FP +83, TN −83, FN −53.
- Consequence: TPR 81.20%→84.73% (+3.53pt), but FPR 14.67%→20.20%
  (+5.53pt); accuracy 83.27%→82.27% (−1.00pt), precision 84.70%→80.75%,
  MCC 0.666→0.646. ClearSight acts as a one-directional no→yes intervention;
  its extra false positives outnumber corrected false negatives by 30.

### 2026-07-19 — Qwen3-VL XEV iteration 1: symmetric keep/drop evidence fails

- **Question:** can an attention-proposed region verify a target by contrasting
  target evidence against open-vocabulary / COCO / oracle alternatives? This is
  the first constructive-method pilot after the 2026-07-07 restart.
- **Setup:** `Qwen3-VL-8B-Instruct`, local copy at
  `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`, Transformers 4.57.6, bf16 eager
  attention. POPE-COCO adversarial first 120 rows = 60 yes / 60 no, 20 unique
  images. One proposal/question from Qwen text layers 18–29, mean over heads and
  layers, top-30%-cumulative visual-attention components, fixed 200704 image
  pixels (roughly 187–256 merged visual tokens). `I_keep` keeps the proposal
  clear and blurs the rest; `I_drop` does the complement; a random same-area box
  is retained as control. Two 4090 shards, 3.65 s/sample mean. Planned score:
  `E(h,R)=m(h,I_keep)-m(h,I_drop)` and target support minus strongest positive
  competing `E`. Candidate sources: COCO-80 train NPMI top-8 (`closed`),
  target-free clear-region inventory (`open`), and COCO GT-present objects
  (`oracle`, ceiling only), all passed through an S/C/N semantic relation filter.
  Target/full/keep/drop prompts reuse each POPE question exactly; only alternative
  hypotheses use generated existence-question templates.
- **Vanilla full-image anchor:** margin AUROC **0.9346**, AUPRC 0.9460; TP/FP/TN/FN
  = 50/4/56/10, accuracy = balanced accuracy **0.8833**, FPR 0.0667, MCC 0.7705,
  yes-rate 0.450.
- **Planned keep-minus-drop score fails and reverses ordering:**

  | score | AUROC | AUPRC | mean yes | mean no |
  |---|---:|---:|---:|---:|
  | target-only `E` | **0.3432** | 0.4485 | -1.217 | 4.246 |
  | random same-area `E` | **0.1808** | 0.3654 | -8.604 | 1.685 |
  | closed relative `E` | 0.4079 | 0.4884 | -3.638 | -0.275 |
  | open relative `E` | 0.4032 | 0.4861 | -0.533 | 3.348 |
  | oracle relative `E` | **0.3483** | 0.4446 | -1.675 | 3.658 |

- **Root cause from saved-margin decomposition:** `m(target,I_keep)` by itself is
  directionally valid (AUROC **0.8383**, means yes/no 4.129/−5.946), and target
  removal `m(full)-m(drop)` is weaker but valid (AUROC **0.7314**, means
  4.021/0.269). Their symmetric difference is invalid because `I_keep` removes
  most context while `I_drop` preserves most context: `keep-full` is −5.238 on
  positives but +3.977 on negatives. The contrast therefore measures isolated-
  region yes-bias/context loss as much as target evidence. **Reject
  `m_keep-m_drop` as the XEV evidence definition.**
- **Post-hoc mechanism diagnostic (not a pre-registered method result):** on the
  same keep view, penalize an alternative only when it outranks target support:
  `d=max(0,m_alt-m_target)`, score `m_target-d`.

  | candidate source | AUROC | all-neg FPR @ matched 80% TPR | related-present FPR |
  |---|---:|---:|---:|
  | target keep margin (no candidates) | **0.8383** | 0.3500 | 0.5000 |
  | closed dominance-gated | 0.8339 | **0.3167** | **0.2500** |
  | oracle dominance-gated | 0.8207 | 0.3333 | **0.2500** |
  | open dominance-gated | 0.8175 | 0.3500 | 0.4375 |

  Ties make the achieved closed/oracle/open TPR 0.8167. The alternative mechanism
  cuts the 16-example related-present FPR slice, but does not improve overall
  ranking. Subtracting competitor excess from the full-image baseline also fails
  the held-out behavior gate: closed/oracle test balanced accuracy **0.8333** vs
  vanilla **0.8571**; open is unchanged at 0.8571. This is not yet a mitigation.
- **Candidate diagnosis:** open generation yields 4.625 phrases/region and all 120
  relation responses parse successfully (217 C / 1415 N / 40 S labels across
  source unions). On the 16 oracle-related negative rows, closed candidates cover
  **81.25%** of GT competing identities and produce alternative dominance in
  62.5%; open candidates cover only **31.25%** (dominance 50.0%). The exact target
  appears in the target-free open inventory on 48.33% of positives vs 5.0% of
  negatives. Open vocabulary is feasible, but candidate/proposal recall is not
  yet competitive with the closed diagnostic.
- **Kill-criterion decision:** oracle dominance loses 0.0176 AUROC vs target keep,
  but lowers related-present FPR by 0.25 at matched TPR. Therefore the narrow
  alternative-confusion lead survives, while **candidate expansion is frozen**:
  fix context matching, proposal recall, and dominance gating with closed/oracle
  candidates before spending compute on broader open-vocabulary acquisition.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/round1_exact_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/round1_exact_summary.json`. Reproducible code:
  `mitigation/src/xev.py`, `mitigation/scripts/run_qwen3vl_xev_pope.py`, and
  `mitigation/scripts/analyze_qwen3vl_xev_pope.py`.

### 2026-07-19 — Qwen3-VL XEV iteration 2: strict identity-switch certificate has zero coverage

- **Pre-registered hypothesis:** a high-precision false-positive veto may be
  possible when a target-free open inventory and the closed COCO-NPMI inventory
  independently name the same competing identity. The rule, fixed using only
  iteration 1, requires: baseline margin >= 0; target absent from the open
  inventory; an exact open/closed candidate intersection labeled C; candidate
  keep margin > 0; and candidate-minus-target keep margin >= 6.0. The action is
  a hard yes-to-no veto. The holdout gate required at least 3 triggers, >= 2/3
  correction precision, lower FPR, TPR loss <= 0.005, and non-decreasing MCC.
- **Confirmatory setup:** `Qwen3-VL-8B-Instruct`, POPE-COCO adversarial rows
  120--599, disjoint from iteration 1 and aligned to image boundaries: 480 rows,
  240 yes / 240 no, 80 unique images. Same attention layers, top-30% cumulative
  mass proposal, fixed 200704 pixels, one region, and exact prompts as iteration
  1. Closed/open candidates were extracted; oracle candidates were omitted.
  Two 4090 shards completed in 12.1 minutes; mean runtime 2.88 s/sample.
- **Vanilla anchor:** margin AUROC **0.9480**, AUPRC 0.9566; TP/FP/TN/FN =
  192/15/225/48; balanced accuracy **0.8688**, TPR 0.8000, FPR 0.0625, MCC
  0.7446, yes-rate 0.4313.
- **Certificate result:** **0 triggers**, 0 false-positive corrections, and 0
  true-positive vetoes. All behavior metrics are unchanged; the pre-registered
  gate fails its minimum-trigger condition. Reject exact open/closed lexical
  agreement as a usable certificate.
- **Diagnostic:** same-view target keep margin remains informative on the larger
  window (AUROC **0.8815**, AUPRC 0.9072), while keep-minus-drop remains reversed
  (target AUROC 0.4038; closed 0.4653; open 0.4152). The failure is candidate
  confirmation coverage, not absence of local target signal. Open generation
  averages 4.648 phrases/region with zero relation parse failures; exact target
  occurrence is 35.83% on positives vs 3.75% on negatives.
- **Decision:** preserve same-view local verification, but replace independent
  existence-margin comparison with an explicit target-vs-alternative visual
  question. Candidate generators remain proposal mechanisms only. Do not relax
  the rejected certificate using this holdout; select the next rule on iteration
  1 and evaluate it on a new disjoint window.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter2_holdout_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter2_holdout_summary.json`.
