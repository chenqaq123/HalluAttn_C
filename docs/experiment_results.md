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

### 2026-07-19 — Qwen3-VL XEV iteration 3 development: counterbalanced pairwise certificate

- **Question:** can explicit target-vs-alternative visual discrimination replace
  the zero-coverage agreement of two candidate inventories while avoiding the
  yes-bias and coexistence failures of independent existence margins?
- **Method:** for up to three target-free open/closed candidates with positive
  keep margin, ask both directions on the same clear-region view: target rather
  than alternative, and alternative rather than target. Define target preference
  `v=0.5*(margin_target-first-margin_alternative-first)`. Reversing the pair
  cancels shared yes/order bias and gives a natural zero threshold. Exact target
  occurrence in the open inventory suppresses candidate testing. Oracle labels
  are not used.
- **Setup:** same 120-row / 20-image POPE-adversarial development set as
  iteration 1, `Qwen3-VL-8B-Instruct`, one top-30%-mass proposal, closed top-8
  NPMI plus up to six open phrases, pairwise top-3. Two 4090 shards; mean runtime
  3.25 s/sample. This reuses the development labels and is not confirmatory.
- **Ablation / failure:** pairwise `v<0` alone triggers 15 times: only 2 baseline
  false-positive corrections versus **13 true-positive vetoes**. Behavior falls
  from balanced accuracy 0.8833 to 0.7917 and MCC 0.7705 to 0.6227. Explicit
  comparison does not by itself solve region coexistence or proposal mismatch.
- **Frozen identity-switch certificate:** veto a baseline yes only when (1)
  target keep margin `<0`, (2) the alternative has semantic relation `C`, and
  (3) counterbalanced `v<0`. All numeric thresholds are their natural zero; no
  logit cutoff is fitted. Candidate generation and visual judgment are separate.
- **Development result:** 2 triggers, both corrections and zero TP vetoes:
  `dining table -> chair` and `handbag -> backpack`. TP/FP/TN/FN changes from
  50/4/56/10 to **50/2/58/10**; TPR stays 0.8333, FPR 0.0667 -> **0.0333**,
  balanced accuracy 0.8833 -> **0.9000**, MCC 0.7705 -> **0.8072**, and yes-rate
  0.4500 -> 0.4333. The old exact-intersection certificate corrects only the
  handbag case; pairwise visual discrimination adds the chair case without
  requiring lexical candidate agreement.
- **Pre-registered confirmatory gate:** on a new disjoint window, require at
  least 3 triggers, correction precision >= 2/3, lower FPR, TPR loss <= 0.005,
  and non-decreasing MCC. Candidate top-k=3, target/open support guard, semantic C
  filter, and all zero thresholds are frozen before the holdout.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter3_dev_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter3_dev_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 4: pairwise hard veto fails confirmatory holdout

- **Frozen setup:** iteration-3 code and rule without modification on disjoint
  POPE-COCO adversarial rows 600--1079: 480 rows, 240 yes / 240 no, 80 unique
  images. This window overlaps neither rows 0--119 (development) nor 120--599
  (iteration-2 holdout). One top-30%-mass region, open+closed candidates,
  pairwise top-k=3, exact target-in-open guard, target keep `<0`, relation C, and
  counterbalanced preference `<0`. All thresholds and the success gate were
  fixed before inference. Two 4090 shards; mean runtime 2.89 s/sample.
- **Vanilla anchor:** margin AUROC **0.9314**, AUPRC 0.9459; TP/FP/TN/FN =
  198/20/220/42; balanced accuracy **0.8708**, TPR 0.8250, FPR 0.0833, MCC
  0.7448, yes-rate 0.4542.
- **Confirmatory result:** the certificate triggers 6 times and **all 6 are true-
  positive vetoes**; it corrects 0 false positives (precision 0). TP/FP/TN/FN
  becomes 192/20/220/48; TPR falls to 0.8000, FPR is unchanged, balanced accuracy
  falls to **0.8583**, MCC to **0.7216**, and yes-rate to 0.4417. Every substantive
  pre-registered gate fails; reject the single-region hard veto.
- **Failure cases:** `motorcycle -> car`, `skateboard -> fence`, `spoon -> knife`,
  `truck -> vehicle`, `car -> concrete curb`, and `bed -> couch`. Four are
  coexisting/associated identities in a region that does not cover the real
  target; two also expose C-relation errors (`vehicle` is a hypernym and curb is
  associated with car). The pairwise margins can be locally coherent while the
  image-level absence inference is wrong.
- **False-positive audit:** among the baseline's 20 FPs, several have plausible
  pairwise alternatives (`chair -> couch`, `handbag -> backpack`, `bicycle ->
  motorcycle`) but are blocked because target keep margin is positive. Thus
  tightening the one-region target gate trades recall and precision rather than
  solving coverage.
- **Decision:** retain counterbalanced pairwise comparison only as a local
  identity test. Replace single-region negative target margin with a target-free
  full-image support guard, evaluated lazily on preliminary local triggers. A
  local alternative cannot by itself prove image-level absence.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter4_holdout_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter4_holdout_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 5 development: lazy global-support consensus

- **Failure-directed redesign:** demote local pairwise comparison from a veto to
  a proposal. A local proposal requires a top-3 positive-keep candidate from the
  closed NPMI source, C relation, and counterbalanced target preference `<0`.
  Only then generate two target-free full-image object inventories: a direct
  inventory and a systematic left-to-right/foreground-to-background inventory
  that discourages guessing related categories. Cancel the veto only when both
  inventories contain a lexical target equivalent or an S relation. No target
  appears in either inventory prompt; no oracle annotations are used.
- **Selected-case audit:** 10 known local proposals, deliberately comprising 5
  prior false-positive corrections and 5 prior true-positive vetoes. This is a
  mechanism audit, not a representative evaluation. Both inventories support all
  five real targets (`dining table`, `motorcycle`, `spoon`, `bottle`, `bed`) and
  suppress their vetoes. Neither jointly supports any of the five absent targets
  (`dining table`, two `handbag` cases, `chair`, `bicycle`), so all five
  corrections remain. Result: 5 corrections / 0 TP vetoes on the selected set.
- **Why two inventories:** the first inventory alone falsely supports absent
  `handbag` via generic `bag` and hallucinates an absent `bicycle`; the second
  returns neither. Requiring consensus removes both false supports. A tested
  full-image pairwise consistency check is rejected: it favors the salient
  alternative on all seven initial cases, including all three real coexisting
  targets, and would reinstate the TP vetoes.
- **Retrospective behavior (development only):** on rows 0--119, predicted
  corrections are qids 42 and 50: TP/FP/TN/FN 50/4/56/10 -> 50/2/58/10, FPR
  0.0667 -> 0.0333, balanced accuracy 0.8833 -> 0.9000, no recall loss. On the
  now-open iteration-4 rows 600--1079, predicted corrections are qids 670, 728,
  and 884: TP/FP/TN/FN 198/20/220/42 -> 198/17/223/42, FPR 0.0833 -> 0.0708,
  balanced accuracy 0.8708 -> 0.8771, MCC 0.7448 -> 0.7583, no recall loss. These
  numbers are not confirmatory because both windows informed the rule.
- **Frozen next test:** local source requirement `closed`, top-k=3, C relation,
  local pairwise `<0`, two target-free inventories of size 12, unanimous support
  guard. On rows 1080--1559 require >=3 triggers, correction precision >=2/3,
  lower FPR, TPR loss <=0.005, and non-decreasing MCC.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter5_consensus_audit_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter5_consensus_audit_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 6: inventory-consensus rule fails holdout

- **Frozen setup:** POPE-adversarial rows 1080--1559, disjoint 480 rows / 80
  images, 240 yes / 240 no. Closed local candidate provenance, top-k=3, C,
  local pairwise preference `<0`, two target-free inventories of size 12, and
  unanimous lexical/S support. All code and gates were fixed in commit `ea2411f`
  before inference. Mean runtime 2.90 s/sample on two 4090 shards.
- **Vanilla anchor:** margin AUROC **0.9286**, AUPRC 0.9428; TP/FP/TN/FN =
  191/16/224/49; balanced accuracy **0.8646**, TPR 0.7958, FPR 0.0667, MCC
  0.7362, yes-rate 0.4313.
- **Confirmatory result:** 11 local proposals, 7 unanimous inventory supports,
  and 4 final triggers. All four are TP vetoes and none correct an FP. TP/FP/TN/FN
  becomes 187/16/224/53; TPR **0.7792**, unchanged FPR 0.0667, balanced accuracy
  **0.8563**, MCC **0.7211**. The pre-registered gate fails.
- **Failures:** `backpack -> suitcase`, `traffic light -> train`, `train -> car`,
  and `handbag -> backpack`. Their pairwise directions are respectively
  `(0.0,0.75)`, `(0.0,5.25)`, `(-8.5,-0.5)`, and `(1.0,5.0)` for
  target-first/alternative-first margins. All have negative *differences*, but
  none exhibits the strict target-no / alternative-yes directional switch.
- **Support-filter diagnosis:** two baseline FPs have strong directional local
  switches but are incorrectly suppressed: absent car is "supported" by
  wheel/tire/door, and absent backpack by generic bag. The model-generated S
  relation is therefore not a reliable support matcher.
- **Decision:** reject two-inventory absence as sufficient evidence. Require the
  strict directional sign switch and replace generated S support with a fixed,
  auditable COCO alias matcher. Rows 1080--1559 become development-only.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter6_holdout_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter6_holdout_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 7 development: directional/alias precision repair

- **Repair 1, local direction:** replace `target_preference<0` with the stronger
  natural sign pattern `target-first margin<0` and `alternative-first margin>0`.
  This requires the model to answer no when target is named first and yes under
  the reversed alternative, rather than accepting a weak negative difference or
  two same-direction answers.
- **Repair 2, global support:** retain the two target-free inventories, but remove
  generated S labels from the decision. Match only lexical identity and an
  explicit alias table for transparent COCO surface variants (`table` for dining
  table, `sofa` for couch, `phone` for cell phone, etc.). Generic `bag` does not
  support handbag/backpack; wheel/tire/door do not support car. The inventory
  phrases themselves remain open-vocabulary.
- **Retrospective development result:** replaying saved directional margins and
  inventories on the full iteration-6 480 rows produces 4 strict local proposals,
  2 inventory-supported real targets, and 2 final triggers. Both are baseline FP
  corrections (`car -> truck`, `backpack -> suitcase`), with zero TP vetoes.
  TP/FP/TN/FN 191/16/224/49 -> 191/14/226/49; FPR 0.0667 -> **0.0583**,
  unchanged TPR 0.7958, balanced accuracy 0.8646 -> **0.8688**, MCC 0.7362 ->
  **0.7455**. This window informed the repair and is not confirmatory.
- **Cross-case check:** on the 10 iteration-5 mechanism cases, the strict rule
  retains four known corrections and no TP veto; it drops the ambiguous q42
  correction because both directional margins are negative. End-to-end smoke
  confirms old TP q1093 no longer forms a proposal and FP q1178 still does.
- **Frozen confirmatory setup:** rows 1560--2519 (960 rows / 160 images), closed
  candidate provenance, top-k=3, C, strict directional signs, two target-free
  inventories of size 12, unanimous alias support. Gate: >=3 triggers, >=2/3
  correction precision, lower FPR, TPR loss <=0.005, non-decreasing MCC.
- **Artifacts (git-ignored):** `experiments/qwen3vl_xev/iter7_smoke.jsonl`.

### 2026-07-19 — Qwen3-VL XEV iteration 8: directional/alias rule passes holdout

- **Confirmatory setup:** frozen commit `2a47784`, POPE-adversarial rows
  1560--2519: 960 balanced rows, 480 yes / 480 no, 160 untouched images. Closed
  top-k=3 local provenance, C relation, strict target-first `<0` and
  alternative-first `>0`, two target-free full-image inventories of size 12,
  and unanimous lexical/explicit-alias support. No generated S relation and no
  oracle labels. Mean runtime 2.67 s/sample on two 4090 shards.
- **Vanilla anchor:** margin AUROC **0.9442**, AUPRC 0.9538; TP/FP/TN/FN =
  404/37/443/76; balanced accuracy **0.8823**, TPR 0.8417, FPR 0.0771, MCC
  0.7671, yes-rate 0.4594.
- **XEV v1 result:** 7 strict local proposals; the global guard suppresses 3 and
  leaves 4 certificates. Three correct baseline FPs (`chair -> couch`, two
  `cup -> wine glass`) and one vetoes a real couch as chair. Correction precision
  is **0.75**. TP/FP/TN/FN becomes 403/34/446/77; FPR **0.0708** (-0.00625), TPR
  **0.8396** (-0.00208), balanced accuracy **0.8844** (+0.00208), MCC **0.7719**
  (+0.00473), yes-rate 0.4552. Action rate is 4/960 = **0.42%**.
- **Pre-registered gate:** PASS. Triggers 4 >=3; correction precision 0.75 >=2/3;
  FPR decreases; TPR loss 0.00208 <=0.005; MCC increases.
- **Residual failure:** for qid 1587, the first inventory raw output contains
  `sofa` after the parser's 12-item cutoff while the second contains it within
  12. Unanimous support therefore fails and a real couch is vetoed. This is
  disclosed but not repaired on the holdout. Inventory length remains frozen.
- **Interpretation:** this is the first usable held-out XEV version, but its gain
  is narrow. It supports a high-precision local identity-switch certificate,
  not a general replacement for Qwen3 decoding. Closed COCO candidates remain a
  reproducible local proposal gate; target-free inventories remain open-vocabulary.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter8_holdout_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter8_holdout_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 9: no-RoPE localization and coverage audit

- **Question:** does removing the multimodal RoPE position prior from attention
  localization fix XEV v1's very low 4/960 action rate?
- **Implementation:** capture each selected text layer's Q/K immediately before
  RoPE and recompute the first-answer-token attention distribution over the full
  causal row. Only the map used for region proposal changes; the original
  post-RoPE forward still produces hidden states and logits. `--attention_map
  no_rope` is now the default, `raw` is retained for ablation, outputs use schema
  v4, and the analyzer now emits a consolidated `coverage_funnel`.
- **Faithfulness smoke:** on qid 2294, raw and no-RoPE baseline margins are both
  0.25 while the proposal moves from `[40,480,320,640]` to `[0,0,200,80]`. Across
  the seven iteration-8 strict-local cases, all baseline margins remain identical;
  3/7 boxes are unchanged, 3 expand modestly, and qid 2294 has zero box overlap.
- **Paired development setup:** POPE-adversarial rows 2520--2639, 120 balanced
  rows / 20 images, otherwise the frozen XEV-v1 directional/alias configuration.
  Run both raw and no-RoPE maps on the exact same rows. This is a localization and
  coverage audit, not a confirmatory behavior test.
- **Localization effect:** only 42/120 proposal boxes are exactly identical;
  raw/no-RoPE box IoU has mean 0.628 and median 0.670, with 15/120 disjoint boxes.
  All 120 baseline margins are exactly identical. RoPE removal therefore changes
  localization materially without changing the base model decision.
- **Coverage funnel:** baseline behavior is TP/FP/TN/FN = 54/5/55/6 for both.

  | stage | raw | no-RoPE |
  |---|---:|---:|
  | baseline Yes | 59 | 59 |
  | pairwise-scored rows | 27 | 31 |
  | negative-preference rows | 21 | 25 |
  | strict closed+C+directional local proposals | 1 | 1 |
  | suppressed by global target support | 1 | 1 |
  | final actions | **0** | **0** |

- **Interpretation:** no-RoPE slightly increases broad pairwise coverage but does
  not pass the strict local certificate more often. The earlier 4/960 result is
  sparse because its funnel contracts 441 baseline Yes -> 242 pairwise-scored ->
  176 negative preference -> 7 strict local -> 4 final actions. The coverage
  bottleneck is the closed-source + C-relation + strict-direction conjunction,
  not the position prior or final global guard. Keep no-RoPE as localization
  hygiene, but do not claim it improves behavior. XEV v1's held-out numbers remain
  tied to the raw map until the no-RoPE variant is separately confirmed.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter9_{no_rope_dev,raw_control}_shard{0,1}.jsonl` and
  corresponding summaries.

### 2026-07-19 — Qwen3-VL XEV iteration 10: broad local conflicts expose ungrounded global support

- **Pre-registered setup:** POPE-COCO adversarial rows 2520--2639, already a
  development-only window: 120 balanced rows / 20 images. Use no-RoPE
  localization and one top-30%-mass proposal. For a baseline Yes with no exact
  target in the local open inventory, accept any positive-keep open/closed
  candidate labeled C or N when target-first margin `<0` and alternative-first
  margin `>0`. Define global target support as complement target margin `>=0` OR
  unanimous lexical/alias target support in two target-free inventories. All
  thresholds and the gate were committed in `527c0d3` before inference.
- **Vanilla anchor:** TP/FP/TN/FN = 54/5/55/6; balanced accuracy **0.9083**,
  TPR 0.9000, FPR 0.0833, MCC **0.8168**, yes-rate 0.4917.
- **Coverage result:** 59 baseline-Yes rows -> 31 pairwise-scored -> 25 negative
  preferences -> **21 broad local conflicts**. Inventory support fires on 11,
  complement support on 15, and their union on 18. The 3 remaining actions are
  all true-positive vetoes and correct no baseline false positive. TP/FP/TN/FN
  becomes 51/5/55/9; balanced accuracy falls to **0.8833**, TPR to 0.8500,
  FPR remains 0.0833, and MCC falls to **0.7684**. The pre-registered gate fails.
- **Support-source audit:** with no support guard, the broad local rule would act
  21 times for 4 corrections / 17 TP vetoes (precision 0.1905). Inventory-only
  leaves 10 actions for 1 correction / 9 TP vetoes (0.10). Complement-only
  leaves 6 actions for 2 corrections / 4 TP vetoes (0.3333). The pre-registered
  union leaves 3 actions for 0 corrections / 3 TP vetoes. These source ablations
  explain the failure; they are not tuned alternatives.
- **Mechanism audit:** the inventories falsely repeat absent targets from scene
  context (for example absent backpack and spoon) while omitting small real
  targets (for example cell phone). The complement margin protects many real
  targets but remains an image-level Yes/No judgment and fails on three real
  small/accessory targets. Thus unioning more nominal support maximizes
  suppression without making that support grounded.
- **Decision:** the C-only relation was indeed a proposal-level category error:
  allowing N raises local coverage 21-fold on this window. But a local conflict
  does not prove image-level absence, and neither free-form full-image inventory
  nor one-region complement supplies reliable independent evidence. Reject the
  support union and do not tune a margin threshold. Next test multi-region local
  identity verification, where attention only proposes regions and all support
  must be tied to a verified region.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter10_local_conflict_dev_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter10_local_conflict_dev_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 11: residual multi-region verification still backfires

- **Pre-registered setup:** the same development-only POPE-adversarial rows
  2520--2639 (120 balanced rows / 20 images). Iteratively extract up to three
  no-RoPE top-30%-mass connected proposals, suppressing each selected grid box
  plus its dilation halo before the next extraction. For every region, compare
  the target against up to three positive-keep open/closed C/N candidates. A
  strict target-no/alternative-yes switch is conflict; the exact mirrored signs
  are target support. Veto a baseline Yes only with any conflict and no support
  across the available proposal prefix. No inventories, complement margin,
  generated S decision, oracle annotation, or fitted threshold is used.
- **Sanity checks:** all 120 rows produce exactly three proposals; 436 pairwise
  comparisons are evaluated with zero relation parse failures. The baseline
  exactly reproduces iteration 10: TP/FP/TN/FN = 54/5/55/6, balanced accuracy
  0.9083, TPR 0.9000, FPR 0.0833, and MCC 0.8168. Mean runtime is 8.10 s/row on
  two RTX 4090s.
- **Paired proposal-prefix result:**

  | prefix | conflict rows | target-support rows | TP-conflict support | actions | FP corrections | TP vetoes | precision |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | K=1 | 28 | 32 | 0.292 | 21 | 4 | 17 | 0.190 |
  | K=2 | 45 | 45 | 0.750 | 14 | 4 | 10 | 0.286 |
  | K=3 | 49 | 45 | 0.773 | 14 | 4 | 10 | 0.286 |

- **Behavior at K=3:** TP/FP/TN/FN becomes 44/1/59/16. FPR improves 0.0833 ->
  0.0167, but TPR collapses 0.9000 -> **0.7333**; balanced accuracy falls to
  **0.8583** and MCC to **0.7402**. Every recall/precision/aggregate gate fails.
  The third proposal adds four conflict rows and no target-support row.
- **Failure decomposition:** five of the ten vetoed true targets have an exact
  target alias in at least one regional target-free open inventory (`wine glass`,
  `umbrella`, `bench`, `surfboard`, or `backpack`), and eight have positive target
  keep margin somewhere. Thus at least half are not simple target-candidate
  recall failures. Ten rows have a target-support direction against one candidate
  and a conflict direction against another candidate in the same region. A coarse
  attention rectangle contains multiple objects, so candidate-relative pairwise
  answers do not form one consistent region identity.
- **False-positive audit:** all five baseline FPs form a conflict; four are
  corrected. One absent backpack receives false regional support against `wall`
  and is suppressed. Mirroring the directional signs therefore does not make
  positive support intrinsically grounded.
- **Decision:** reject residual multi-region attention coverage and do not tune
  K, mass, region size, or pairwise thresholds. Multi-region search partly fixes
  coverage but not proposal purity or verifier consistency. The next iteration
  must obtain target-focused or object-pure proposals, with proposal confidence
  still excluded from the final decision.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter11_multiregion_dev_shard{0,1}.jsonl` and
  `experiments/qwen3vl_xev/iter11_multiregion_dev_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 12: grounded re-identification is precise but low-recall

- **Pre-registered stage 1:** the 14 iteration-11 final actions, deliberately
  selected before inference: 10 true-positive vetoes and 4 false-positive
  corrections. Ask Qwen3-VL to locate each queried category using its official
  normalized 0--1000 JSON grounding format, parse at most three boxes, and ignore
  generated labels. On each exact clear-box/full-canvas view, require target-free
  alias recovery, target margin `>0`, and no strict directional conflict against
  up to three positive-keep open/closed C/N candidates. Grounding output alone is
  never support or absence evidence.
- **Cycle funnel:** valid grounding box **14/14**; positive regional target margin
  11/14; target-free target recovery 5/14; any directional conflict 11/14; full
  cycle support **2/14**. The two supports are real `wine glass` and `umbrella`
  targets. None of the four absent targets receives support, giving support
  precision 1.0.
- **Gate:** FAIL. The cycle suppresses only **2/10** prior true-positive vetoes
  versus the required 8/10. It retains all 4 false-positive corrections but also
  leaves 8 TP vetoes. The 49-conflict-row stage 2 is therefore not run.
- **Failure cases:** target-free recognition names `bench` and `backpack` in
  their grounded boxes and their target margins are positive, but an overlapping
  `person` produces a strict N-relation conflict. For real `bowl`, recognition
  names the target but the target margin is negative. Other true targets fail
  target-free re-identification even when the generated box is plausible.
- **Oracle coordinate audit (diagnostic only):** eight of the ten real targets
  have matching COCO instance boxes. Their maximum generated-box IoU has mean
  **0.607** and median **0.576** (range 0.224--0.964). Grounding coordinates are
  therefore not the only failure. The exact clear-box rendering preserves the
  full image canvas, so small grounded objects such as cell phone remain small
  after preprocessing rather than being zoomed.
- **Decision:** reject the full-canvas grounded cycle without expanding boxes,
  changing K, relaxing C/N conflicts, or extending to stage 2. Retain native
  grounding strictly as a proposal lead. Next test exact box crop-and-zoom with
  target-free re-identification, and treat only C (not overlapping N identities
  such as person/hand) as an exclusive-box contradiction.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter12_grounded_cycle_actions.jsonl` and
  `experiments/qwen3vl_xev/iter12_grounded_cycle_actions_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 13: exact crop improves support but remains incomplete

- **Pre-registered paired setup:** reuse the 14 iteration-12 action rows and
  their parsed pixel boxes exactly. Do not rerun grounding or expand/select boxes.
  Replace the full-canvas clear-box view with the exact crop, and restrict
  mutually exclusive pairwise conflicts to C alternatives. Retain target-free
  alias recovery, positive target margin, and natural zero thresholds.
- **Paired funnel (full canvas -> exact crop):** valid box 14 -> 14;
  target-free target recovery 5 -> 4; positive target margin 11 -> 9;
  directional conflict 11 -> 3 (with the frozen C-only relation gate); full
  cycle support **2 -> 4**.
- **Support behavior:** exact crop supports four real targets: `wine glass`,
  `bench`, `cell phone`, and `backpack`. It adds the latter three but loses the
  previous `umbrella` support. All four absent targets remain unsupported;
  support precision is **1.0** and all four prior false-positive corrections are
  retained.
- **Gate:** FAIL. The verifier suppresses only **4/10** prior true-positive
  vetoes versus the required 8/10, leaving six TP vetoes. No broader development
  replay is run.
- **Scale audit:** exact crop sizes range from 8x11 to 640x480 source pixels.
  The lost `umbrella` box is 19x7 and target-free recognition changes from
  `umbrella` on the full-canvas clear view to `cloud` on the exact crop. The
  unrecovered `surfboard` is 43x12. Yet a 15x12 `cell phone` crop succeeds, so
  crop resolution is not a sufficient explanation; object identity also depends
  on surrounding context and box alignment.
- **Decision:** reject exact crop as the sole verifier view, without adding
  padding or changing scales on these cases. Retain its three new true-target
  recoveries and the full-canvas `umbrella` recovery as evidence that the views
  are complementary. Next test a frozen geometric context pyramid rather than
  choosing one scale post hoc.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter13_grounded_crop_actions.jsonl` and
  `experiments/qwen3vl_xev/iter13_grounded_crop_actions_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 14: repeated multiscale alternative has zero coverage

- **Fresh development setup:** POPE-COCO adversarial rows 2640--2759, 120
  balanced rows / 20 images, untouched when the protocol was committed. Re-run
  frozen no-RoPE K=3 local conflicts, then ground every conflict target and
  evaluate unique centered crops at fixed scales `{1,2,4}`.
- **Baseline / proposal funnel:** baseline `TP/FP/TN/FN = 45/5/55/15`, TPR
  0.7500, FPR 0.0833, balanced accuracy 0.8333, MCC 0.6761. Forty-one rows have
  strict local conflicts. All 41 have valid grounding boxes.
- **Pyramid funnel:** 143 unique crop views (48 scale-1, 48 scale-2, 47 scale-4).
  Target support occurs on 22/29/23 views respectively and protects 29 distinct
  rows. A target-free open C alternative passes strict directional verification
  on 3/2/5 views respectively.
- **Registered certificate:** only **2/41** rows repeat the same alternative in
  at least two distinct scales: real `person -> motorcycle` and real
  `car -> van`, both at scales 2 and 4. Both also have target support and are
  suppressed. The final action count is **0**, so behavior is identical to
  baseline and the development gate fails.
- **Absent-target audit:** absent `car` grounds to a truck and receives a valid
  open `truck` directional conflict at scale 4, but the open name does not repeat
  at another scale. Two absent labels (`person`, `traffic light`) receive visual
  target support. Another absent `person` produces car/trash N identities.
  Absent `couch` produces flower/vase/table inventories without a strict C
  directional switch.
- **Decision:** reject exact alternative-name repetition across scales. It is
  safe but too dependent on lexical candidate stability. Preserve the fixed
  pyramid and target-support guard; next test repeated negative target evidence
  across scales plus one independently verified open C alternative, on a
  separately pre-registered development replay.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter14_multiregion_dev.jsonl`,
  `experiments/qwen3vl_xev/iter14_multiregion_dev_summary.json`,
  `experiments/qwen3vl_xev/iter14_grounded_pyramid_dev.jsonl`, and
  `experiments/qwen3vl_xev/iter14_grounded_pyramid_dev_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 15 stage 1: replicated target-negative certificate passes

- **Pre-registered mechanism setup:** reuse the ten one-region strict
  directional C-conflict rows from the now-development-only iteration-4 window
  (rows 600--1079): seven real targets / three absent targets. Generate fresh
  target grounding and fixed `{1,2,4}` crops. Require negative target margin at
  at least two unique views of one box, one target-free open C strict directional
  alternative at any view of that box, and no cycle-consistent target support
  anywhere.
- **Funnel:** valid grounding 10/10; any-scale target support 6/10; replicated
  target-negative evidence 2/10; open directional alternative 4/10; joint
  certificate **2/10**.
- **Actions:** two absent targets are corrected: `handbag -> backpack` and
  `bicycle -> motorcycle`. Both alternatives are target-free open candidates and
  pass strict reversed-prompt C verification. There are **0 true-positive
  vetoes**, so correction precision is **1.0** and the stage-1 gate passes.
- **Historical 480-row replay:** baseline `TP/FP/TN/FN = 198/20/220/42`
  becomes `198/18/222/42`. TPR is unchanged at 0.8250; FPR falls 0.0833 ->
  0.0750; balanced accuracy rises 0.8708 -> 0.8750; MCC rises 0.7448 -> 0.7538.
  These are development-only numbers because the rows and labels predate the
  rule.
- **Safety audit:** six of seven real conflict targets receive target support.
  Real `truck` receives no target support but also lacks the registered joint
  certificate, so it is not vetoed. The absent `chair` case also remains
  unchanged because it lacks the joint certificate.
- **Decision:** retain the exact rule and advance to a separately committed
  480-row fresh-window protocol. Do not change scales, negative-view count,
  relation gate, aliases, candidate K, or thresholds.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter15_grounded_absence_stage1.jsonl` and
  `experiments/qwen3vl_xev/iter15_grounded_absence_stage1_summary.json`.

### 2026-07-19 — Qwen3-VL XEV iteration 15 stage 2: replicated target evidence fails confirmation

- **Protocol correction before result inspection:** the adversarial split ends
  at index 2999, so the original 480-row range was invalid. The runner was
  stopped after nine rows before inspecting outputs. A committed amendment used
  the actual untouched remainder, indices 2760--2999: 240 balanced rows / 40
  images. The method stayed unchanged; proportional gates required >=2 actions
  and allowed TPR loss <=0.01.
- **Frozen local stage:** all 240 rows / question IDs 2761--3000 complete under
  one no-RoPE proposal. Fourteen rows have strict directional C conflicts: 12
  true targets and 2 absent targets.
- **Grounded funnel:** valid box 14/14; any-scale target support 4/14;
  replicated target-negative evidence 3/14; open C strict alternative 6/14;
  joint certificate **1/14**.
- **Action:** the only certificate vetoes a real `vase` as `pot`/`flowerpot`.
  Its target margin is negative at scales 2 and 4 and the alternatives pass
  strict reversed-prompt verification, but the relation filter labels these
  near-equivalent container identities C rather than S. Neither absent conflict
  (`cup`, `car`) forms the joint certificate. Result: **0 corrections / 1 TP
  veto**, correction precision 0.
- **Behavior:** baseline `TP/FP/TN/FN = 108/9/111/12`, TPR 0.9000, FPR 0.0750,
  balanced accuracy 0.9125, MCC 0.8253. The method gives `107/9/111/13`, TPR
  0.8917, unchanged FPR 0.0750, balanced accuracy 0.9083, MCC 0.8171. The
  confirmatory gate fails decisively.
- **Decision:** reject the replicated target-negative grounded extension. Do not
  add `vase/pot` aliases or retune relation labels on the final adversarial rows.
  XEV v1 remains the only constructive version that passed a held-out gate. Any
  next grounded method must solve semantic equivalence on separate development
  data and use another benchmark/split for evaluation; no untouched POPE-
  adversarial rows remain.
- **Artifacts (git-ignored):**
  `experiments/qwen3vl_xev/iter15_stage2_local.jsonl`,
  `experiments/qwen3vl_xev/iter15_stage2_grounded.jsonl`, and
  `experiments/qwen3vl_xev/iter15_stage2_summary.json`.

### 2026-07-20 — Position-free detector iteration 1, Stage-A extraction smoke

- **Setup:** LLaVA-1.5-7B, fp16 eager attention, Transformers 4.48.1; first 16
  COCO images in the frozen object cache. Read-only pre-hooks capture pre-RoPE
  Q/K at layers 22 and 31. Features are summarized per head for raw, no-RoPE,
  and no-RoPE plus sink-deletion branches. No model module or generated caption
  is changed.
- **Output:** 43/43 object mentions have eight valid nearby non-object rows.
  Each branch has shape `(43, 2, 32, 8)` and contains only finite values. Class
  balance is 35 grounded / 8 hallucinated. Mean detected sink counts are 18.74
  at layer 22 and 16.30 at layer 31.
- **Non-interference gate:** comparing the same first input before and after
  registering the hooks gives max logit error **0** and max ordinary raw-
  attention error **0**. Both no-RoPE and sink deletion produce non-identical
  detector features, so neither branch is an accidental no-op.
- **Legacy-cache diagnostic:** mention positions and token sequences align, but
  86 raw layer-row comparisons against the unversioned May 22 per-head cache do
  not reproduce exactly: max feature errors `[1.0098, 0.3048, 0.1411, 175.5422]`
  for entropy, negative top-1, negative top-5, and negative max/mean. The same-
  runtime identity gate replaces this invalid cross-runtime oracle; all current
  branches will be compared within one extraction run. This amendment was made
  before Stage B and did not alter a score or metric.
- **Decision:** Stage A passes. Do not interpret performance on this 43-row,
  imbalanced smoke; proceed to the locked full extraction and OOF evaluation.
- **Artifact (git-ignored):**
  `experiments/position_free_detector/smoke.npz`.
