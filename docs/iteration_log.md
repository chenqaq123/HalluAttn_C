# Iteration Log

> Append-only. One entry per completed iteration (a decision, pivot, or
> implementation milestone). Newest at the bottom. Each entry records the date,
> what changed, why, and what it implies for the proposal / next steps.

<!-- Template:
## YYYY-MM-DD — <title>
**What changed:** ...
**Why:** ...
**Evidence/refs:** ...
**Implications / next:** ...
-->

## 2026-07-07 — Restart from the two-pronged motivation

**What changed:**
- Created branch `restart-two-pronged` and **tore the project back down to its
  original starting point**: the two-pronged critique that attention-based
  hallucination *detection* and *mitigation* both fail.
- **Kept only the reproducible baselines** and the apparatus needed to reproduce
  the two-pronged critique:
  - Detection: `detection/baselines/` (NLL, Entropy, IC, GLSim, SVAR, PAS,
    Beyond-ADS/CGC) + `detection/src/sinkdetect` + row-cache scripts, under the
    length/position-controlled metrics in `design.md`.
  - Mitigation: `mitigation/baselines/` (vanilla, PAI, ClearSight, VisAttnSink,
    VCD, DAMRO, OPERA, SPIN) + `mitigation/src/{interventions,decoding,data,
    evaluation}.py` + the runner/audit scripts (`run_task.py`,
    `run_parallel_mitigation.sh`, `compare_methods.py`, `merge_evaluate.py`,
    `audit_results.py`, `audit_attention_shift.py`, `run_qwen25vl_pope.py`,
    `check_opera_support.py`).
- **Removed** all prior method superstructure: `mitigation/src/tdev_core.py`, the
  ~44 TDEV/hidden-margin/answer-confidence/CHAIR-intervention/OWLv2/BLINK scripts
  in `mitigation/scripts/`, the TDEV/decode-gate/atomic-detail/caption-edit
  scripts in `detection/scripts/` (+ `detection/scripts/legacy/`), the top-level
  `scripts/` TDEV summary builders, and the method/planning docs
  (`general_tdev_design.md`, `no_external_detector_summary.md`,
  `tdev_detector_positioning.md`, `current_result_baseline_comparison.md`,
  `semantic_neighbor_audit.md`, `aaai2027_paper_plan.md`, `docs/archive/`).
- **Reset the three living docs** to a fresh minimal starting point anchored on
  the two motivations. `docs/experiment_results.md` now holds only §A/§B/§C
  (detection + mitigation baseline reproduction); all TDEV result entries removed.
- **Kept untouched** (user decision): `paper/`, `ref/`, `docs/design.md`,
  `docs/project_structure.md`.

**Why:**
- The accumulated TDEV method family had drifted far from the original,
  clean two-pronged story and had plateaued (POPE MCC ceiling ≈ 0.740 vs OWLv2
  0.763, MCQ negative, CHAIR intervention post-hoc only). The user chose to
  rebuild the constructive method from the two motivations rather than keep
  iterating on the plateaued design.
- The two-pronged critique — reading attention collapses under length control
  (Prong 1); pushing attention only adds yes-bias / backfires (Prong 2) — is the
  durable core and is fully supported by the retained baselines.

**Evidence/refs:**
- Prong 1: `experiment_results.md` §A/§B (`sink_only_conc_top10_mass`
  0.8099 overall → 0.5087 within-bin; PAS/SVAR collapse; IC survives at 0.686).
- Prong 2: `experiment_results.md` §C (ClearSight yes-rate 0.450→0.486, related
  FPR 0.114→0.160, Macro MCC 0.730→0.723; PAI inert).
- Full torn-down history preserved on `main`.

**Implications / next:**
1. Re-confirm the retained baselines still run end-to-end on this branch
   (detection `run_all_baselines.py`; mitigation `run_parallel_mitigation.sh`).
2. Rebuild the constructive method from the two-pronged motivation; do not
   resurrect torn-down TDEV code without re-justifying it.
3. Re-derive the mitigation-side related-present vs. plain-absent FPR
   decomposition (the measurable form of Prong 2) as the first analysis step.
4. Honesty guardrails carried over: Prong 1 is background (HaloProbe pre-empts);
   Prong 2 is scoped to attention *amplification* (PAI is inert).

## 2026-07-08 — Baseline reproduction verified against source papers

**What changed:** Audited every retained detection and mitigation baseline
against its source paper / official code (6 parallel reviews). Recorded verdicts,
the reproduced full-POPE/CHAIR numbers we can stand behind, and the required
fixes in `experiment_results.md` §E (plus fidelity notes on §A/§C).

**Why:** Before rebuilding the method, the two-pronged critique must rest on
reproductions that actually match the papers.

**Evidence/refs:** `experiment_results.md` §E (2026-07-08 entry).
- **Faithful (report as-is):** PAS (line-for-line `lanl/pas`; its length-control
  collapse is genuine), IC (conservative), GLSim, NLL, Entropy; PAI (attn-only,
  documented), VisAttnSink, VCD.
- **Faithful but needs a caveat/relabel:** SVAR is layer-0 mass, not the paper's
  middle-layer SVAR; ClearSight's POPE suppression region ≈1 token
  (enhancement-only on POPE); Beyond-ADS/CGC are simplified unsupervised proxies.
- **Not valid reproductions (bugs / not runnable):** SPIN (prefill masks all
  prompt tokens → always-yes), DAMRO (CLS-attention collapses CLIP's 16 heads,
  `decoding.py:285`), OPERA (delegation-only wrapper, no algorithm code, no
  recorded results, hard-fails on stock transformers).

**Implications / next:**
1. Prong 1 (PAS/IC + sink shape) and Prong 2 (PAI/ClearSight/VisAttnSink/VCD) are
   both reproduced well enough to anchor the paper's motivation.
2. Fix-or-relabel backlog: (a) relabel SVAR as "layer-0 visual attention mass" or
   implement the middle-layer sum; (b) note ClearSight's POPE suppression scope;
   (c) footnote IC's single-layer readout. These are cheap and paper-facing.
3. Repair backlog if these baselines are to be reported: SPIN prefill scope
   (`interventions.py:168-189`), DAMRO CLS attention (`decoding.py:285`), OPERA
   official-fork wiring + `key_position` verification. Otherwise drop them from
   the reportable set (they are not central to the amplification thesis).

## 2026-07-08 — Fixed SPIN + DAMRO, dropped OPERA

**What changed:** Acted on the fidelity audit. Fixed the two local bugs and
removed OPERA. Code + docs only; no re-run yet.
- SPIN (`interventions.py` `_spin_head_mask`): gate only the last query position
  in prefill (+ decoded tokens), truncation head count — unit-tested.
- DAMRO (`decoding.py` `_select_damro_outlier_features`): CLIP 16-head CLS
  attention with per-head scaling, averaged over heads — unit-tested (selection
  changes vs the buggy version).
- OPERA: deleted `generate_opera_beam`, `check_opera_support.py`,
  `baselines/opera/`, and all run wiring. Not reproducible without the fork.

**Why:** User chose "fix SPIN + DAMRO, drop OPERA." SPIN/DAMRO bugs were tractable
local fixes; OPERA needs an external fork and is peripheral to the amplification
thesis.

**Evidence/refs:** `experiment_results.md` §E (2026-07-08 fix entry). Fixes
validated with dummy-tensor unit tests; `py_compile` + `bash -n` clean.

**Implications / next:**
1. **Re-run SPIN and DAMRO** on POPE (and CHAIR) to get valid numbers; only then
   add them to the reportable table. Until re-run, report only vanilla / PAI /
   ClearSight / VisAttnSink / VCD.
2. Paper-facing relabels still pending (SVAR layer-0, ClearSight POPE suppression,
   IC single-layer) — recorded as caveats, applied as doc relabels only.

## 2026-07-08 — Fixed SVAR + Beyond-ADS/CGC per the papers and re-ran on full COCO

**What changed:** Reimplemented SVAR and Beyond-ADS/CGC faithfully and re-ran on
the full 16426-mention detection set. Added `detection/scripts/extract_beyond_svar_features.py`
(GPU per-layer feature extraction) and `evaluate_beyond_svar.py` (offline scoring).
Updated `experiment_results.md` §A + §E and `detection/docs/results_summary.md`.

**Why:** The prior SVAR was layer-0 (not Jiang et al.'s middle-layer sum) and the
Beyond scores were simplified unsupervised proxies. The user asked to fix them to
the papers and report the corrected numbers.

**Evidence/refs:** `experiment_results.md` §E (2026-07-08 fix+rerun entry).
Pipeline validated by reproducing the reference layer-0 SVAR exactly (0.8336).
- SVAR 5–18: 0.805 / within 0.568 — still collapses under length control.
- ADS blob-mass: 0.688 / 0.554 — still weak within-bin (attention-based).
- CGC sim-topk: 0.872 / **0.752** — survives the control (semantic).
- ADS+CGC supervised OOF: 0.890 / **0.793** — strongest controlled detector.

**Implications / next:**
1. **Prong 1 is sharpened, not weakened:** attention-mass/shape detection (PAS,
   SVAR, ADS) collapses under length control even when faithfully implemented;
   what survives is grounding/semantic consistency (CGC, IC, supervised ADS+CGC).
   Frame CGC/ADS+CGC as *representation* detectors, distinct from the attention
   critique.
2. Env note recorded: COCO images moved to `/srv/common_dataset/coco-2014-dataset`;
   update `.env` COCO_PATH if the detection baselines are re-run.
3. Remaining relabels (ClearSight POPE suppression, IC single-layer) unchanged.

## 2026-07-08 — Official-repo audit + three new 2026 attention detectors (SAE/LaSCD/TVER)

**What changed:**
- Diffed the fixed detection scores against the **official GitHub repos** (not just
  the papers). Found the "CGC survives length control (0.752)" result was an
  indexing artifact: we read the object hidden state at `token_pos+1`, but the
  official `token-grounding-detector` reads the *generating* row (`token_pos`).
  Fixed → faithful CGC within-bin **0.752 → 0.574** (it also collapses).
- Confirmed **single-layer ADS ≈ 0.5 is by design** (the paper's ADS is a supervised
  per-layer-*vector* classifier, no length control; our supervised overall 0.865 ≈
  paper 0.88), so it is not a reproduction bug.
- Ran a 4-venue proceedings sweep (CVPR/ICCV/ECCV, ACL/EMNLP/NAACL, NeurIPS/ICML/ICLR,
  surveys+citation graph). Field is **saturated**; only one clean new unsupervised
  attention detector surfaced (**TVER**, ICCV 2025).
- Added three new faithful 2026 attention detectors to the length-controlled table:
  **SAE** (arXiv:2603.16558, true version with Mask2Former-Swin-L segmentation; seg-free
  kept as ablation), **LaSCD** (arXiv:2605.11559, `lascd_raw` = official `_deco_layer_energy`),
  **TVER** (arXiv:2507.00898, ONLY, layer-0 `H(text)/H(vis)`). Plus supervised
  ADS-only / CGC-only as the faithful multi-layer Prong-1 test.

**Why:** the user asked to (a) explain the ADS≈0.5 result against the repo and
(b) broaden the comparison with published, unsupervised, attention-based detectors,
reproducing each faithfully from paper + code.

**Evidence/refs:** `experiment_results.md` §A (updated table) + §E (2026-07-08
repo-audit entry). Repos: `tuandung2812alt3/token-grounding-detector`,
`macovaseas/LaSCD`, `zifuwan/ONLY`. Artifacts `experiments/beyond_svar_v3/`.

**Key numbers (within-bin AUROC):** TVER 0.566 (overall 0.830 = position baseline),
SAE-true 0.575, LaSCD 0.500 (chance), CGC 0.574; supervised ADS 0.664 / CGC 0.669 /
ADS+CGC 0.719; IC 0.686.

**Implications / next:**
1. **Prong 1 is airtight and reproduction-faithful:** every unsupervised single-score
   attention detector collapses under length control — PAS, SVAR, ADS, **SAE (even
   with its real segmenter)**, **LaSCD (chance)**, **TVER (overall = position baseline)**.
2. **Reframe the "survivor" claim:** it is NOT "grounding survives, attention dies"
   (the faithful CGC also collapses). What retains within-bin signal is **supervised
   multi-layer pooling** (≤0.719) and the single logit-lens **IC** (0.686). Update
   proposal §positioning accordingly.
3. New dependency: Mask2Former-Swin-L at `experiments/mask2former_swinL/` (git-ignored,
   for the true SAE). New feature/eval keys: `sae_seg`, `lascd_raw/norm`, `tver`.
4. Re-sweep ACL 2026 once camera-ready posts (not public at search time).

## 2026-07-08 — LaSCD review: why it's near-chance as a detector (off-label, not a bug)

**What changed:** Reviewed LaSCD against `macovaseas/LaSCD` + paper after it scored
near-chance. Confirmed our `_deco_layer_energy` reproduction is verbatim-faithful;
the weakness is a **fundamental usage mismatch**, and corrected how we report it.

**Findings (repo + paper, quoted in §E):**
- LaSCD's energy `E^(l)` is a **within-token, cross-layer selector**: per decoding
  step it takes `argmax_l E` = premature/hallucinating layer and `argmin_l E` = GT
  layer, feeding the contrastive-decoding logit remap (Eq. 4→7). The repo never
  thresholds or compares E across tokens; the paper reports **no per-sample AUROC**
  (only Fig. 3 layer-level evidence).
- The implemented **raw** energy is mass-weighted (Eq. 3). Cross-token it becomes a
  **visual-attention-mass proxy that anti-correlates** with hallucination (grounded
  > hallu; per-layer overall AUROC 0.2–0.4). The paper explicitly concedes (App. A)
  the score "combines the spatial-shape term with the visual-attention mass" and uses
  Theorem 1 only as a geometric explanation of the *shape* component at fixed mass.
- The fair test (mass-**normalized** shape energy, LaSCD's own band L11–28) has the
  correct sign but is near-chance: mean 0.559/0.522, best layer 0.589/0.569.

**Why:** the user asked to review LaSCD's poor result against paper + GitHub.

**Change applied:** `evaluate_beyond_svar.py` now reports LaSCD as the mass-normalized
shape energy (mean L11–28 + peak-min gap + best layer) and flags the raw energy as a
mass proxy. Docs §A/§E + `results_summary.md` updated. No re-extraction needed
(`lascd_norm` already saved).

**Implication:** LaSCD stays in the Prong-1 "attention collapses" set, but the honest
framing is "off-label as a detector" — its energy was only ever validated as a layer
diagnostic / decoding driver, so near-chance detection is expected, not a repro error.

## 2026-07-13 — H3 qualitative attention figures; case 1 fails the residual-concentration criterion

**What changed:** Added `mitigation/scripts/visualize_qualitative_attention.py`,
which renders the H3 qualitative cases (object co-occurrence spec §5.4) offline
from the saved npz artifacts — five-panel layout (image+bbox / vanilla
localization / ClearSight localization on a shared scale / positive matched
residual / measurements), plus per-layer (9–14) supplements and an exact-values
sidecar. Overlay style follows `MM2026/Hallu_Sink_Correlation/show_attention_map.ipynb`
(24×24 patch map, nearest-upsampled, semi-transparent over the processed image),
with `magma` + value-modulated opacity + p99.5-capped scales instead of jet.
Also wrote the spec-required result `README.md` that was missing.

**Why:** user request — visualize sample-case attention distributions for the
mitigation experiment following the MM2026 notebook example.

**Evidence/refs:** `experiment_results.md` §E (2026-07-13);
`mitigation/results/object_cooccurrence_prior/qualitative_cases_attention.{pdf,png}`.

**Implications / next:**
1. The figure makes the saved measurements visible and exposes a selection
   problem: **case 2 (truck/car) fits the H3 mechanism (97.3% of the positive
   residual in-region) but case 1 (chair/dining-table) does not (2.3%
   in-region; related-region mass decreases under ClearSight)**. Re-select
   case 1 from the saved 72-candidate pool or present it under the spec §5.1
   fallback framing.
2. H2 (paired difference-in-differences + bootstrap) remains the pending
   quantitative half of the co-occurrence experiment.

## 2026-07-14 — Clean-map visualization (position prior + sink tokens removed)

**What changed:** Per the user's clarified request, the qualitative attention
visualization now features a **clean** attention map — RoPE position prior removed
(no-RoPE recomputation) and detected sink tokens deleted — reproducing the
sink/position machinery of `MM2026/Hallu_Sink_Correlation/utils/AttnAdapter.py`.
Added `mitigation/scripts/extract_qualitative_sink_attention.py` (a fresh vanilla
forward that captures pre-RoPE key norms, detected sinks, no-RoPE and purified
attention; reuses the repo's pure-torch `sink_utils` and asserts a faithfulness
gate against the saved `vanilla_attention`), and extended
`visualize_qualitative_attention.py` with `render_clean_figure`.

**Why:** user request — "用去掉位置先验与sink token的干净的attention map进行可视化".

**Evidence/refs:** `experiment_results.md` §E (2026-07-14);
`mitigation/results/object_cooccurrence_prior/qualitative_cases_clean_attention.{pdf,png}`
and its README section.

**Implications / next:**
1. On the clean map, case 2 (truck/car) shows the related-object mass fraction
   rising raw→no-RoPE→clean (0.318→0.519→0.573), i.e. removing position prior +
   sinks concentrates attention on the semantic neighbour — a cleaner Prong-2
   illustration than raw attention. Case 1 stays flat (unchanged selection caveat).
2. Env/repro note captured: only `latentGuard` (transformers 4.48.1) is usable
   here, so the extraction is deliberately self-contained (no `DetectionAdapter`
   import, which needs tfm 5.x); model cache is `/srv/common_model/huggingface`.
3. H2 quantitative half still pending.

## 2026-07-14 — Consolidated the qualitative visualization to a clean 3-image triptych

**What changed:** Per the user's request to cut complexity ("我只需要三张图"),
the qualitative attention visualization is now a single **clean** map (RoPE
position prior removed + detected sink tokens deleted) rendered as three images
per case — original / attention map / overlay — by
`mitigation/scripts/visualize_clean_triptych.py`. Removed the earlier
non-clean visualizer `mitigation/scripts/visualize_qualitative_attention.py` and
its result files (`qualitative_cases.{pdf,png}`,
`qualitative_cases_attention.{pdf,png}` + data json, per-layer supplements), and
also the multi-panel clean analysis figure `qualitative_cases_clean_attention.*`
it produced. Kept the clean pipeline: `extract_qualitative_sink_attention.py`
(clean/sink data) + the triptych renderer + the base/sink npz.

**Why:** user request — delete the previous non-clean attention-map visualizations
and keep only the clean 3-image output.

**Evidence/refs:** `mitigation/results/object_cooccurrence_prior/README.md`
(rewritten to match) and `qualitative_case_{1,2}_clean*.png`.

**Note:** the underlying data/faithfulness (2026-07-14 clean-map entry above) is
unchanged; only the figure surface was simplified. The multi-panel clean analysis
figure can be regenerated from the sink npz if ever needed (git history holds the
removed script).

## 2026-07-14 — Clean-map case gallery for selection (24 bare triptychs)

**What changed:** Added `mitigation/scripts/generate_case_triptychs.py`, which
rebuilds the candidate pool from source (POPE-adversarial related-present
negatives with vanilla=no & ClearSight=yes — 76 in total; ranked by best
related-present co-occurrence rank asc, then qid) and renders one **bare** figure
per case — `[ original | clean attention map | overlay ]` horizontally
concatenated, nothing else (no titles/axes/colorbars/boxes). Clean map = no-RoPE +
sink-deleted, layers 9–14, first-answer-token row. Output:
`mitigation/results/object_cooccurrence_prior/case_gallery/` (24 PNGs +
`gallery_manifest.csv`).

**Why:** user asked for more cases to choose from, each figure containing only the
image triptych.

**Provenance recovered (answering "where do the examples come from"):** the two
original cases (qid 1788 chair/dining-table, 2728 truck/car) are POPE-adversarial
`negative_related_present` negatives; answers from
`coco_llava_7b_attention_only/pope/adversarial/{vanilla,clearsight}/predictions.jsonl`,
rows from `semantic_neighbor_audit/adversarial_semantic_neighbor_rows.csv`,
co-occurrence from `semantic_neighbor_audit/cooccurrence_neighbors.json`. The new
generator makes this pool reproducible (the original selection script was lost).

**Next:** user picks cases from the gallery; chosen ones become the H3 figure set.

## 2026-07-14 — Pinned the 14 selected cases into a reproducible regenerator

**What changed:** The user selected 14 cases from the gallery; the other 86 PNGs
were deleted. Added `mitigation/scripts/regenerate_selected_cases.py` — a
self-contained script that pins those 14 (split, qid, image_id, target, question)
and rebuilds them into `case_gallery/` with identical filenames/format/location.
All color/appearance knobs are in a single STYLE block at the top of the file
(colormap, overlay alpha + gamma curve, scale-cap percentile, separator px/color,
optional map background) so figure styling can be tweaked without touching the
attention compute. Verified: reproduces the kept gallery to within ±1 gray level
on <0.01% of pixels (GPU float16 nondeterminism only).

**Why:** user wants a one-command regenerator for the final 14 and will adjust
figure colors themselves.

**Refs:** `case_gallery/gallery_manifest.csv` (the 14 cases, all vanilla=no →
ClearSight=yes semantic-neighbor flips across adversarial/popular/random).

**Update (same day):** split `regenerate_selected_cases.py` into a cache + render
pipeline so figure re-styling needs no GPU. First run computes the 14 clean maps +
processed images once and caches them to
`case_gallery/selected_cases_cache.npz` (torch/model imported lazily only then);
every subsequent run reads the cache and re-renders in ~2.5 s with no model loaded
(RSS ~130 MB). `--refresh` recomputes the cache. All appearance is in the STYLE
block, so a color change is just an edit + re-run. Verified the cache-hit render
reproduces the gallery and that a cmap/separator change takes effect from cache
alone.

## 2026-07-14 — HCVR/LCVR amplification: reproduced Fig. 4, found the asymmetry

**What changed:** Added `mitigation/scripts/hcvr_lcvr_intervention.py` (cache +
render, GPU once then restyle offline) to test the paper's HCVR↑→"yes" /
LCVR↑→"no" claim on case 1268. Regions are the top/bottom quantile of the CLEAN
map (no-RoPE + sinks removed, sinks excluded); intervention is a **post-softmax,
no-renorm** multiply of the region's attention columns via a gated monkeypatch of
tfm-4.48 `eager_attention_forward`. Ran two quantile splits (25/25, 10/50).

**Finding:** HCVR amplification pushes P(yes) ~+30% (robust); LCVR amplification
moves P(no) only ~3–4% and reverses to yes-bias at high factors. Both regions,
pushed hard, add yes-bias — the paper's symmetry fails in magnitude. **top 25% /
bottom 25% is the better split** (equal HCVR effect, cleaner+larger LCVR trend,
balanced region sizes, matches the paper). This is a clean Prong-2 datapoint:
*pushing* attention injects answer bias rather than verifying target evidence.

**Figure style (per user):** all 4 figures are bare — no titles, tight margins;
fig1/fig2 render the region as a **binary** highlight (every in-region patch = 1);
fig3/fig4 **share one x-axis** with equal intervals whose ticks end in 0/5.

**Why:** the user is building the Prong-2 "pushing attention backfires / only adds
yes-bias" evidence; this single-case reproduction quantifies exactly that.

**Refs:** `docs/experiment_results.md` 2026-07-14 HCVR/LCVR entry; artifacts under
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_{t25_b25,t10_b50}/`.

## 2026-07-14 — HCVR/LCVR: switched region definition to cumulative-score (mass)

**What changed:** Added `--select_by {count,mass}` to `hcvr_lcvr_intervention.py`.
Previously HCVR/LCVR were the top/bottom fraction of *tokens*; `mass` makes them the
smallest high/low token sets whose cumulative clean-attention *score* reaches the
fraction. Output dirs are now tagged with the mode (`…_count`, `…_mass`).

**Why (user's point, correct):** on a spike-ish map, top-25%-of-tokens already holds
47.7% of the mass while bottom-25%-of-tokens holds only 10.8%, so a same-factor
amplification injects ~4.4× more absolute mass into HCVR — confounding the HCVR≫LCVR
asymmetry with "HCVR just had more mass." Mass-based equalizes both at 25.1% mass; the
asymmetry survives unchanged (HCVR +31% yes vs LCVR +4% no), so it is genuinely about
credibility/location, not injected mass. Adopt `--select_by mass` for the paper figure.

**Refs:** `docs/experiment_results.md` 2026-07-14 region-definition entry;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Restyled the mass-based HCVR/LCVR figures

**What changed:** Updated `mitigation/scripts/hcvr_lcvr_intervention.py` so the
binary HCVR/LCVR overlays in fig1/fig2 use deep blue (`#08306b`), and removed the
x/y axis titles from the fig3/fig4 amplification plots while retaining ticks,
axes, curves, and legends. Re-rendered the PNG/PDF artifacts from the existing
mass-based cache; no model inference or attention recomputation was performed.

**Why:** User requested a cleaner paper-facing visual style.

**Refs:** `mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Unified HCVR/LCVR figure typography

**What changed:** Set the Matplotlib font family for all four mass-based
HCVR/LCVR figures to the locally installed `Comic Sans MS`, then re-rendered the
PNG/PDF artifacts from the existing cache without loading the model.

**Why:** User requested consistent Comic Sans MS typography across the figure set.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Audited ClearSight's paired prediction shift on POPE adversarial

**What changed:** Paired all 3000 vanilla and ClearSight predictions by
`question_id` and recorded the answer distribution, transition matrix, and
label-conditioned outcome changes in `experiment_results.md`.

**Why:** User requested the before/after prediction distribution rather than
aggregate metrics alone.

**Evidence:** ClearSight changes 136 predictions, all in the no→yes direction:
53 false negatives are corrected, while 83 true negatives become false positives.
Yes-rate rises 47.93%→52.47%, while accuracy falls 83.27%→82.27%.

**Refs:** `mitigation/results/coco_llava_7b_attention_only/pope/adversarial/{vanilla,clearsight}/predictions.jsonl`.

## 2026-07-15 — Set HCVR/LCVR overlay opacity to 0.80

**What changed:** Increased the fig1/fig2 navy overlay opacity from `0.65` to
`0.80` and re-rendered the mass-based figures from the existing cache.

**Why:** User requested an approximately 0.8 opacity for a more solid overlay.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Increased HCVR/LCVR overlay solidity

**What changed:** Corrected the styling direction and raised the fig1/fig2 navy
overlay opacity from `0.25` to `0.65`, making the selected regions substantially
more solid while retaining some source-image detail. Re-rendered from cache only.

**Why:** User clarified that the desired effect was a more solid, less transparent overlay.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Further reduced HCVR/LCVR overlay opacity

**What changed:** Reduced the fig1/fig2 navy overlay opacity from `0.35` to
`0.25` and re-rendered from the existing mass-based cache without model loading.

**Why:** User requested a further reduction in opacity.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Reduced HCVR/LCVR overlay opacity

**What changed:** Reduced the fig1/fig2 navy region-overlay opacity from `0.50`
to `0.35`, retaining `#001f3f` and all other styling. Re-rendered from the
existing mass-based cache without model loading.

**Why:** User requested a lighter, less opaque overlay so the source image remains clearer.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-15 — Darkened the HCVR/LCVR region overlays

**What changed:** Darkened the binary region-overlay color in fig1/fig2 from
`#08306b` to navy `#001f3f`, retaining the existing opacity and all other figure
styling. Re-rendered from the existing mass-based cache without model loading.

**Why:** User requested a darker blue overlay.

**Refs:** `mitigation/scripts/hcvr_lcvr_intervention.py`;
`mitigation/results/object_cooccurrence_prior/hcvr_lcvr_1268_t25_b25_mass/`.

## 2026-07-19 — XEV iteration 1 on Qwen3-VL: rejected symmetric keep/drop evidence

**What changed:** Established the first constructive-method implementation after
the restart. Added `mitigation/src/xev.py` (POPE target parsing, cumulative-mass
region proposals, matched keep/drop and random views, candidate/relation parsing,
relative scores), `run_qwen3vl_xev_pope.py` (Qwen3-VL first-answer attention-row
capture, COCO knowledge cache, closed/open/oracle candidates, S/C/N relation
filter, resumable JSONL shards), `analyze_qwen3vl_xev_pope.py`, and focused unit
tests. Copied `Qwen3-VL-8B-Instruct` to `/home/chenguanxu/models/` and used an
isolated Transformers-4.57.6 environment so the LLaVA Transformers-4.48.1 setup
was not modified. Fixed a smoke-test failure mode by chunking relation prompts to
8 candidates (the original 12-label response hit the 32-token cap) and added a
conservative rule for target-as-modifier phrases such as `snowboard boot`.

**Experiment:** Ran a two-GPU, 120-row balanced POPE-COCO adversarial pilot (20
images, 60 yes/60 no, one attention proposal). Full-image Qwen3 is strong (AUROC
0.9346, balanced accuracy 0.8833). The planned
`E=m(keep)-m(drop)` target score reverses the labels (AUROC 0.3432; random control
0.1808), and oracle candidates do not rescue it (0.3483). Saved-margin analysis
shows `m(keep)` alone is 0.8383 and `m(full)-m(drop)` is 0.7314: subtracting the
complement mixes two views with very different retained context and creates the
reversal.

**Decision / proposal change:** Formally reject symmetric keep-minus-drop as
XEV's evidence definition. Preserve the narrower hypothesis that a *locally
dominant* mutually exclusive alternative is useful: a post-hoc same-view
dominance gate lowers related-present FPR at matched 80% TPR from 0.50 to 0.25
with closed/oracle candidates, but loses overall AUROC and does not improve
held-out behavior. Update `docs/proposal.md` accordingly. The next iteration
must fix context-matched region scoring and proposal recall, using closed/oracle
candidates as diagnostic ceilings. Freeze open-vocabulary expansion until that
ceiling improves overall held-out behavior; current open coverage of oracle
related alternatives is only 31.25% vs 81.25% for closed candidates.

**Evidence:** `docs/experiment_results.md` 2026-07-19 entry; generated artifacts
under git-ignored `experiments/qwen3vl_xev/`.

## 2026-07-19 — XEV iteration 2: rejected exact candidate-agreement certificate

**What changed:** Added a pure identity-switch certificate and unit tests, stored
region/question certificates in the Qwen3 runner, added `--start` for disjoint
contiguous evaluation windows, and extended the analyzer with frozen holdout
success gates and correction diagnostics.

**Experiment:** Fixed the rule and margin threshold on iteration 1, then ran a
confirmatory 480-row / 80-image POPE-adversarial window starting at row 120. The
baseline reaches AUROC 0.9480 and balanced accuracy 0.8688. The strict
certificate produces zero triggers, so it changes neither the 0.0625 FPR nor the
0.8000 TPR and fails the pre-registered minimum-trigger gate. Target keep margin
still reaches AUROC 0.8815, isolating the failure to candidate confirmation.

**Decision / proposal change:** Reject exact lexical intersection between open
and closed candidate inventories as a verification certificate. It is too sparse
to be usable. Keep candidate acquisition separate from verification and move to
an explicit same-view, pairwise target-vs-alternative visual judgment. The
iteration-2 holdout is closed to further threshold selection; the next rule must
be selected on iteration 1 and evaluated on a new disjoint window.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-2 entry;
`experiments/qwen3vl_xev/iter2_holdout_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 3: froze counterbalanced pairwise certificate

**What changed:** Added target-free open/closed candidate ranking, reversed
target-vs-alternative prompts on the same clear-region view, a counterbalanced
preference that exposes common yes-bias, and a pure pairwise identity-certificate
function. The runner stores directional margins, candidate provenance, relation,
preference, common bias, and certificates; the analyzer reports both the unsafe
raw pairwise rule and the gated behavior. Added `--pairwise_top_k` and explicit
development/confirmatory analysis status, plus focused tests.

**Experiment:** Re-ran the 120-row iteration-1 set as development data with
top-k=3. Raw pairwise preference below zero vetoes 13 true positives for only 2
false-positive corrections. Requiring the target keep margin below zero and a C
relation reduces this to two triggers, both correct (`dining table -> chair` and
`handbag -> backpack`). FPR falls 0.0667 to 0.0333 with unchanged 0.8333 TPR;
balanced accuracy rises 0.8833 to 0.9000 and MCC 0.7705 to 0.8072.

**Decision / proposal change:** Promote the conjunction to the first usable XEV
method candidate, but label it development-only. Freeze candidate top-k=3, the
target-in-open support guard, C relation, and all zero-threshold signs. A new
disjoint confirmatory window must meet: >=3 triggers, >=2/3 correction precision,
FPR decrease, TPR loss <=0.005, and non-decreasing MCC. Do not tune on the closed
iteration-2 holdout.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-3 entry;
`experiments/qwen3vl_xev/iter3_dev_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 4: rejected single-region pairwise hard veto

**Experiment:** Ran the frozen iteration-3 rule on a new 480-row / 80-image
POPE-adversarial window (rows 600--1079). Baseline balanced accuracy is 0.8708
with 0.8250 TPR, 0.0833 FPR, and 0.7448 MCC. The certificate triggers six times;
all six veto true positives and none correct false positives. TPR falls to
0.8000, FPR is unchanged, balanced accuracy falls to 0.8583, and MCC to 0.7216.

**Diagnosis:** A target-conditioned attention proposal can land on another real
object while the queried target exists elsewhere. Pairwise comparison then
correctly describes the selected region but cannot establish image-level target
absence. C-relation errors for hypernyms/associated objects add another failure
mode. Conversely, several baseline false positives have good local alternatives
but positive target keep margins, so tightening that margin is not a solution.

**Decision / proposal change:** Reject the single-region hard veto and demote
pairwise preference to local evidence only. The next iteration must add an
independent target-free, full-image support guard before any veto. Audit the six
false triggers and candidate false positives first, then use a new window for any
confirmatory claim.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-4 entry;
`experiments/qwen3vl_xev/iter4_holdout_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 5: froze lazy global-support consensus

**What changed:** Added a robust local-identity proposal separate from the veto,
target-free full-image inventory generation, configurable local candidate
provenance, lazy evaluation only on local proposals, two-prompt inventory
consensus, direct question-id selection for mechanism audits, schema-v3 outputs,
offline diagnostics, and focused tests.

**Mechanism audit:** Tested 10 selected cases spanning five desired corrections
and five previous TP vetoes. With closed-source local proposals and one global
inventory, all TP vetoes are suppressed but two absent targets receive false
support (`bag` for handbag; hallucinated bicycle). The second independently
worded target-free inventory does not repeat either error. Unanimous support
therefore retains all five corrections and suppresses all five TP vetoes. A
full-image pairwise consistency variant was tested and rejected because salience
made it favor the alternative even when both objects were present.

**Decision / proposal change:** Freeze closed candidate provenance, top-k=3, C,
local pairwise sign, two inventories of size 12, and unanimous target-support
consensus. Closed categories are a reliability gate for this pilot, not the
claimed final vocabulary. Rows 0--119 and 600--1079 are now development-only;
evaluate unchanged on rows 1080--1559 with the pre-registered iteration-3 gate.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-5 entry;
`experiments/qwen3vl_xev/iter5_consensus_audit_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 6: rejected generated-inventory absence

**Experiment:** Ran the frozen iteration-5 rule on disjoint rows 1080--1559.
Among 480 rows, 11 local proposals invoke the double-inventory guard, 7 are
suppressed, and all 4 remaining triggers veto true positives. TPR falls 0.7958
to 0.7792, FPR remains 0.0667, balanced accuracy falls 0.8646 to 0.8563, and MCC
falls 0.7362 to 0.7211. The confirmatory gate fails with zero corrections.

**Diagnosis:** Negative counterbalanced preference alone accepts weak ties and
cases where both directional questions answer no or both answer yes. Separately,
the S relation marks object parts/hypernyms (wheel, tire, door, generic bag) as
target support and suppresses genuine baseline-FP corrections.

**Decision / proposal change:** Reject generated-inventory absence as an
image-level certificate. Preserve the target-free inventories only as positive
support. Require the stronger natural directional signs (target-first `<0`,
alternative-first `>0`) and use an explicit COCO alias map instead of generated
S labels. This window is now development data; any next claim needs untouched
rows and a larger sample for sparse triggers.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-6 entry;
`experiments/qwen3vl_xev/iter6_holdout_summary.json` (git-ignored).
