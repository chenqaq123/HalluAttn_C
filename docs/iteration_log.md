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

## 2026-07-19 — XEV iteration 7: froze directional/alias precision repair

**What changed:** Added a strict directional-switch option for local proposals
and an auditable target-alias matcher for target-free inventories. Added CLI and
schema metadata selecting pairwise gate and support matcher, with unit tests.
The alias matcher replaces generated S decisions in the current method and also
removes their extra inference calls.

**Development result:** Offline replay on all 480 iteration-6 rows removes all
four prior TP vetoes and retains two FP corrections (`car -> truck`, `backpack ->
suitcase`). FPR would fall 0.0667 to 0.0583 with unchanged 0.7958 TPR; balanced
accuracy would rise 0.8646 to 0.8688 and MCC 0.7362 to 0.7455. A two-case
end-to-end smoke reproduces the expected reject/retain decisions.

**Decision / proposal change:** Freeze closed provenance, top-k=3, C relation,
strict target-no/alternative-yes directional signs, two target-free inventories,
and unanimous alias support. Because the replay yields only two triggers in 480
rows, use 960 untouched rows 1560--2519 for confirmation while retaining the
minimum-three-trigger gate. All prior windows are development-only.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-7 entry;
`experiments/qwen3vl_xev/iter7_smoke.jsonl` (git-ignored).

## 2026-07-19 — XEV iteration 8: adopted first held-out usable version

**Experiment:** Ran the frozen iteration-7 rule on 960 untouched balanced
POPE-adversarial rows (1560--2519). Baseline TP/FP/TN/FN is 404/37/443/76. Seven
strict local proposals invoke the global guard; three are suppressed and four
act. Three actions correct FPs and one vetoes a TP, yielding 403/34/446/77.

**Result:** The pre-registered gate passes: 4 triggers, 0.75 correction precision,
FPR 0.0771 to 0.0708, TPR 0.8417 to 0.8396 (loss 0.00208), balanced accuracy
0.8823 to 0.8844, and MCC 0.7671 to 0.7719. The action rate is only 0.42%.

**Decision / proposal change:** Adopt the frozen directional/alias certificate as
**XEV v1**, the first usable constructive result after restart. Scope the claim
to sparse high-precision correction, disclose the one couch/chair support-recall
failure, and do not tune inventory length on this holdout. Next work is external
validation across POPE splits and models, not another in-window threshold change.

**Evidence:** `docs/experiment_results.md` 2026-07-19 iteration-8 entry;
`experiments/qwen3vl_xev/iter8_holdout_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 9: removed RoPE from localization and audited sparse coverage

**What changed:** Added a Qwen3-VL localization-only no-RoPE path. The runner
captures pre-RoPE Q/K, recomputes the last-query attention row without modifying
the model forward, defaults `--attention_map` to `no_rope`, retains `raw` for
ablation, records map type under schema v4, and exposes a consolidated analyzer
coverage funnel.

**Experiment:** A faithfulness smoke keeps the qid-2294 baseline margin exactly
0.25 while moving its proposal. On paired POPE-adversarial rows 2520--2639,
no-RoPE changes proposal geometry substantially (42/120 exact boxes, mean IoU
0.628, 15 disjoint), but raw and no-RoPE both yield one strict local proposal,
one global-support suppression, and zero final actions. No-RoPE raises broad
pairwise-scored rows 27 -> 31 and negative preferences 21 -> 25 without improving
the strict stage.

**Decision / proposal change:** Keep no-RoPE as the default localization hygiene
fix, but do not transfer XEV v1's raw-map held-out claim to it. The 4/960 action
rate is caused by the closed provenance + C relation + strict directional gate:
the v1 funnel is 441 baseline Yes -> 242 pairwise -> 176 negative preference -> 7
strict local proposals -> 4 actions. The next iteration must increase local
proposal recall instead of adding another hard gate; retain an independent
image-level target-support check to protect coexisting targets.

**Evidence:** `docs/experiment_results.md` iteration-9 entry;
`experiments/qwen3vl_xev/iter9_{no_rope_dev,raw_control}_summary.json`
(git-ignored).

## 2026-07-19 — XEV iteration 10: rejected ungrounded global-support union

**What changed:** Added a configurable non-support relation gate so strict local
identity switches may use open or closed C/N candidates. Added separate
inventory, region-complement, and union support sources plus analyzer diagnostics
for each. XEV-v1 defaults remain unchanged for reproducibility.

**Experiment:** On development rows 2520--2639, broadening the local gate raises
strict conflicts from 1 to 21. The pre-registered complement-or-inventory union
suppresses 18, but all 3 remaining actions veto true positives. It corrects none
of 5 baseline false positives; balanced accuracy falls 0.9083 to 0.8833 and MCC
falls 0.8168 to 0.7684. Inventory-only produces 1 correction / 9 TP vetoes;
complement-only produces 2 / 4. No threshold was tuned.

**Diagnosis:** The local C restriction was applied at the wrong semantic level,
but relaxing it reveals the real bottleneck. A target-free full-image inventory
is still model generation and can reproduce the queried hallucination from scene
associations, while missing small true targets. A single complement margin is
not localized evidence and has the opposite failures. Their union protects more
false positives than true positives.

**Decision / proposal change:** Reject the union support guard while retaining
the broader gate as a diagnostic. The next method must replace global free-form
support with multi-region grounded verification: attention proposes diverse
regions, counterbalanced visual comparisons establish local identity, and an
image-level veto requires conflict plus no target-supporting region. This follows
the paper's central rule that attention allocation proposes where to inspect but
never constitutes evidence.

**Evidence:** `docs/experiment_results.md` iteration-10 entry;
`experiments/qwen3vl_xev/iter10_local_conflict_dev_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 11: rejected coarse multi-region attention coverage

**What changed:** Added residual diverse region proposals, strict mirrored
regional target-support certificates, and analyzer replay for every K=1/2/3
proposal prefix. Exact target mention in an open regional inventory no longer
short-circuits pairwise verification in regional-support mode. Existing connected
proposal and XEV-v1 paths remain available.

**Experiment:** On development rows 2520--2639, every sample receives three
regions and the baseline exactly reproduces. K=1 makes 21 actions (4 corrections
/ 17 TP vetoes). K=2 raises TP-conflict support from 0.292 to 0.750 and reduces
the result to 4 / 10. K=3 adds four conflict rows but no support or action change.
Final precision is 0.286; TPR falls 0.9000 to 0.7333, balanced accuracy to 0.8583,
and MCC to 0.7402. The pre-registered gate fails.

**Diagnosis:** The failure is partly target coverage and partly region identity.
Five of ten vetoed real targets are already named in a regional open inventory,
and eight have positive keep margin in some region, but strict pairwise support
is absent. Ten rows support the target against one candidate and conflict with it
against another candidate in the same region. Residual attention boxes are
spatially distinct but still multi-object; a binary candidate-relative test has
no unique regional identity to recover.

**Decision / proposal change:** Reject the variant without threshold tuning.
Attention remains admissible as a proposal source, but increasing K cannot be the
constructive contribution. The next proposal must be target-focused or
class-agnostic and object-pure, after which the verifier can be retested. Scores
or confidences from that proposal stage must not become decision evidence.

**Evidence:** `docs/experiment_results.md` iteration-11 entry;
`experiments/qwen3vl_xev/iter11_multiregion_dev_summary.json` (git-ignored).

## 2026-07-19 — XEV iteration 12: rejected full-canvas grounded cycle

**What changed:** Added a strict JSON parser for Qwen3-VL normalized grounding
boxes, a lazy grounded-cycle runner over saved iteration-11 conflicts, and a
stage-aware analyzer. Grounding labels and missing boxes are ignored. Positive
support requires target-free local alias recovery, positive target margin, and
no strict C/N directional conflict.

**Experiment:** All 14 selected action cases return valid boxes. Eleven have
positive target margin, five recover the target without naming it in the prompt,
and only two pass the full cycle. Both supports are real targets and all four
absent targets remain unsupported, but true-target support recall is 2/10. The
stage-1 gate fails, so the planned 49-row extension is not run.

**Diagnosis:** Grounding itself remains promising: on eight true targets with
COCO boxes, mean maximum IoU is 0.607. The rendering discards that advantage by
keeping the exact box on the full canvas, so small objects are not enlarged.
Also, real bench/backpack boxes are rejected by overlapping person conflicts,
showing that image-level N relations should not automatically be exclusive
within a target-grounded box.

**Decision / proposal change:** Reject the iteration-12 cycle unchanged. The
next hypothesis is grounded crop-and-zoom followed by target-free
re-identification, with C-only exclusive contradiction inside the grounded box.
Grounding coordinates, labels, and box presence remain proposal-only signals.

**Evidence:** `docs/experiment_results.md` iteration-12 entry;
`experiments/qwen3vl_xev/iter12_grounded_cycle_actions_summary.json`
(git-ignored).

## 2026-07-19 — XEV iteration 13: exact crop is useful but not sufficient

**What changed:** Reused iteration-12 boxes exactly, presented each as an
independent crop, and restricted exclusive-box contradictions to C relations.
Added reusable saved-box/crop switches and paired funnel reporting while keeping
iteration-12 defaults intact.

**Experiment:** On the same 10 true-target / 4 absent-target mechanism audit,
cycle support rises from 2 to 4. Exact crop newly recovers real `bench`,
`cell phone`, and `backpack`, loses real `umbrella`, and supports no absent
target. Precision remains 1.0, but true-target support recall is only 4/10, so
the stage-1 gate fails.

**Diagnosis:** Crop-and-zoom fixes some small-object verification failures but
can remove the context needed to interpret an under-resolved or imperfect box.
The 19x7 umbrella proposal changes from `umbrella` on the full-canvas view to
`cloud` on the crop, while a 15x12 cell-phone proposal succeeds. Neither exact
crop nor source-pixel area is a sufficient universal view rule.

**Decision / proposal change:** Reject exact crop as the only verifier view.
Retain grounding as proposal-only and move from single-scale verification to a
predefined context pyramid. Evidence must still come from target-free identity,
target margin, and explicit C-alternative checks at the evaluated view; proposal
coordinates themselves remain non-evidential.

**Evidence:** `docs/experiment_results.md` iteration-13 entry;
`experiments/qwen3vl_xev/iter13_grounded_crop_actions_summary.json`
(git-ignored).

## 2026-07-19 — XEV iteration 14: repeated alternative identity is too sparse

**What changed:** Added centered dyadic crop expansion, duplicate-view removal,
and a multiscale certificate requiring one target-free C alternative to pass
strict reversed pairwise verification in at least two scales of the same
grounding box. Target support at any scale blocks the action.

**Experiment:** On fresh rows 2640--2759, frozen K=3 verification produces 41
local-conflict rows. All have grounding boxes and 29 receive target support.
Only two rows repeat an open alternative across scales; both queried targets are
real and both are correctly protected by target support. The final method has
zero actions and fails its coverage gate.

**Diagnosis:** The scale pyramid exposes useful evidence, but target-free
candidate strings are not stable enough to serve as the replicated unit. The
clean absent-car case names `truck` and strictly prefers it only at scale 4;
smaller scales give related closed conflicts or object parts. Conversely, the
guard prevents multiscale motorcycle/van conflicts from vetoing two real targets.

**Decision / proposal change:** Reject repeated alternative naming, not the
grounded pyramid or target-support guard. The next certificate should replicate
negative target evidence across scales and require one target-free C alternative
with a strict directional switch. This shifts replication to a fixed queried
hypothesis while retaining explicit alternative identity verification.

**Evidence:** `docs/experiment_results.md` iteration-14 entry;
`experiments/qwen3vl_xev/iter14_grounded_pyramid_dev_summary.json`
(git-ignored).

## 2026-07-19 — XEV iteration 15 stage 1: replicated target-negative gate passes

**What changed:** Moved the replicated unit from generated alternative names to
the fixed queried target. The new certificate requires at least two negative
target-margin crop views from one grounding box, plus one target-free open C
strict directional alternative from that box. Any-scale target support blocks
the action.

**Experiment:** On ten historical development conflicts, all rows have valid
boxes, six receive target support, and two satisfy the joint certificate. Both
actions correct absent targets (`handbag -> backpack`, `bicycle -> motorcycle`)
with no real-target veto. The pre-registered mechanism gate passes.

**Interpretation:** The positive result addresses iteration 14's lexical
instability without dropping independent alternative verification. Cross-scale
negative target evidence supplies replication; one explicit target-free
alternative supplies visual identity. The guard and full conjunction matter:
neither target-support absence nor one alternative conflict alone is an action.

**Decision / proposal change:** Retain the rule unchanged and pre-register a
fresh 480-row stage 2. Treat the current FPR/MCC improvement as development-only;
the historical rows cannot establish generalization.

**Evidence:** `docs/experiment_results.md` iteration-15 stage-1 entry;
`experiments/qwen3vl_xev/iter15_grounded_absence_stage1_summary.json`
(git-ignored).

## 2026-07-19 — XEV iteration 15 stage-2 protocol boundary correction

**Issue:** The pre-registered stage-2 range 2760--3239 assumed 480 remaining
POPE-adversarial rows, but the split ends at index 2999. The runner reported 240
selected rows. It was interrupted after nine rows before any saved output or
method outcome was inspected.

**Correction:** Supersede the invalid range with the actual final remainder,
indices 2760--2999: 240 balanced rows / 40 images. Keep the model and method
identical. Scale the minimum action count 3 -> 2 and TPR tolerance 0.005 -> 0.01
so one tolerated positive error remains one error when positives change from 240
to 120. Precision, maximum TP-veto count, FPR, balanced-accuracy, and MCC gates
remain unchanged.

**Evidence:** dataset length/label-count validation before result inspection;
`docs/iteration_protocols/xev_iteration_15_stage2_amended.md`.

## 2026-07-19 — XEV iteration 15 stage 2: grounded extension rejected

**Experiment:** The amended final 240-row POPE-adversarial remainder produces 14
strict local conflicts. All receive grounded pyramids, four receive target
support, and one reaches the final joint certificate. That action is a real
`vase`; neither of the two absent conflicts is corrected. TPR, balanced accuracy,
and MCC fall while FPR is unchanged, so the confirmatory gate fails.

**Diagnosis:** The unique failure is semantically coherent but benchmark-wrong:
target-free crops name `pot` and `flowerpot`, and the visual pairwise verifier
strictly prefers them to `vase`. The relation filter labels both as competing C
identities instead of near-equivalent/support identities. Localization and
cross-scale signs cannot repair a faulty object-identity boundary.

**Decision / proposal change:** Reject the replicated target-negative extension
unchanged. Do not add a post-hoc vase/pot alias or alter relation classes on this
closed final window. Retain XEV v1 as the only held-out constructive method. A
future grounded branch requires a separately developed semantic-equivalence
module and evaluation outside POPE adversarial, whose rows are now exhausted.

**Evidence:** `docs/experiment_results.md` iteration-15 stage-2 entry;
`experiments/qwen3vl_xev/iter15_stage2_summary.json` (git-ignored).

## 2026-07-19 — Outer-loop route review: XEV closed, TCS pre-registered

**Evidence review:** Reinterpreted the nominal 960-row XEV held-out result at
the action-count level: four interventions produced three corrections and one
new error, for a net change of only two errors. Subsequent coverage expansions
did not confirm the effect, and the final untouched 240 POPE-adversarial rows
produced one action that changed a true positive to a false negative. This is
not evidence of a useful mitigation method.

**Route diagnosis:** The failure is shared by keep/drop regions, generated
alternatives, full-image inventories, native grounding, exact crops, and
multiscale crops. All ask the same Qwen3-VL model to judge derived views selected
from its own representations. Better localization does not add an independent
object-identity signal, so the loop preserves the co-occurrence prior and
unstable category boundary it is meant to correct. XEV is closed as a route;
there will be no further alias, threshold, crop, proposal, or verifier-prompt
tuning.

**Literature audit:** Checked the rebuilt direction against recent primary work.
R-CoV directly occupies same-model coordinate/region-description verification;
oDPO already uses target-object removal; VES-RFT uses generic image/no-image
sensitivity; Causal-LLaVA and CIPHER cover broader causal deconfounding and
counterfactual representation steering. Therefore none of self-verification,
object removal, visual sensitivity, or causal training alone is a defensible
novelty claim. The dated audit is in
`literature/2026-07-19_constructive_route_pivot.md`.

**Proposal change:** Pivoted the constructive method to Target-Causal
Selectivity (TCS). For a queried target and visible semantic neighbor, training
must increase answer sensitivity to target removal while enforcing invariance to
neighbor removal. A natural related-present negative and matched-background edit
control artifact shortcuts and global `No` bias. Inference is the unmodified
single model forward; masks and inpainting are training-data construction only.

**Next gate:** Pre-registered `docs/iteration_protocols/tcs_iteration_1.md`.
Stage A audits counterfactual validity and frozen-model margins before training.
Stage B compares natural SFT, four-view SFT, target-removal-only, and full TCS on
held-out target categories and POPE-popular. Full TCS is rejected unless its
paired selectivity exceeds both controls while improving balanced accuracy/MCC
without material TPR or yes-rate collapse. No new experimental result was
produced in this route-review iteration.

## 2026-07-19 — TCS cancelled; Null-Evidence Attention pre-registered

**Design rejection:** Cancelled TCS before implementation. No counterfactual
image was generated and no model result was inspected. Four-view binary labels
already impose most of the proposed target-sensitivity/neighbor-invariance
behavior, making it unclear that the paired regularizer adds a mechanism beyond
ordinary counterfactual SFT. Unequal inpainting artifacts and scene changes
further weaken the causal interpretation. The committed TCS protocol is retained
with cancelled status rather than silently deleted.

**Literature refresh:** Hidden-state object verification is now directly covered
by OPV; explicit `<REJ>` plus mask/box grounding is covered by ICLR-2026
PostAlign; no-target grounding is established in Ground-V/GREC; external visual
evidence is covered by MARINE, VEP, and Kestrel; sink-triggered attention
modulation is covered by SAGE. Generic null/no-op attention is also established
by Softmax1/StreamingLLM and related transformer work. The open boundary cannot
be "add a verifier," "predict rejection," or "add a sink token."

**New hypothesis:** Pre-register Null-Evidence Attention (NEA). At selected
Qwen3-VL heads, preserve the original post-RoPE visual mass and text contribution,
but replace the conditional visual distribution with a pre-RoPE content
competition between real visual keys and one learned null key. When null wins, a
norm-bounded null value represents absence evidence. This neither reads raw
attention as verification nor increases visual mass.

**Why it fits the evidence:** Related objects currently win a relative visual
allocation even when the target is absent; attention amplification then turns
that proxy into `Yes`. NEA adds the missing "none of these visual tokens" outcome
inside the existing visual budget. Natural present, related-absent, and random-
absent images provide supervision without inpainting, candidates, boxes,
external detectors, or a second model call.

**Gate:** `docs/iteration_protocols/nea_iteration_1.md` requires unseen-category
null routing to beat query-only and visual-energy-only probes, depend on the
explicit visual-key energy, and fail after counterlabel image swapping before
behavior-changing values are trained. An annotation-only feasibility audit fixed
the neighbor co-occurrence floor at 50 (the original 100 left `bear` and
`hair drier` without a related neighbor) and doubled held-out rows to 40 per
target. Full NEA must then beat zero-null, logit-bias, and parameter-matched
residual-adapter controls on
POPE-popular with at least ten net corrections, lower related-present FPR,
non-decreasing MCC/balanced accuracy, and tightly bounded TPR/yes-rate changes.
Failure closes the route without LoRA or threshold tuning. No new experimental
result was produced in this design iteration.

## 2026-07-19 — NEA cancelled; existential set evidence pre-registered

**Design rejection:** Cancelled NEA before Stage A. No NEA inference, routing
fit, or null-value training was run. Although the null key/value was located
inside an attention head, its behavior still formed an absence-conditioned
correction to the ordinary answer. Its distinction from contrastive or
calibrated decoding was therefore primarily architectural placement rather than
a different decision principle.

**Literature audit:** The replacement search also rejected several nearby
routes. OPPO already optimizes ordered visual-evidence strength and derives local
visual sensitivity; *Same Attention, Different Truths* reads and injects visual-
token logits; CausalLens and query-guided visual-focus methods reinforce selected
patches; Slot-MLLM occupies object-centric tokenization; LPFV verifies sampled
caption candidates. These boundaries are recorded in
`literature/2026-07-19_existential_evidence_redesign.md`.

**New hypothesis:** Replace compulsory relative allocation with
**Existential Evidence Aggregation (EEA)**. A shared open-vocabulary matcher
produces unnormalized target-patch compatibility, and a patch-count-normalized
smooth maximum approximates "at least one patch supports this target." A related
object may win attention while still failing this absolute match. The first
pilot is diagnostic only; any later EEA head is a training objective and is
removed when evaluating ordinary single-forward answers.

**Novelty guardrail:** Multiple-instance learning, smooth-max pooling,
patch-text matching, and weakly supervised localization are prior art. The
intended claim exists only if the same matcher with existential pooling beats
relative softmax, global pooling, and top-attention-patch controls on unseen
targets, then improves ordinary model behavior beyond answer-only SFT.

**Gate:** `docs/iteration_protocols/eea_iteration_0.md` freezes Qwen3-VL and
tests 60 training versus 20 held-out COCO targets. The primary control holds the
image patches fixed while swapping the queried identity between one visible
object and one related absent object. EEA must reach 0.75 macro AUROC, beat every
matched pooling control by 0.05, pass the same-image ordering test, and lose
signal under image/patch swaps. Failure closes the route before LoRA. No new
experimental result was produced in this design iteration.

## 2026-07-20 — EEA cancelled; pair-centered preference alignment pre-registered

**Design rejection:** Cancelled EEA before feature caching or matcher training.
Although its absolute patch-set score avoided attention amplification and
contrastive decoding, the method became a weakly supervised detector whose
claim depended on smooth existential pooling beating established MIL and
patch-text controls. It also did not directly constrain the empirical Prong-2
failure: both present and related-absent answer margins can move toward `Yes`.

**Literature refresh:** Ordinary faithful-versus-hallucinated DPO is established
by multimodal preference methods including HA-DPO/HALVA and CHAIR-DPO. FINER is
the closest data-level overlap: it trains with fine-grained negative queries
whose mismatched and genuinely present elements co-occur. mDPO already diagnoses
image-ignoring preference learning and adds image preference plus reward
anchoring. OPPO orders one faithful answer over strong/original/weak image views.
Therefore neither DPO, semantic hard negatives, same-image co-occurrence,
anchoring, nor evidence-strength ordering is claimed as new. The boundary audit
is in `literature/2026-07-20_pair_centered_preference_redesign.md`.

**New hypothesis:** Pre-register Pair-Centered Preference Alignment (PCPA). One
unchanged natural image supplies a visible target `t+` and a related absent
target `t-`. The preferred joint outcome is `Yes(t+) + No(t-)`; the rejected
outcome flips both assignments. Both sides contain one `Yes` and one `No`, so a
global answer-prior shift cancels from the pair score. A reference-relative
centering penalty constrains the average of the two margin updates to zero, and
a balanced sign anchor prevents a correctly ranked pair from retaining two
positive or two negative margins. Inference remains one ordinary Qwen3-VL pass.

**Gate:** `docs/iteration_protocols/pcpa_iteration_1.md` compares frozen base,
balanced answer SFT, independent DPO, anchor-matched independent DPO, joint
preference without centering, and full PCPA under identical LoRA capacity. The
method must beat SFT and ordinary DPO on 20 unseen target categories, lower
related-absent FPR without TPR or yes-rate collapse, improve POPE-popular by at
least ten net corrections, exhibit near-zero common-mode update, and lose at
least half its gain when images are removed or permuted. POPE-random remains
untouched confirmation data and POPE-adversarial remains exhausted. No new
experimental result was produced in this design iteration.

## 2026-07-20 — PCPA cancelled; position-free detector audit pre-registered

**Direction correction:** Cancelled PCPA before data-manifest construction,
reference-margin caching, or LoRA training. It modifies the LVLM and therefore
does not satisfy the required training-free deployment setting. The project
returns to detection before attempting another mitigation method. The LVLM and
its generated outputs are frozen; an optional supervised external detector is
allowed but must be reported separately from a label-free score.

**Existing evidence:** On the same 16,426 LLaVA/COCO object mentions, the best
label-free controlled score is IC at 0.686 within-bin AUROC. Mean-over-head
attention shapes collapse near chance, but a raw post-RoPE layer-31 per-head
logistic probe reaches 0.726 under image-grouped OOF. This establishes a narrow
feasibility gap: useful late-head information exists, but no label-free readout
or source-level position removal has recovered it.

**New hypothesis:** Test clean per-head grounding gain. Recompute visual
attention from pre-RoPE Q/K, remove low-key-norm sink patches, preserve heads,
and compare each object-generating row with nearby generated non-object rows.
The fixed unsupervised score aggregates the strongest quartile of per-head JSD
and concentration gains. The supervised ceiling fits a no-intercept linear
probe only on hallucinated-minus-grounded differences matched within generation-
position bins, with image-grouped OOF and no position feature.

**Gate:** `docs/iteration_protocols/position_free_detector_iteration_1.md`
requires the label-free score to reach 0.70 within-bin and beat IC, or the clean
conditional probe to reach 0.75 and beat the raw layer-31 probe by 0.02. Raw,
no-RoPE, sink-clean, local-difference, all-head, and layer controls are frozen.
Matched-pair/residual AUROC and score-position correlations are mandatory. If
both routes fail, constructive attention detection closes; no head/layer/window
tuning is allowed on the full result. No new experimental result was produced
in this design iteration.

## 2026-07-20 — Position-free detector implementation and Stage-A smoke

**Implementation:** Added a frozen-LVLM extractor that registers read-only
pre-hooks on LLaVA layers 22 and 31, captures pre-RoPE per-head Q/K, detects
low-key-norm visual sinks, and stores compact raw/no-RoPE/clean object-versus-
local features. Added the locked offline evaluator for train-fold empirical-CDF
label-free scoring and image-grouped, position-matched pairwise logistic probes.
No model attention module is replaced and no detector feature contains position.

**Stage-A evidence:** On 16 images / 43 mentions, all three `(43, 2, 32, 8)`
feature tensors are finite. The hooks change neither logits nor raw attention
(both max errors 0), and no-RoPE and sink deletion are active rather than no-op
branches. The evaluator's fold, matching, residualization, and metric paths also
pass a 500-row synthetic runtime check.

**Protocol amendment:** The unversioned May 22 raw per-head cache cannot be an
exact numerical oracle: token alignment is correct, but 86 raw layer-row
comparisons show material cross-runtime drift. The discrepancy is retained as a
diagnostic; same-runtime no-hook versus hook identity is the hard faithfulness
gate, and every Stage-B branch is extracted together. This amendment precedes
all Stage-B metrics and changes no method hyperparameter or success threshold.

**Decision / next:** Stage A passes. Commit the implementation before launching
the one-shot full four-shard extraction and locked evaluation.

## 2026-07-20 — Position-free detector iteration 1 failed both full-data gates

**Execution:** Completed the committed one-shot four-shard extraction on all
16,426 LLaVA/COCO object mentions and the locked five-fold image-grouped OOF
evaluation. Object IDs are complete and unique, labels/positions align with the
baseline cache, and every branch is finite.

**Label-free result:** The pre-registered clean layer-31 top-eight local score is
below chance: 0.3881 overall, 0.4553 within-bin, 0.4633 matched, and 0.4766 OOF
residual. It loses decisively to IC at 0.6856 within-bin and also loses to its
clean object-only ablation at 0.5236. Neither no-RoPE nor sink deletion gives a
positive within-bin change. The unsupervised gate fails every substantive item.

**Supervised result:** The same-run raw conditional probe reaches 0.7405 within-
bin. Removing RoPE reaches 0.7431, only +0.0026 with paired 95% CI
`[-0.0102, 0.0174]`, while residual AUROC drops 0.6762 -> 0.6633. Sink deletion
then lowers within-bin to 0.6998; clean minus no-RoPE CI is
`[-0.0574, -0.0302]`. The clean probe misses the 0.75/0.72/0.68 thresholds and
does not beat raw. The supervised gate fails.

**Mechanistic decision:** RoPE removal does reduce position correlation, but it
does not expose a stronger position-independent verification signal. Deleting
low-key-norm sink patches removes useful evidence. Most importantly, nearby
non-object-row distribution change is not a grounding certificate; hallucinated
mentions do not show the pre-registered lack-of-local-gain ordering.

**Direction decision:** Close constructive attention-based detection exactly as
pre-registered. Do not rescue it by flipping the unsupervised sign or sweeping
heads, layers, windows, sink thresholds, or regularization on the full result.
Keep the 0.7431 no-RoPE probe only as a supervised external analysis ceiling,
not as an unsupervised or novel method. The durable two-pronged critique remains;
the next constructive design must be training-free and move outside attention
allocation.

## 2026-07-20 — Audit the 0.9 target before choosing the next detector

**Literature correction:** HaloProbe's reported 0.935 AUROC is its final
Bayesian posterior on the natural held-out distribution. Its own
position-balanced test AUROC is 0.876 without balanced training and 0.877 with
balanced training. Treat 0.9 within-position AUROC as a stronger stretch target,
not as a number already established by HaloProbe.

**Exploratory ceiling:** On the existing image-grouped folds, an untuned NumPy
pairwise ridge diagnostic reaches 0.7946 within-bin from raw L31 attention plus
IC/NLL/entropy and 0.8018 after adding L22. Adding object-word identity reaches
0.8264, but that gain is category prior rather than visual verification. Full
numbers and caveats are appended to `docs/experiment_results.md`.

**Decision:** Do not optimize toward 0.9 by adding position or object-frequency
shortcuts to a monolithic classifier. A HaloProbe-style full feature replica is
useful as a controlled baseline, but the constructive method must add evidence
that tests whether the target token is supported by visual content. Candidate
families to protocol next are target-logit-aligned visual value contributions,
layerwise multimodal evidence trajectories, and attention-preserving visual
value ablations. No proposal direction is selected in this audit.

## 2026-07-20 — Select Visual Value Evidence for the next detector iteration

**Selection:** The next active route is Visual Value Evidence (VVE). For every
object-generating row, layer, and head, decompose the visual-token part of the
attention output after value aggregation and the head-specific output projection,
then measure its alignment and normalized logit gain for the emitted object.
This asks whether visual content supports the target, rather than treating
attention allocation as support.

**Position contract:** The primary branch recomputes QK without layer-local
RoPE over the full causal prefix. The classifier receives no position,
repetition, response-length, or object-identity feature; training is balanced
inside position bins and evaluation remains image-grouped and within-bin. Raw
post-RoPE evidence is a matched control. The failed sink deletion is not reused.

**Novelty audit:** HaloProbe uses fine-grained attention plus confidence but no
value-to-target attribution. Owl/VTACR averages visual/text attention weights;
VISTA and the 2026 Overthinking detector trace layerwise logits without isolating
their visual source. Direct logit attribution remains acknowledged prior art;
the proposed claim is limited to source-decomposed, target-aligned visual
residual evidence for controlled object-hallucination detection.

**Protocol:** Locked formulas, Stage-A numerical gates, OOF controls, and
full-data thresholds in
`docs/iteration_protocols/visual_value_evidence_iteration_1.md` before any VVE
implementation or result inspection. A fused 0.85 within-bin AUROC is the
advance gate; 0.90 is explicitly a stretch target.

## 2026-07-20 — Implement VVE and pass the Stage-A numerical gate

**Implementation:** Added pure PyTorch source-decomposition primitives, a frozen
LLaVA all-layer extractor, an image-grouped conditionally balanced MLP evaluator,
and unit tests. The extractor caches compact per-head visual mass, value norm,
target dot/cosine, normalized target-logit gain, joint features, and decoder
confidence at fixed pre/post rows for raw and no-RoPE branches. Existing closed
attention code is unchanged.

**Stage-A result:** The 16-image run retained 43 mentions with finite
`(43,32,2,32,5)` per-head tensors. Forward logit error is 0, attention-output
reconstruction error is `3.8147e-06`, per-head source-sum error is
`2.8610e-06`, raw mass cache error is 0, and raw/no-RoPE maximum row difference
is 0.9432. Four mathematical unit tests and the evaluator's end-to-end smoke
paths pass.

**Decision / next:** Stage A passes without method changes or outcome-based
selection. Commit this implementation milestone, then run full sharded feature
extraction and the locked five-fold OOF evaluation. Do not interpret the
43-mention smoke metrics.

## 2026-07-20 — VVE supervised route passes; label-free route fails

**Execution:** Completed two all-layer extraction shards for all 16,426 cached
LLaVA/COCO object mentions and the registered five-fold image-grouped OOF
evaluation. Added the mandatory pre/post, same-position/same-category, and
leave-category-out audits without changing detector features or training
hyperparameters. A small-sample bootstrap edge case now returns an explicit
empty interval instead of failing.

**Primary result:** Evidence-only no-RoPE VVE reaches 0.8872 within-bin AUROC,
versus 0.7988 for the identical-MLP attention-plus-confidence baseline and
0.6724 for confidence. Its +0.0884 advantage over attention plus confidence has
image-bootstrap 95% CI `[0.0763, 0.1026]`. Confidence fusion reaches 0.8873 and
therefore contributes essentially no additional signal. The supervised route
passes the registered 0.85 / +0.03 / evidence-only / matched-residual gates but
narrowly misses the 0.90 stretch target.

**Leakage and transfer audits:** Pre-only VVE remains high at 0.8747 within-bin,
showing that post-target self-conditioning is not required. Inside the same
position and object category, VVE reaches 0.8581 versus 0.7648 for attention
plus confidence. With complete object categories held out from training, VVE
reaches 0.8040 versus 0.7539; the +0.0501 interval is `[0.0350, 0.0665]`.
However, category-held-out residual AUROC is lower than attention (0.5626 vs.
0.5994), and an intermediate MLP seed offset produced 0.8223, exposing material
seed sensitivity that must be averaged rather than selected.

**Negative boundaries:** The fixed label-free score reaches only 0.5518 within-
bin and is rejected. No-RoPE VVE improves over raw by only 0.0052 with interval
`[-0.0009, 0.0124]`; de-rotation is retained as position-reducing hygiene, not
as the source of the detector gain. The successful method is a supervised
external probe over a frozen LVLM and must be named accordingly.

**Decision / next:** Advance VVE to a separately pre-registered causal audit,
not another full-data feature sweep. Iteration 2 must (1) repeat key OOF results
over fixed classifier seeds, (2) permute target directions to measure lexical
shortcut dependence, and (3) intervene on visual values while preserving
attention weights to test whether high VVE predicts a target-logit or behavior
change. Only a stable causal result should be ported to Qwen3-VL and an
independent hallucination dataset.
