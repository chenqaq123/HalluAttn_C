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
