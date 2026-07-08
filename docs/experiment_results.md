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
| SVAR (layer-0 mass) | 0.834 | 0.575 | 0.576 | 0.557 | same position issue; layer-0 mass, not the paper's middle-layer SVAR (relabel) |
| IC | 0.776 | **0.686** | **0.703** | 0.633 | strongest position-controlled baseline |
| GLSim-local | 0.772 | 0.564 | 0.593 | 0.601 | some residual signal |
| Entropy | 0.721 | 0.637 | 0.655 | 0.634 | reliable uncertainty baseline |
| NLL | 0.711 | 0.636 | 0.652 | 0.630 | ~Entropy |
| Beyond ADS | 0.503 | 0.511 | 0.515 | 0.501 | near random |
| Beyond CGC | 0.692 | 0.521 | 0.520 | 0.518 | weak after control |
| Beyond ADS+CGC | 0.693 | 0.523 | 0.524 | 0.521 | CGC carries the combined score |

Takeaway: PAS/SVAR's high overall AUROC is position-confounded (collapse to
~random within-bin). IC is the strongest position-controlled baseline. This is
the Prong-1 reproduction: attention-mass detectors do not survive length control.

> **Reproduction fidelity (verified 2026-07-08, see §E):** PAS is a line-for-line
> faithful port of `lanl/pas` — its collapse is genuine. IC (final-layer only, a
> conservative lower bound), GLSim, NLL, Entropy are faithful. **SVAR here is
> layer-0 visual-attention mass, not the paper's middle-layer (5–18) SVAR** —
> relabel or fix. Beyond-ADS/CGC are simplified unsupervised single-layer proxies
> (`paper_reimplementation`); ADS≈0.503 reflects the proxy, not the method.

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
