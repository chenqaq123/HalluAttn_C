# Object Co-occurrence / Associated-Evidence Experiment Specification

## 1. Purpose

This document specifies the remaining experiments for the paper subsection on
the **object co-occurrence prior**. The experiments support the paper's central
claim:

> Visual attention allocation is not target-object evidence verification.

The relevant failure mode is **associated-evidence grounding**: the queried
target is absent, but a related object is visible. An attention intervention may
increase visual influence or route attention toward that real object and still
produce an incorrect “yes,” because the visible evidence is associated with the
target rather than discriminative for it.

This is a diagnostic experiment, not a new benchmark and not an attempt to
re-establish the already-known fact that object co-occurrence contributes to
hallucination.

## 2. Scope and hypotheses

### H1 — Existing co-occurrence prior (textual argument only)

Do **not** add a new experiment whose main conclusion is simply that related
objects cause more hallucinations. POPE already uses object co-occurrence when
constructing difficult/adversarial negatives, and prior work already motivates
this relationship.

The paper should establish H1 through citations and the POPE construction, then
ask the new question: **do attention interventions amplify this existing prior?**

### H2 — Intervention-induced yes bias on related negatives (main quantitative experiment)

For target-absent POPE questions, compare each intervention with vanilla on the
same questions. Separate negatives into:

- `related_present`: the target is absent and at least one predefined related
  object is annotated as present;
- `plain_absent`: the target is absent and no predefined related object is
  annotated as present.

The primary claim is not the raw difference between these two subsets. Vanilla
already has a co-occurrence gap. The primary estimand is whether an intervention
increases the yes ratio more on `related_present` than on `plain_absent`.

For method \(m\) and subset \(S\), define

\[
\Delta_{\mathrm{yes}}^m(S)
=
\operatorname{YesRate}_m(S)
-
\operatorname{YesRate}_{\mathrm{vanilla}}(S).
\]

On negative questions, yes rate equals false-positive rate. Define the
co-occurrence-specific amplification:

\[
\operatorname{Amplification}_m
=
\Delta_{\mathrm{yes}}^m(\mathrm{related})
-
\Delta_{\mathrm{yes}}^m(\mathrm{plain}).
\]

Interpretation:

- positive: additional intervention harm is concentrated on related negatives;
- approximately zero: a global yes/FPR shift rather than co-occurrence-specific
  amplification;
- negative: no evidence that the intervention amplifies the co-occurrence prior;
- near-zero deltas on both subsets: the intervention is behaviorally inert.

### H3 — Two qualitative associated-evidence cases

Select two reproducible cases in which:

1. the POPE label is negative and the queried target is absent;
2. a strongly related object is visibly present and has a COCO annotation;
3. vanilla answers `no` and an attention intervention answers `yes` whenever
   suitable cases exist;
4. the intervention-induced attention increase is spatially concentrated on the
   present related object.

These examples are **mechanism illustrations**, not aggregate or causal proof.
The paper must use language such as “illustrates” or “is consistent with,” not
“proves” or “establishes in general.”

## 3. Existing inputs to reuse

The implementation should reuse saved predictions and existing audit metadata
where possible. Do not rerun full POPE generation merely to recompute H2.

Primary inputs:

- `mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv`
- `mitigation/results/semantic_neighbor_audit/{random,popular,adversarial}_semantic_neighbor_rows.csv`
- `mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json`
- `mitigation/results/coco_llava_7b_attention_only/pope/<split>/<method>/predictions.jsonl`
- `mitigation/results/coco_llava_7b_attention_only/pope/<split>/comparison.json`
- `mitigation/results/coco_llava_7b_attention_audit/pope/adversarial/<method>/attention_audit.jsonl`
- `mitigation/results/pope_mechanism_alignment_full/`

At minimum, evaluate:

- `vanilla`
- `pai`
- `clearsight`
- `visattnsink`

If aligned full predictions are already available, also report decoding controls
such as VCD and NoLan-compatible, but keep the attention-intervention family
visually distinct from non-attention controls.

Before analysis, verify that joins by split, question id, image id, target, and
label are one-to-one and that all compared methods cover exactly the same rows.
Abort on duplicates, missing predictions, or label disagreements.

## 4. H2 quantitative analysis

### 4.1 Required row-level table

Create one row per `(split, question_id, method)` with at least:

- `split`
- `question_id`
- `image_id`
- `target`
- `label`
- `subset` (`related_present` or `plain_absent`)
- `vanilla_answer`
- `method`
- `method_answer`
- `vanilla_yes`
- `method_yes`
- `answer_delta = method_yes - vanilla_yes`
- `transition` (`no_to_no`, `no_to_yes`, `yes_to_no`, `yes_to_yes`)
- `present_objects`
- `related_present_objects`

Only negative-label rows enter the primary H2 analysis.

### 4.2 Required metrics

For every method and each POPE split, compute separately on `related_present`
and `plain_absent`:

- number of questions;
- vanilla yes ratio/FPR;
- method yes ratio/FPR;
- `delta_yes_ratio` relative to vanilla;
- `no_to_yes_rate`;
- `yes_to_no_rate`;
- net paired change: `no_to_yes_rate - yes_to_no_rate`.

Then compute:

- `cooccurrence_amplification = delta_related - delta_plain`;
- `delta_tpr` on positive questions;
- `related_selectivity = delta_tpr - delta_related`.

Also provide macro averages across random, popular, and adversarial splits. Macro
means must average the three split-level rates rather than pooling all rows and
calling the result macro.

### 4.3 Statistical uncertainty

Use a paired bootstrap because vanilla and intervention predictions correspond
to the same questions.

- bootstrap unit: `image_id`;
- repetitions: at least 10,000;
- seed: fixed and recorded, default 0;
- interval: percentile 95% confidence interval.

Bootstrap and report confidence intervals for:

- `delta_yes_ratio` on each subset;
- `cooccurrence_amplification`;
- `related_selectivity`.

If several questions share an image, sample images and retain all their
questions together.

### 4.4 Primary output table

Produce a paper-ready CSV and LaTeX table with this logical schema:

| Method | Delta TPR | Delta FPR plain | Delta FPR related | Amplification | Related selectivity |
|---|---:|---:|---:|---:|---:|

All values should be reported in percentage points with 95% confidence
intervals available in the CSV/JSON. The main table may show macro values;
split-wise results belong in the appendix.

### 4.5 Primary figure

Create a grouped bar plot centered at zero. For each method, show:

- `Delta Yes Ratio: Plain-Absent`;
- `Delta Yes Ratio: Related-Present`.

Requirements:

- vanilla is the zero reference and need not be drawn as a method bar;
- include paired-bootstrap 95% CI error bars;
- annotate `related - plain` above each method group;
- use the same method colors as other paper mitigation figures where possible;
- write exact plotted values to a sidecar CSV/JSON;
- export both PDF and PNG.

The figure should make the following distinctions visible rather than forcing a
single conclusion:

- co-occurrence-specific amplification;
- global yes bias;
- inert intervention;
- suppression of false positives.

### 4.6 Required interpretation guardrail

Do not infer co-occurrence-specific amplification merely because a method has a
higher raw FPR on related than plain negatives. That gap already exists in
vanilla. A method-specific claim requires a positive paired difference-in-
differences estimate.

Current audited numbers suggest that ClearSight may show this pattern, while PAI
is close to inert and VisAttnSink may be closer to a global yes shift. The new
analysis must report what the paired estimates show and must not force all
attention methods into the same failure mode.

## 5. H3 qualitative attention cases

### 5.1 Case selection protocol

Build the candidate pool using rules fixed before inspecting attention maps:

1. negative POPE label;
2. `related_present` subset;
3. target absent according to the audit/COCO annotations;
4. at least one annotated related object with a valid bounding box;
5. prefer `vanilla=no` and `intervention=yes` transitions;
6. require the related-object box to be visible and non-trivial in area;
7. select two cases with different target–neighbor pairs and preferably
   different scene types.

Do not select cases by maximizing attention/bounding-box overlap. For a
reproducible ranking, use annotation-side quantities only, such as
co-occurrence rank/strength and visible bounding-box area. Save the complete
candidate list and the deterministic ranking rule.

ClearSight is the preferred intervention if H2 confirms positive
co-occurrence-specific amplification. If it does not, choose the method
supported by H2 or explicitly present the example only as a general
associated-evidence false positive.

### 5.2 Attention tensor and aggregation

Use the answer-decision query row corresponding to the first generated yes/no
answer token. Keep image, prompt, query position, decoding settings, and
layer/head aggregation identical between vanilla and the intervention.

Use a layer range defined by the intervention configuration, not a layer chosen
because it produces the cleanest heatmap. Aggregate heads with a fixed rule
(normally the mean) and record the exact layers and heads in the metadata.

Two complementary maps are required:

1. **Within-visual localization map**

   \[
   \widetilde A_j = A_j / \sum_{k\in\mathcal V} A_k.
   \]

   This shows where visual attention is located, but it removes total visual
   mass and therefore must not be used to quantify amplification.

2. **Matched intervention residual**

   \[
   \Delta A_j = A_j^{m} - A_j^{\mathrm{vanilla}}.
   \]

   Visualize the positive component `max(delta, 0)` and retain the signed values
   in the sidecar data. Because the same image, query, and token position are
   compared, call this a **position-matched residual** or **matched
   intervention-induced attention change**, not a generally
   “position-debiased attention map.”

Do not normalize the residual within visual tokens before measuring magnitude;
doing so can erase the total visual-mass increase caused by amplification.

### 5.3 Related-region measurement

Map the COCO bounding box for the related object onto the model's visual-patch
grid. For each case, report:

- absolute related-region attention mass under vanilla;
- absolute related-region attention mass under the intervention;
- signed change in related-region mass;
- fraction of positive residual mass inside the related region;
- region area as a fraction of image/visual patches.

If multiple related-object boxes overlap, use their union and record the
individual categories and boxes.

### 5.4 Qualitative figure layout

Create a two-row figure, one row per case, with columns:

1. original image with target/related-object annotation;
2. vanilla within-visual attention localization;
3. intervention within-visual attention localization;
4. positive matched residual `intervention - vanilla`;
5. compact text panel with query, answers, related object, and region-mass
   values.

Overlay the related-object bounding box consistently in all spatial panels.
Use one shared color scale for vanilla/intervention localization within a case
and a separately labeled scale for the residual.

The caption must state that the cases illustrate the mechanism and that the
aggregate behavioral result is provided by H2.

## 6. Deliverables

Place new experiment artifacts under a dedicated result directory, suggested:

```text
mitigation/results/object_cooccurrence_prior/
```

Required artifacts:

```text
object_cooccurrence_rows.csv
object_cooccurrence_metrics.csv
object_cooccurrence_metrics.json
object_cooccurrence_bootstrap.json
object_cooccurrence_main_table.tex
object_cooccurrence_yes_delta.pdf
object_cooccurrence_yes_delta.png
object_cooccurrence_yes_delta_data.csv
qualitative_candidates.csv
qualitative_cases.json
qualitative_cases.pdf
qualitative_cases.png
qualitative_case_1_attention.npz
qualitative_case_2_attention.npz
README.md
```

The result `README.md` must record:

- exact commands;
- git commit;
- model/checkpoint;
- data paths;
- method configurations;
- random seeds;
- layer/head aggregation;
- bootstrap settings;
- any missing rows or deviations from this specification.

Add or update a paper-number audit so every number in the new table and figure
can be checked against the CSV/JSON artifacts.

## 7. Acceptance criteria

The experiment is complete only if:

1. all H2 comparisons use identical, one-to-one aligned questions;
2. raw FPR, paired transitions, delta yes ratios, and difference-in-differences
   are all reported;
3. 95% paired-bootstrap confidence intervals are included;
4. macro and split-wise results are both available;
5. exact figure values are saved separately from the rendered figure;
6. qualitative cases follow a deterministic, attention-independent selection
   protocol;
7. each qualitative case records the exact attention extraction and aggregation
   settings;
8. the visualizations distinguish localization from absolute attention
   amplification;
9. conclusions distinguish co-occurrence-specific amplification, global yes
   bias, and inert behavior;
10. no new H1 experiment or broad claim is added unless the results reveal a
    genuine need for it.

## 8. Intended paper-level conclusion

The desired analysis should support only the strongest conclusion licensed by
the results:

- The co-occurrence prior is known from prior work and POPE construction.
- H2 tests whether each attention intervention amplifies that prior beyond its
  effect on plain negatives.
- H3 gives two concrete illustrations in which additional attention is placed
  on real but non-discriminative associated evidence.
- Together, a positive H2 result and consistent H3 examples support the claim
  that increasing visual attention can strengthen plausibility without
  verifying the queried target.

If H2 is not positive for a method, describe it as globally biased or inert as
appropriate; do not use the qualitative examples to override the aggregate
result.
