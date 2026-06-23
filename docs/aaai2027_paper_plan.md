# AAAI 2027 Paper Plan

Target venue: AAAI 2027 Main Technical Track.

Working title:

> Looking Is Not Verifying: Diagnosing Attention-Based Detection and Mitigation of Object Hallucinations in LVLMs

## 1. Central Claim

Attention-based hallucination methods rely on a brittle proxy:

    visual attention allocation ≈ object-level evidence verification

This paper shows that the proxy fails in both directions:

- **Detection:** attention scores can rank hallucinations well globally because they track nuisance factors such as generation position, not because they verify visual evidence.
- **Mitigation:** attention interventions can change routing and output style, but the change is not selective for true object presence.

The paper should not claim that attention is useless. The sharper, safer, and more interesting claim is:

> Attention can contain grounding information, but raw attention scores and attention amplification are not grounding guarantees. Hallucination methods must test whether attended evidence is target-discriminative.

## 2. Reviewer-Facing Story

### Opening Hook

Object hallucination is a grounding failure, so many methods try to inspect or manipulate visual attention. This is intuitive: if the model looks at the image, it should hallucinate less.

The hidden problem is that "looking" is not the same as "checking". A model can attend to a real region that is semantically related to the queried object, while the target object itself is absent.

Example:

    Query: "Is there a sink?"
    Image: toilet / bathroom context, no sink
    Attention: toilet or bathroom region
    Answer: yes

This attention is not random. It is visually meaningful but non-discriminative.

### Core Difficulty

The difficulty is not merely that attention is noisy. The hard case is that attention can be *plausible but wrong*. A heatmap may look grounded because it points to a real visual region, yet that region only supports an associated prior rather than the target object claim.

This makes both evaluation and intervention misleading:

- AUROC can reward positional shortcuts.
- POPE F1 can reward yes-prior movement.
- CHAIR can hide caption-style changes.
- Attention audits can show real routing changes that still fail to separate TP from FP.

### Paper Contribution Type

This should be framed as an analysis + diagnostic framework paper, not as a new detector paper.

The "method" is the diagnostic protocol:

1. Position-controlled detection evaluation.
2. Answer-prior-controlled mitigation evaluation.
3. Attention-shift selectivity audit.
4. Semantic-neighbor grounding audit.

This is enough for AAAI if the experiments are broad and visually clear, because the contribution is new knowledge plus a reusable evaluation protocol.

## 3. Claims and Evidence Map

| Claim | Evidence Needed | Current Status |
| --- | --- | --- |
| Global attention-detector AUROC is position-confounded. | Position-only AUROC, within-bin AUROC, matched-pair AUROC, residual AUROC. | Mostly done. |
| Attention mass and attention shape fail after position control. | PAS/SVAR/Beyond/SinkDetect comparison table. | Mostly done. |
| The failure is not "attention has no information". | Per-head diagnostic probe, IC/entropy/NLL contrast. | Partially done; should be framed as diagnostic, not final method. |
| Attention interventions shift behavior but not verification. | POPE yes-rate/TPR/FPR/MCC, CHAIR length/object count/hallucinated count. | Done for current methods. |
| Routing changes are real but not selective. | Attention-shift audit TP vs FP, pre/post visual/prefix/sink mass. | Partially done; make table stronger. |
| Associated evidence explains why plausible attention can still hallucinate. | Semantic-neighbor audit with absent target + related object present. | Must add. This is the key missing experiment. |

## 4. Required New Experiment: Semantic-Neighbor Grounding Audit

This should become the paper's most distinctive experiment.

### Goal

Test whether detection and mitigation methods fail hardest when the target object is absent but semantically or contextually related visual evidence is present.

### Dataset Construction

Start from POPE/COCO object labels and construct three negative query subsets:

1. **Random negatives:** target absent, no strong related object requirement.
2. **Co-occurrence negatives:** target absent, but a frequently co-occurring object is present.
3. **Semantic-neighbor negatives:** target absent, but a visually or semantically related object is present.

Examples:

| Target absent | Related present | Rationale |
| --- | --- | --- |
| sink | toilet | bathroom co-occurrence |
| fork | knife | dining object relation |
| bus | car/truck | vehicle relation |
| bed | couch/chair | furniture/indoor relation |
| cup | bottle | container relation |

Use two relation sources:

- COCO co-occurrence matrix from ground-truth annotations.
- Text embedding similarity among object names, with manual filtering for obvious bad pairs.

### Metrics

For detection:

- AUROC / AP on each negative subset.
- False-grounded rate at fixed overall FPR.
- Score gap: present target vs semantic-neighbor negative.

For mitigation:

- Yes rate on each negative subset.
- FPR increase over vanilla.
- Delta TPR - Delta FPR.

For attention audit:

- visual attention mass change;
- related-object-region attention mass if boxes are available;
- TP vs semantic-neighbor-FP attention gap.

### Expected Interpretation

The strongest result would be:

> Attention methods fail disproportionately on semantic-neighbor negatives, where attention can point to meaningful but non-target evidence.

This turns the story from "attention is unreliable" into "attention is insufficient because it lacks target-discriminative verification."

## 5. Experiment Matrix for AAAI

### E1. Detection Main Result

Purpose: show global AUROC is misleading.

Table:

| Method family | Method | Overall AUROC ↑ | Position-only floor | Within-bin AUROC ↑ | Same-object AUROC ↑ | Residual AUROC ↑ | Retained signal ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

Rows:

- Position only.
- NLL.
- Entropy.
- IC.
- GLSim.
- SVAR.
- PAS.
- Beyond ADS.
- Beyond CGC.
- Beyond ADS+CGC.
- SinkDetect best shape.
- Per-head probe diagnostic, clearly marked supervised.

Message:

> The apparent strongest detectors are often the most position-confounded.

### E2. Position Confound Figure

Purpose: make the confound visually undeniable.

Figure:

- x-axis: generated object position bin.
- left y-axis: hallucination rate.
- right y-axis: mean PAS/SVAR/SinkDetect score.
- optional: overlay position-only score.

Message:

> Hallucination labels and attention scores drift together with decoding position.

### E3. Detection Stress Test: SinkDetect

Purpose: show that the failure is not just crude attention mass.

Table or compact figure:

| Variant | no-RoPE | sink removal | top-mass | Best overall | Best within-bin | Best residual |
| --- | --- | --- | --- | ---: | ---: | ---: |

Message:

> Removing obvious attention artifacts does not recover robust target evidence from mean-over-head attention shape.

### E4. Per-Head Diagnostic

Purpose: avoid overclaiming that attention contains no information.

Figure:

- bar plot by layer group: early, middle, late;
- metrics: within-bin AUROC or matched AUROC of supervised probe;
- optionally compare mean-over-head handcrafted score vs per-head probe.

Message:

> Attention-derived features can contain signal, but it is head-specific and not captured by global handcrafted summaries.

### E5. Mitigation Main Result

Purpose: show attention interventions do not reliably improve verification.

Table:

| Method | POPE Acc ↑ | F1 ↑ | Yes rate | TPR ↑ | FPR ↓ | ΔTPR-ΔFPR ↑ | MCC ↑ | CHAIR_i ↓ | Caption len | Obj mentions | Hallu obj mentions ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Rows:

- Vanilla.
- PAI attention-only.
- ClearSight.
- Visual Attention Sink.
- Head-selection methods, once reproduced.

Message:

> Some methods increase recall or object richness, but not selective verification.

### E6. Attention-Shift Selectivity Audit

Purpose: show interventions do what they intend mechanically, but not in the right samples.

Table:

| Method | Δ visual mass | Δ prefix mass | Δ sink mass | Decision changed? | TP visual mass | FP visual mass | TP-FP gap ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Message:

> Routing changes are real; selectivity is missing.

### E7. Semantic-Neighbor Grounding Audit

Purpose: make the paper excellent rather than merely negative.

Table:

| Setting | Vanilla yes rate | Method yes rate | FPR increase | Attention target/related gap | Detection false-grounded rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random negative | | | | | |
| Co-occurrence negative | | | | | |
| Semantic-neighbor negative | | | | | |

Case-study panel:

Panel A: query target present, attention on target, correct yes.
Panel B: target absent, related object present, attention on related object, false yes.
Panel C: intervention amplifies visual attention, answer remains or becomes false yes.

Message:

> The failure mode is not absent visual attention; it is associated evidence being mistaken for target evidence.

## 6. Figure Plan

### Figure 1: Teaser / Mechanism

One-page message:

    Looking -> associated evidence -> false claim
    Verifying -> target-discriminative evidence -> grounded claim

Layout:

- Left: target present, attention on target, "verified".
- Middle: target absent but related object present, attention on related object, "plausible but false".
- Right: intervention amplifies visual routing, but FP remains.

Status: TODO. Needs real case examples from POPE/COCO if possible.

### Figure 2: Diagnostic Framework

Show the unified framework:

    attention proxy
      -> detection score
      -> position control
      -> mitigation intervention
      -> answer-prior control
      -> semantic-neighbor audit

Status: can be drawn as a clean vector diagram.

### Figure 3: Position Confound

Use existing detection results.

Status: likely already available as `fig_per_bin.png` / controlled analysis; should regenerate in AAAI style.

### Figure 4: Per-Head Signal

Use per-head diagnostic outputs.

Status: partially available; needs final plotting script.

### Figure 5: Semantic-Neighbor Case Study

Use real images, questions, answers, and attention maps.

Status: requires new semantic-neighbor audit.

## 7. Table Plan

Main paper should contain 4 tables max:

1. **Main detection table:** exact AUROC and controlled metrics.
2. **Strong controls table:** same-object / nonlinear residual / retained signal.
3. **Mitigation behavior table:** POPE + CHAIR behavior.
4. **Semantic-neighbor audit table:** the new key result.

Appendix tables:

- all baseline variants;
- all SinkDetect ablations;
- per-split POPE results;
- all attention-shift layers;
- extra semantic-neighbor object pairs.

## 8. Seven-Page AAAI Main Text Allocation

Assume 7 pages of technical content. This must be verified against the final AAAI-27 author kit.

Recommended allocation:

1. Introduction: 1.0 page.
2. Attention-as-Verification Proxy + Diagnostic Protocol: 1.1 pages.
3. Experimental Setup: 0.6 page.
4. Detection Findings: 1.5 pages.
5. Mitigation Findings: 1.2 pages.
6. Semantic-Neighbor Audit: 0.9 page.
7. Discussion / Recommendations / Conclusion: 0.7 page.

This means the current paper should merge "Attention as a Proxy" and "Diagnostic Protocol" into one compact section, and avoid a long standalone discussion. The discussion should be integrated into result paragraphs with bold takeaway sentences.

## 9. Section Structure

Recommended visible structure:

1. Introduction
2. Attention Allocation vs. Evidence Verification
   - shared proxy
   - diagnostic protocol
   - setup
3. Detection: Global Scores Do Not Imply Grounding
   - position confound
   - controlled baseline comparison
   - SinkDetect stress test
   - per-head diagnostic
4. Mitigation: More Visual Routing Does Not Imply Fewer Hallucinations
   - behavior metrics
   - attention-shift selectivity audit
5. Associated Evidence Audit
   - construction
   - detection failures
   - mitigation failures
   - case study
6. Conclusion

Appendix:

- full implementation details;
- full baseline definitions;
- additional plots/tables;
- reproducibility checklist;
- ethical / limitation discussion.

## 10. Reviewer Attack Points and Fixes

### Attack 1: "This is only negative; where is the method?"

Fix:

Frame the contribution as a diagnostic framework with a new semantic-neighbor audit. Use "method" language for the protocol:

> We propose a verification-centered diagnostic protocol that separates routing changes from target-object verification.

### Attack 2: "Attention methods may still work with better head selection."

Fix:

Do not claim all attention methods fail. Include per-head diagnostic and head-selection baselines once reproduced. State:

> Our results rule out raw mass, mean-over-head shape, and unselective amplification as grounding guarantees; they motivate head-specific verification rather than invalidate attention as a feature.

### Attack 3: "Your mitigation reproductions are incomplete."

Fix:

Be explicit:

- PAI is attention-only.
- ClearSight and Visual Attention Sink are attention-intervention ports.
- Full pipelines with decoding branches are separate methods.

Add at least one official-code parity check for PAI/ClearSight/VisAttnSink on a small subset.

### Attack 4: "One model/dataset is too narrow."

Fix:

Minimum acceptable expansion:

- Detection: add one more LVLM, preferably Qwen2.5-VL or InstructBLIP.
- Mitigation: add one more model only for POPE if compute allows.
- Semantic-neighbor audit: at least run on LLaVA and one second model for the headline result.

### Attack 5: "Associated evidence is hand-wavy."

Fix:

Make it empirical:

- define semantic-neighbor pairs systematically;
- report subset metrics;
- show real case panels;
- compute attention target-vs-related gap when boxes are available.

## 11. Concrete Next Experiments

Priority order:

1. Implement semantic-neighbor audit construction from COCO labels.
2. Run detection scores on semantic-neighbor negative subsets using existing baseline_scores.csv.
3. Run mitigation POPE-style queries on semantic-neighbor negatives.
4. Extract attention maps for 50-100 representative semantic-neighbor examples.
5. Reproduce at least one head-selection mitigation baseline or explicitly mark it as out-of-scope with a small literature comparison.
6. Add a second LVLM for the core detection table or semantic-neighbor audit.
7. Regenerate all figures in AAAI style.

## 12. Paper Positioning Sentence

Use this as the paper's spine:

> We show that the dominant attention-as-grounding proxy fails because routing visual computation is weaker than verifying target-object evidence; this failure appears in both hallucination detection and mitigation, and becomes clearest under position, answer-prior, and semantic-neighbor controls.

