# Semantic-Neighbor Negative Audit

This audit builds harder POPE negative subsets from COCO annotations. The goal
is to separate plain absent targets from absent targets whose image contains
objects that are semantically or empirically related to the queried object.

## Tool

Run:

```bash
python mitigation/scripts/build_semantic_neighbor_audit.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --output_dir mitigation/results/semantic_neighbor_audit
```

The result directory is ignored by git because it contains regenerated CSV/JSON
outputs.

The script:

1. loads COCO `instances_val2014.json`;
2. maps each image to its COCO object categories;
3. builds top-k object neighbors by image-level Jaccard co-occurrence;
4. parses each POPE object question, including the `imange` typo present in
   some adversarial questions;
5. labels negative rows as:
   - `negative_related_present` when a top co-occurrence neighbor or same
     supercategory sibling is present;
   - `negative_absent_plain` otherwise;
   - `negative_target_present_coco_label` if COCO says the target is actually
     present despite a negative POPE label.

Positive rows are labeled as `positive_present` or
`positive_missing_coco_label`.

## Current Full-Data Check

Using COCO val2014 and the three POPE splits under
`/home/chenguanxu/common_dataset/pope`, the script covers all 9,000 POPE rows:

| Split | Rows | Related-present negatives | Plain absent negatives | Related rate among negatives |
|---|---:|---:|---:|---:|
| random | 3,000 | 840 | 660 | 56.0% |
| popular | 3,000 | 980 | 520 | 65.3% |
| adversarial | 3,000 | 1,272 | 228 | 84.8% |

Parser diagnostics:

- unparsed questions: 0;
- missing image ids: 0;
- unknown target objects: 0.

This confirms that POPE adversarial negatives are much more concentrated in
related-object contexts than random negatives. The next evaluation step is to
join these labels with each method's POPE predictions and report FPR/MCC by
negative subset.

## Attention-Only Mitigation Check

Run:

```bash
python mitigation/scripts/evaluate_semantic_neighbor_subsets.py \
  --result_root mitigation/results/coco_llava_7b_attention_only \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --output_dir mitigation/results/semantic_neighbor_audit/attention_only_subset_eval
```

Current FPR by negative subset:

| Split | Method | All negatives | Related-present | Plain absent | Gap |
|---|---|---:|---:|---:|---:|
| random | vanilla | 3.7% | 5.5% | 1.4% | +4.1% |
| random | PAI | 3.5% | 5.1% | 1.4% | +3.8% |
| random | ClearSight | 6.1% | 8.3% | 3.3% | +5.0% |
| random | VisAttnSink | 4.4% | 6.2% | 2.1% | +4.1% |
| popular | vanilla | 7.7% | 10.0% | 3.5% | +6.5% |
| popular | PAI | 7.5% | 9.7% | 3.5% | +6.2% |
| popular | ClearSight | 11.1% | 14.4% | 4.8% | +9.6% |
| popular | VisAttnSink | 8.6% | 10.9% | 4.2% | +6.7% |
| adversarial | vanilla | 14.7% | 16.4% | 5.3% | +11.1% |
| adversarial | PAI | 14.3% | 15.9% | 5.3% | +10.6% |
| adversarial | ClearSight | 20.2% | 22.3% | 8.3% | +14.0% |
| adversarial | VisAttnSink | 15.8% | 17.3% | 7.5% | +9.8% |

The gap is consistent across all methods and grows on harder POPE splits. This
supports the current paper claim that attention-only interventions do not solve
target-discriminative verification: related visible objects remain the main
source of false-positive yes answers.

## VCD-Greedy Check

Run:

```bash
python mitigation/scripts/evaluate_semantic_neighbor_subsets.py \
  --result_root mitigation/results/pope_full_vcd_greedy_audit \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --output_dir mitigation/results/semantic_neighbor_audit/vcd_greedy_subset_eval \
  --methods vanilla,vcd
```

Current FPR by negative subset:

| Split | Method | All negatives | Related-present | Plain absent | Gap |
|---|---|---:|---:|---:|---:|
| random | vanilla | 3.7% | 5.5% | 1.4% | +4.1% |
| random | VCD-greedy | 4.5% | 6.1% | 2.4% | +3.6% |
| popular | vanilla | 7.7% | 9.9% | 3.5% | +6.4% |
| popular | VCD-greedy | 9.2% | 11.8% | 4.2% | +7.6% |
| adversarial | vanilla | 14.5% | 16.2% | 5.3% | +10.9% |
| adversarial | VCD-greedy | 16.2% | 17.8% | 7.5% | +10.3% |

VCD-greedy therefore does not resolve the related-object false-positive mode.
It increases related-present FPR on every split, while the related-minus-plain
gap remains large. This supports the paper direction: contrastive decoding can
reduce some language-prior reliance, but it is not target-discriminative visual
verification.
