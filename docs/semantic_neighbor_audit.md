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
