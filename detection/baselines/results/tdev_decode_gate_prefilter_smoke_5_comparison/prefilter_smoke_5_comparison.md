| Run | Images | Mode | Changed | Removed | Introduced hallucinated | Unsupported open-vocab | Note |
|---|---:|---|---:|---|---|---|---|
| prefilter_single_token_only | 5 | hard | 1 | bird | person | - | removes bird on image 22596 but routes to unsupported person |
| prefilter_all_first | 5 | hard | 2 | bird | person | above, accompanied, brown, brown wooden, wooden | same person route; extra open-vocab flags on image 6213 are mostly non-object words |
| closed_loop_top30_22596 | 1 | hard | 1 | bird | person | - | top30 absent-object list still misses person |
| closed_loop_all_22596 | 1 | hard | 1 | bird | - | - | person is denied, but caption ends with incomplete "two ch" |
| prefilter_iter2_single_token_only | 5 | hard | 1 | bird | - | - | adds person after first audit; no new COCO claim, but caption still ends with incomplete "two ch" |
| prefilter_iter2_soft_p4 | 5 | soft p=4.0 | 1 | bird | - | - | same as hard iter2 on 5 images: no new COCO claim, but still incomplete "two ch" |
| prefilter_iter2_soft_p1_22596 | 1 | soft p=1.0 | 0 | - | - | - | too weak on stress image; leaves original bird claim unchanged |
