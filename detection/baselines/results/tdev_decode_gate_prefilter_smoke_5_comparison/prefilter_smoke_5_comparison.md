| Run | Images | Changed | Removed | Introduced hallucinated | Unsupported open-vocab | Note |
|---|---:|---:|---|---|---|---|
| prefilter_single_token_only | 5 | 1 | bird | person | - | removes bird on image 22596 but routes to unsupported person |
| prefilter_all_first | 5 | 2 | bird | person | above, accompanied, brown, brown wooden, wooden | same person route; extra open-vocab flags on image 6213 are mostly non-object words |
| closed_loop_top30_22596 | 1 | 1 | bird | person | - | top30 absent-object list still misses person |
| closed_loop_all_22596 | 1 | 1 | bird | - | - | person is denied, but caption ends with incomplete "two ch" |
