| Run | Images | Tokens | Mode | Changed | Removed | Introduced hallucinated | Unsupported open-vocab | Note |
|---|---:|---:|---|---:|---|---|---|---|
| prefilter_single_token_only | 5 | - | hard | 1 | bird | person | - | removes bird on image 22596 but routes to unsupported person |
| prefilter_all_first | 5 | - | hard | 2 | bird | person | above, accompanied, brown, brown wooden, wooden | same person route; extra open-vocab flags on image 6213 are mostly non-object words |
| closed_loop_top30_22596 | 1 | - | hard | 1 | bird | person | - | top30 absent-object list still misses person |
| closed_loop_all_22596 | 1 | - | hard | 1 | bird | - | - | person is denied, but caption ends with incomplete "two ch" |
| prefilter_iter2_single_token_only | 5 | - | hard | 1 | bird | - | - | adds person after first audit; no new COCO claim, but caption still ends with incomplete "two ch" |
| prefilter_iter2_soft_p4 | 5 | - | soft p=4.0 | 1 | bird | - | - | same as hard iter2 on 5 images: no new COCO claim, but still incomplete "two ch" |
| prefilter_iter2_soft_p1_22596 | 1 | - | soft p=1.0 | 0 | - | - | - | too weak on stress image; leaves original bird claim unchanged |
| prefilter_iter2_t96 | 5 | 96 | hard | 5 | bird, bottle, car | elephant, person, zebra | bottes | longer budget produces complete substitute/escape claims: bottes, elephant/zebra, chickens, person |

## Sentence Repair

| Setting | Repaired | Mean Removed Words | Gated CHAIRi | Repaired CHAIRi | Gated Hall. | Repaired Hall. |
|---|---:|---:|---:|---:|---:|---:|
| iter2 t64 | 4/5 | 5.00 | 0.1111 | 0.0588 | 2 | 1 |
| iter2 t96 | 4/5 | 8.20 | 0.2500 | 0.1786 | 8 | 5 |

Scope: offline sentence-boundary truncation of incomplete trailing fragments; no new text is generated.
