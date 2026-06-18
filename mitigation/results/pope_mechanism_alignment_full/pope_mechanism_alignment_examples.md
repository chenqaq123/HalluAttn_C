# POPE Mechanism Alignment Examples

These are related-present negative POPE rows where vanilla answers `yes`, TDEV answers `no`, and the strongest semantic-neighbor evidence exceeds target evidence. They illustrate associated-evidence grounding: the image contains plausible related objects, but not the queried target.

Vanilla related-present FPR in this table source: 0.114. TDEV corrects 0.398 of vanilla related-present false positives. Among vanilla related-present false positives, neighbor evidence exceeds target evidence in 0.960.

| split | question_id | target | related_present | best_neighbor | target_score | best_neighbor_score | tdev_margin | vanilla | pai | clearsight | visattnsink | tdev | lh_absent_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 340 | dog | bed|book | bed | 0.033 | 0.534 | -0.501 | yes | yes | yes | yes | no | 2.121 |
| random | 772 | dog | couch|person | couch | 0.074 | 0.532 | -0.458 | yes | yes | yes | yes | no | -2.354 |
| random | 1720 | tie | cell phone|chair|person | cell phone | 0.012 | 0.467 | -0.456 | yes | yes | yes | yes | no | 2.496 |
| random | 1772 | umbrella | bicycle|handbag|person | bicycle | 0.115 | 0.464 | -0.349 | yes | yes | yes | yes | no | -1.328 |
| random | 96 | tie | cake|dining table|knife|person | cake | 0.052 | 0.457 | -0.405 | yes | yes | yes | yes | no | 3.700 |
| random | 92 | teddy bear | cake | cake | 0.101 | 0.457 | -0.356 | yes | yes | yes | yes | no | -0.526 |
| popular | 2448 | cup | dining table|fork|knife|pizza|wine glass | pizza | 0.113 | 0.694 | -0.581 | yes | yes | yes | yes | no | -3.937 |
| popular | 2016 | chair | couch|person|tv | tv | 0.119 | 0.678 | -0.559 | yes | yes | yes | yes | no | -4.473 |
| popular | 2850 | cup | bottle|dining table|fork|knife|pizza | pizza | 0.065 | 0.568 | -0.503 | yes | yes | yes | yes | no | 1.623 |
| popular | 2334 | chair | person|tv | tv | 0.061 | 0.550 | -0.488 | yes | yes | yes | yes | no | 1.837 |
| popular | 1020 | chair | couch|cup|laptop|person | laptop | 0.117 | 0.548 | -0.431 | yes | yes | yes | yes | no | 0.372 |
| popular | 1512 | cup | chair|dining table | chair | 0.051 | 0.547 | -0.496 | yes | yes | yes | yes | no | 1.957 |
| adversarial | 1110 | baseball glove | baseball bat|person | baseball bat | 0.089 | 0.829 | -0.741 | yes | yes | yes | yes | no | -2.410 |
| adversarial | 1232 | potted plant | book|vase | vase | 0.053 | 0.789 | -0.736 | yes | yes | yes | yes | no | 1.248 |
| adversarial | 804 | traffic light | car|fire hydrant|stop sign | fire hydrant | 0.064 | 0.772 | -0.708 | yes | yes | yes | yes | no | -3.127 |
| adversarial | 528 | bowl | broccoli|chair | broccoli | 0.096 | 0.703 | -0.607 | yes | yes | yes | yes | no | -2.855 |
| adversarial | 2446 | cup | dining table|fork|knife|pizza|wine glass | pizza | 0.113 | 0.694 | -0.581 | yes | yes | yes | yes | no | -3.937 |
| adversarial | 2016 | chair | couch|person|tv | tv | 0.119 | 0.678 | -0.559 | yes | yes | yes | yes | no | -4.473 |
