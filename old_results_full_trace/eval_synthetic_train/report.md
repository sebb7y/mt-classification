# Tweezcat synthetic/real eval report

Methods from PLAN_SYNTHETIC_REAL_ALIGNMENT.md. Same train/test split for all.

## Summary by method

| Method | Pooled accuracy | Mean per-exp accuracy | N test experiments |
|--------|-----------------|------------------------|-------------------|
| approach1_scaled_dt | 0.6174 | 0.6368 | 8 |
| approach4_rules | 0.6037 | 0.6186 | 8 |
| baseline_dt | 0.6518 | 0.6637 | 8 |
| baseline_lr | 0.6312 | 0.6505 | 8 |
| baseline_rf | 0.6535 | 0.6661 | 8 |

## Predicted good vs bad (and TP/TN/FP/FN)

Shows we are not just predicting one class. Good bead = positive class.

| Method | Predicted good | Predicted bad | TP | TN | FP | FN |
|--------|----------------|---------------|----|----|----|----|
| approach1_scaled_dt | 72 | 1674 | 24 | 1054 | 48 | 620 |
| approach4_rules | 138 | 1608 | 45 | 1009 | 93 | 599 |
| baseline_dt | 372 | 1374 | 204 | 934 | 168 | 440 |
| baseline_lr | 18 | 1728 | 9 | 1093 | 9 | 635 |
| baseline_rf | 377 | 1369 | 208 | 933 | 169 | 436 |

---

## Per-experiment accuracy

| experiment_id | source | approach1_scaled_dt | approach4_rules | baseline_dt | baseline_lr | baseline_rf |
|---|---|---|---|---|---|---|
| 20250514_TX_OF_7pN_wtholo_1437_r1000ga2 | TX_OF | 0.325 | 0.317 | 0.558 | 0.325 | 0.575 |
| 20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2 | TX_OF | 0.488 | 0.442 | 0.695 | 0.472 | 0.695 |
| 20250630_1559_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.757 | 0.730 | 0.730 | 0.757 | 0.730 |
| 20250701_1005_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.806 | 0.759 | 0.731 | 0.833 | 0.731 |
| 20250705_1523_TX_wtholo_AF_3pN_Pb37 | TX_AF | 0.588 | 0.667 | 0.539 | 0.627 | 0.539 |
| 20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000 | TX_AF | 0.603 | 0.640 | 0.569 | 0.630 | 0.572 |
| experiment | ssRNA | 0.724 | 0.710 | 0.742 | 0.756 | 0.742 |

## Low-accuracy experiments (< 0.6)

- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (approach1_scaled_dt): 0.588
- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (baseline_dt): 0.539
- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (baseline_rf): 0.539
- **20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000** (baseline_dt): 0.569
- **20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000** (baseline_rf): 0.572
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (approach1_scaled_dt): 0.325
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (approach4_rules): 0.317
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (baseline_dt): 0.558
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (baseline_lr): 0.325
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (baseline_rf): 0.575
- **20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2** (approach1_scaled_dt): 0.488
- **20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2** (approach4_rules): 0.442
- **20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2** (baseline_lr): 0.472
