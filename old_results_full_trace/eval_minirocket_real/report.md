# Tweezcat synthetic/real eval report

Methods from PLAN_SYNTHETIC_REAL_ALIGNMENT.md. Same train/test split for all.

## Summary by method

| Method | Pooled accuracy | Mean per-exp accuracy | N test experiments |
|--------|-----------------|------------------------|-------------------|
| minirocket | 0.6214 | 0.6451 | 8 |

## Predicted good vs bad (and TP/TN/FP/FN)

Shows we are not just predicting one class. Good bead = positive class.

| Method | Predicted good | Predicted bad | TP | TN | FP | FN |
|--------|----------------|---------------|----|----|----|----|
| minirocket | 53 | 1693 | 18 | 1067 | 35 | 626 |

---

## Per-experiment accuracy

| experiment_id | source | minirocket |
|---|---|---|
| 20250514_TX_OF_7pN_wtholo_1437_r1000ga2 | TX_OF | 0.383 |
| 20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2 | TX_OF | 0.466 |
| 20250630_1559_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.757 |
| 20250701_1005_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.824 |
| 20250705_1523_TX_wtholo_AF_3pN_Pb37 | TX_AF | 0.588 |
| 20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000 | TX_AF | 0.625 |
| experiment | ssRNA | 0.733 |

## Low-accuracy experiments (< 0.6)

- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (minirocket): 0.588
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (minirocket): 0.383
- **20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2** (minirocket): 0.466
