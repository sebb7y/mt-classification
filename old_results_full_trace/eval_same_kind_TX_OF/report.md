# Tweezcat synthetic/real eval report

Methods from PLAN_SYNTHETIC_REAL_ALIGNMENT.md. Same train/test split for all.

## Summary by method

| Method | Pooled accuracy | Mean per-exp accuracy | N test experiments |
|--------|-----------------|------------------------|-------------------|
| approach1_scaled_dt | 0.7385 | 0.6138 | 5 |
| approach4_rules | 0.7162 | 0.5776 | 5 |
| baseline_dt | 0.7695 | 0.7190 | 5 |
| baseline_lr | 0.7398 | 0.6094 | 5 |
| baseline_rf | 0.8488 | 0.7823 | 5 |

## Predicted good vs bad (and TP/TN/FP/FN)

Shows we are not just predicting one class. Good bead = positive class.

| Method | Predicted good | Predicted bad | TP | TN | FP | FN |
|--------|----------------|---------------|----|----|----|----|
| approach1_scaled_dt | 47 | 760 | 23 | 573 | 24 | 187 |
| approach4_rules | 27 | 780 | 4 | 574 | 23 | 206 |
| baseline_dt | 218 | 589 | 121 | 500 | 97 | 89 |
| baseline_lr | 0 | 807 | 0 | 597 | 0 | 210 |
| baseline_rf | 180 | 627 | 134 | 551 | 46 | 76 |

---

## Per-experiment accuracy

| experiment_id | source | approach1_scaled_dt | approach4_rules | baseline_dt | baseline_lr | baseline_rf |
|---|---|---|---|---|---|---|
| 20250513_TX_OF_7pN_Pb37_wtholo_1437_r250 | TX_OF | 0.426 | 0.434 | 0.664 | 0.434 | 0.746 |
| 20250514_TX_OF_7pN_wtholo_1437_r1000ga2 | TX_OF | 0.317 | 0.317 | 0.667 | 0.325 | 0.742 |
| 20250630_1255_TX_wt_OF_10pN_Pb37 | TX_OF | 0.653 | 0.510 | 0.694 | 0.633 | 0.694 |
| 20250716_1519_TX_wt_OF_7pN_Pb37_NTPs50_r50 | TX_OF | 0.960 | 0.937 | 0.846 | 0.965 | 0.937 |
| TXTL_OF_fp3h3_61_wtholo_7pN_37_PURE_I_only | TX_OF | 0.713 | 0.690 | 0.724 | 0.690 | 0.793 |

## Low-accuracy experiments (< 0.6)

- **20250513_TX_OF_7pN_Pb37_wtholo_1437_r250** (approach1_scaled_dt): 0.426
- **20250513_TX_OF_7pN_Pb37_wtholo_1437_r250** (approach4_rules): 0.434
- **20250513_TX_OF_7pN_Pb37_wtholo_1437_r250** (baseline_lr): 0.434
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (approach1_scaled_dt): 0.317
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (approach4_rules): 0.317
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (baseline_lr): 0.325
- **20250630_1255_TX_wt_OF_10pN_Pb37** (approach4_rules): 0.510
