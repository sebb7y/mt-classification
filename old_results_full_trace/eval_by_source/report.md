# Tweezcat synthetic/real eval report

Methods from PLAN_SYNTHETIC_REAL_ALIGNMENT.md. Same train/test split for all.

## Summary by method

| Method | Pooled accuracy | Mean per-exp accuracy | N test experiments |
|--------|-----------------|------------------------|-------------------|
| approach1_scaled_dt | 0.6479 | 0.6374 | 14 |
| approach4_rules | 0.6437 | 0.6544 | 14 |
| baseline_dt | 0.6599 | 0.6608 | 14 |
| baseline_lr | 0.6808 | 0.6931 | 14 |
| baseline_rf | 0.6723 | 0.6725 | 14 |

## Predicted good vs bad (and TP/TN/FP/FN)

Shows we are not just predicting one class. Good bead = positive class.

| Method | Predicted good | Predicted bad | TP | TN | FP | FN |
|--------|----------------|---------------|----|----|----|----|
| approach1_scaled_dt | 679 | 2397 | 289 | 1704 | 390 | 693 |
| approach4_rules | 336 | 2740 | 111 | 1869 | 225 | 871 |
| baseline_dt | 646 | 2430 | 291 | 1739 | 355 | 691 |
| baseline_lr | 0 | 3076 | 0 | 2094 | 0 | 982 |
| baseline_rf | 420 | 2656 | 197 | 1871 | 223 | 785 |

---

## Per-experiment accuracy

| experiment_id | source | approach1_scaled_dt | approach4_rules | baseline_dt | baseline_lr | baseline_rf |
|---|---|---|---|---|---|---|
| 20250705_1227_TX_wtholo_AF_7pN_Pb37 | TX_AF | 0.502 | 0.551 | 0.506 | 0.506 | 0.502 |
| 20250705_1523_TX_wtholo_AF_3pN_Pb37 | TX_AF | 0.557 | 0.667 | 0.548 | 0.627 | 0.583 |
| 20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000 | TX_AF | 0.611 | 0.640 | 0.613 | 0.630 | 0.623 |
| 20250708_1033_TX_wtholo_AF_10pN_Pb37 | TX_AF | 0.435 | 0.783 | 0.609 | 0.783 | 0.609 |
| 20251204_2006_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.640 | 0.719 | 0.660 | 0.754 | 0.704 |
| 20251205_1525_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.718 | 0.755 | 0.732 | 0.782 | 0.755 |
| 20251205_1820_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.620 | 0.679 | 0.636 | 0.717 | 0.690 |
| 20260214_1439_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_1 | ssRNA | 0.665 | 0.633 | 0.679 | 0.683 | 0.674 |
| 20260214_1826_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_1 | ssRNA | 0.697 | 0.566 | 0.708 | 0.652 | 0.715 |
| experiment | ssRNA | 0.744 | 0.684 | 0.752 | 0.804 | 0.784 |

## Low-accuracy experiments (< 0.6)

- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (approach1_scaled_dt): 0.502
- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (approach4_rules): 0.551
- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (baseline_dt): 0.506
- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (baseline_lr): 0.506
- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (baseline_rf): 0.502
- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (approach1_scaled_dt): 0.557
- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (baseline_dt): 0.548
- **20250705_1523_TX_wtholo_AF_3pN_Pb37** (baseline_rf): 0.583
- **20250708_1033_TX_wtholo_AF_10pN_Pb37** (approach1_scaled_dt): 0.435
- **20260214_1826_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_14_sep_25nM_nsp10_spikeup_1uM_NTPs_500uM_2pCMeCTP_1000uM_AD_test_repeat_run** (approach4_rules): 0.566
