# Tweezcat synthetic/real eval report

Methods from PLAN_SYNTHETIC_REAL_ALIGNMENT.md. Same train/test split for all.

## Summary by method

| Method | Pooled accuracy | Mean per-exp accuracy | N test experiments |
|--------|-----------------|------------------------|-------------------|
| approach4_rules | 0.6093 | 0.6090 | 38 |

## Predicted good vs bad (and TP/TN/FP/FN)

Shows we are not just predicting one class. Good bead = positive class.

| Method | Predicted good | Predicted bad | TP | TN | FP | FN |
|--------|----------------|---------------|----|----|----|----|
| approach4_rules | 484 | 6640 | 142 | 4199 | 342 | 2441 |

---

## Per-experiment accuracy

| experiment_id | source | approach4_rules |
|---|---|---|
| 20250411_TX_OF_of1437_wtrnap_7pN_p37 | TX_OF | 0.382 |
| 20250513_TX_OF_7pN_Pb37_wtholo_1437_r250 | TX_OF | 0.434 |
| 20250513_TX_OF_7pN_Pb37_wtholo_1437_r50 | TX_OF | 0.310 |
| 20250514_TX_OF_7pN_Pb37_wtholo_1437_ga2 | TX_OF | 0.336 |
| 20250514_TX_OF_7pN_wtholo_1437_r1000ga2 | TX_OF | 0.317 |
| 20250625_1418_TX_wt_OF_3pN_Pb37_r1000 | TX_OF | 0.496 |
| 20250625_1709_TX_wt_OF_3pN_Pb37 | TX_OF | 0.580 |
| 20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2 | TX_OF | 0.442 |
| 20250630_1255_TX_wt_OF_10pN_Pb37 | TX_OF | 0.510 |
| 20250630_1559_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.730 |
| 20250701_1005_TX_wt_OF_10pN_Pb37_r1000 | TX_OF | 0.759 |
| 20250703_1622_TX_OF_7pN_Pb37_ng1000na200ga2 | TX_OF | 0.463 |
| 20250705_1227_TX_wtholo_AF_7pN_Pb37 | TX_AF | 0.551 |
| 20250705_1523_TX_wtholo_AF_3pN_Pb37 | TX_AF | 0.667 |
| 20250707_1254_TX_wtholo_AF_7pN_Pb37_r1000 | TX_AF | 0.640 |
| 20250708_1033_TX_wtholo_AF_10pN_Pb37 | TX_AF | 0.783 |
| 20250715_1750_TX_wtholo_OF_7pN_Pb37_NTPs50_na200 | TX_OF | 0.750 |
| 20250716_1519_TX_wt_OF_7pN_Pb37_NTPs50_r50 | TX_OF | 0.937 |
| 20250716_1844_TX_wt_OF_7pN_Pb37_NTPs50_ga2 | TX_OF | 0.723 |
| 20250717_1058_TX_wt_OF_7pN_Pb37_NTPs50_r50ga2 | TX_OF | 0.705 |
| 20250717_1448_TX_wt_OF_7pN_Pb37_NTPs50_r50na200ga2 | TX_OF | 0.726 |
| 20250717_1754_TX_wt_OF_7pN_Pb37_NTPs100_r500ga2 | TX_OF | 0.685 |
| 20251204_2006_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.719 |
| 20251205_1525_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.755 |
| 20251205_1820_core_RTC_30nM_nsp13_NTPs_500uM_contr | dsRNA | 0.679 |
| 20260214_1439_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_1 | ssRNA | 0.633 |
| 20260214_1826_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_1 | ssRNA | 0.566 |
| TXTL_OF_7pN_37_wtholo_fp3h3_PUREfull | TX_OF | 0.638 |
| TXTL_OF_7pN_37_wtholo_fp3h3_PUREfull_r1000ga2 | TX_OF | 0.685 |
| TXTL_OF_7pN_wtholo_59_fp3h3_37_PUREfull_r1000 | TX_OF | 0.542 |
| TXTL_OF_fp3h3_61_wtholo_7pN_37_PURE_I_only | TX_OF | 0.690 |
| TXTL_OF_fp3h3_61_wtholo_7pN_37_PURE_I_only_r1000 | TX_OF | 0.632 |
| TXTL_OF_fp3h3_61_wtholo_7pN_37_PURE_full_r1000 | TX_OF | 0.508 |
| experiment | ssRNA | 0.529 |

## Low-accuracy experiments (< 0.6)

- **20250705_1227_TX_wtholo_AF_7pN_Pb37** (approach4_rules): 0.551
- **20250625_1709_TX_wt_OF_3pN_Pb37** (approach4_rules): 0.580
- **20250627_1406_TX_wt_OF_7pN_Pb37_r1000nusa200ga2** (approach4_rules): 0.442
- **20250630_1255_TX_wt_OF_10pN_Pb37** (approach4_rules): 0.510
- **20250703_1622_TX_OF_7pN_Pb37_ng1000na200ga2** (approach4_rules): 0.463
- **TXTL_OF_7pN_wtholo_59_fp3h3_37_PUREfull_r1000** (approach4_rules): 0.542
- **TXTL_OF_fp3h3_61_wtholo_7pN_37_PURE_full_r1000** (approach4_rules): 0.508
- **20260214_1826_core_RTC_nsp9_2uM_nsp13_30nM_nsp10_14_sep_25nM_nsp10_spikeup_1uM_NTPs_500uM_2pCMeCTP_1000uM_AD_test_repeat_run** (approach4_rules): 0.566
- **experiment** (approach4_rules): 0.529
- **20250411_TX_OF_of1437_wtrnap_7pN_p37** (approach4_rules): 0.382
- **20250513_TX_OF_7pN_Pb37_wtholo_1437_r250** (approach4_rules): 0.434
- **20250513_TX_OF_7pN_Pb37_wtholo_1437_r50** (approach4_rules): 0.310
- **20250514_TX_OF_7pN_Pb37_wtholo_1437_ga2** (approach4_rules): 0.336
- **20250514_TX_OF_7pN_wtholo_1437_r1000ga2** (approach4_rules): 0.317
- **20250625_1418_TX_wt_OF_3pN_Pb37_r1000** (approach4_rules): 0.496
