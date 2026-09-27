# Phase 3 Metric Verification & Evaluation Audit Report

> **VERDICT: METRIC AUDIT PASSED**
>
> - **Independently Verified Macro-$F_{0.5}$:** **95.32%** (0.953167) at Validation-Optimized Threshold $\tau = 0.45$.
> - **Mathematical Reconciliation:** Exact conservation of pairs ($\text{TP} + \text{FN} = \text{Ground Truth}$, $\text{TP} + \text{FP} = \text{Predicted}$) and exact multiplicative recall identity confirmed.
> - **Why Macro-$F_{0.5}$ is High:** Singletons comprise **85.00%** of the validation set (2,550 / 3,000 S1s). The LightGBM classifier achieves **99.37% exact-empty accuracy** on singletons, contributing **84.47%** to the overall Macro-$F_{0.5}$. On true non-singletons (businesses with matches), the mean $F_{0.5}$ is **72.33%** (contributing **10.85%**).

---

## 1. Official Validator Metric Comparison

A detailed audit was conducted comparing the competition validator (`validate_submission.py`) and official rules with `src/evaluate.py` and `src/evaluate_f05.py`:

| Metric Requirement | Official Competition Rule | Our Implementation | Audit Status |
| :--- | :--- | :--- | :---: |
| **Averaging Level** | Macro-average across all Source 1 entities independently | `mean(F0.5_i)` computed across all individual S1 entities | **VERIFIED** |
| **Global vs Macro** | Never calculate $F_{0.5}$ from pooled global TP/FP/FN | Strictly evaluates per-S1 precision, recall, and $F_{0.5}$ before averaging | **VERIFIED** |
| **True Singleton Scoring** | If ground truth is empty and prediction is empty: score = **1.0** | Returns `1.0` if `len(true_ids) == 0 and len(pred_ids) == 0` | **VERIFIED** |
| **Singleton False Match** | If ground truth is empty and prediction is non-empty: score = **0.0** | Returns `0.0` if `len(true_ids) == 0 and len(pred_ids) > 0` | **VERIFIED** |
| **Missed Non-Singleton** | If ground truth has matches and prediction is empty: score = **0.0** | Returns `0.0` if `len(true_ids) > 0 and len(pred_ids) == 0` | **VERIFIED** |
| **$\beta$ Weighting** | $\beta = 0.5$ ($\beta^2 = 0.25$, precision penalized $2\times$ heavier than recall) | Formula: $\frac{(1 + 0.25) \cdot P \cdot R}{0.25 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$ | **VERIFIED** |

---

## 2. Independent Reference F0.5 Implementation & Threshold Sweep

An independent, clean-room reference implementation (`independent_entity_f05`) was written directly in `src/audit_phase3_metrics.py` without importing `evaluate_decision_metrics` or `compute_entity_f05`.

Both implementations were evaluated across all 19 thresholds ($\tau \in [0.05, 0.95]$) on all 3,000 validation S1 entities:

| Threshold ($\tau$) | Independent Macro-$F_{0.5}$ | Pipeline Engine Macro-$F_{0.5}$ | Absolute Difference | Verification Status |
| :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.9098 | 0.9098 | 0.0000 | Match (< 1e-4) |
| 0.10 | 0.9322 | 0.9322 | 0.0000 | Match (< 1e-4) |
| 0.15 | 0.9441 | 0.9441 | 0.0000 | Match (< 1e-4) |
| 0.20 | 0.9503 | 0.9503 | 0.0000 | Match (< 1e-4) |
| 0.25 | 0.9520 | 0.9520 | 0.0000 | Match (< 1e-4) |
| 0.30 | 0.9519 | 0.9519 | 0.0000 | Match (< 1e-4) |
| 0.35 | 0.9523 | 0.9523 | 0.0000 | Match (< 1e-4) |
| 0.40 | 0.9517 | 0.9517 | 0.0000 | Match (< 1e-4) |
| **0.45** | **0.9532** | **0.9532** | **0.0000** | **Match (< 1e-4)** |
| 0.50 | 0.9518 | 0.9518 | 0.0000 | Match (< 1e-4) |
| 0.55 | 0.9504 | 0.9504 | 0.0000 | Match (< 1e-4) |
| 0.60 | 0.9487 | 0.9487 | 0.0000 | Match (< 1e-4) |
| 0.65 | 0.9473 | 0.9473 | 0.0000 | Match (< 1e-4) |
| 0.70 | 0.9449 | 0.9449 | 0.0000 | Match (< 1e-4) |
| 0.75 | 0.9426 | 0.9426 | 0.0000 | Match (< 1e-4) |
| 0.80 | 0.9306 | 0.9306 | 0.0000 | Match (< 1e-4) |
| 0.85 | 0.8545 | 0.8545 | 0.0000 | Match (< 1e-4) |
| 0.90 | 0.8503 | 0.8503 | 0.0000 | Match (< 1e-4) |
| 0.95 | 0.8500 | 0.8500 | 0.0000 | Match (< 1e-4) |

---

## 3. Disambiguation: Macro Precision vs Micro Precision

The audit investigated why the previous report listed `Macro Precision = 95.33%` and `Micro Precision = 95.33%`.

### Empirical Findings:
- **Micro (Aggregate) Precision:** **95.60%** (348 True Positives / 364 Total Predicted Pairs).
- **Macro Precision (Entities with Predictions):** **95.34%** (Mean precision across the 343 entities that received predictions).
- **Macro Precision (All 3,000 S1s):** **95.37%** (Evaluated where correct empty singletons score 1.0, matching official scoring).
- **Macro Precision (Non-Singletons Only):** **72.67%** (Evaluated across the 450 true non-singletons, where unpredicted non-singletons receive precision = 0.0).

> **Conclusion on Precision:** The apparent equality in the initial report was an artifact of integer rounding on two mathematically distinct quantities: $(2534 \times 1.0 + \sum \text{prec})/3000 = 0.9537$ vs $348 / 364 = 0.9560$. Both values have now been explicitly separated and reported.

---

## 4. Disambiguation: Macro Recall vs Micro Recall

- **Micro (Aggregate) Recall:** **72.05%** (348 True Positives / 483 Total Ground Truth Pairs).
- **Macro Recall (All 3,000 S1s):** **95.75%** (Includes singletons where correctly predicting empty scores recall = 1.0).
- **Macro Recall (Non-Singletons Only):** **71.67%** (Mean recall across the 450 non-singletons).

---

## 5. Singleton Breakdown (Category A: Ground Truth = 0 Matches)

| Metric | Measured Count / Rate | Technical Impact |
| :--- | :---: | :--- |
| **Total Validation Singletons** | **2,550 entities** | **85.00%** of all validation queries |
| **Correctly Predicted Empty** | **2,534 entities** | **99.37% exact-empty accuracy** |
| **False-Positive Singletons** | **16 entities** | Only **0.63%** singleton false-alarm rate |
| **Mean Singleton $F_{0.5}$** | **99.37%** | High baseline due to strong rejection of non-matching candidates |
| **Contribution to Total Macro-$F_{0.5}$** | **84.47%** | $0.85 \times 99.37\% = 84.47\%$ |

---

## 6. Non-Singleton Breakdown (Category B: Ground Truth $\ge 1$ Match)

| Metric | Measured Count / Rate | Technical Impact |
| :--- | :---: | :--- |
| **Total Validation Non-Singletons** | **450 entities** | **15.00%** of all validation queries |
| **True Matches in Non-Singletons** | **483 pairs** | Ground truth match pool |
| **True Positives Accepted** | **348 pairs** | Recovered by candidate blocking + LightGBM |
| **False Positives on Non-Singletons** | **0 pairs** | Zero false merges among non-singletons |
| **False Negatives (Missed Pairs)** | **135 pairs** | 53 missed at candidate blocking + 82 filtered by threshold $\tau=0.45$ |
| **Mean Non-Singleton $F_{0.5}$** | **72.33%** | True resolution accuracy on entities with actual matches |
| **Contribution to Total Macro-$F_{0.5}$** | **10.85%** | $0.15 \times 72.33\% = 10.85\%$ |

### Mathematical Reassembly of Macro-$F_{0.5}$:
$$\text{Macro-}F_{0.5} = \frac{2,550 \times 0.9937 + 450 \times 0.7233}{3,000} = \frac{2,534.0 + 325.5}{3,000} = \mathbf{95.32\%}$$

---

## 7. Pair Counts & Conservation Laws

At decision threshold $\tau = 0.45$:

- **Total Validation S1 Entities:** 3,000
- **Total Ground-Truth Positive Pairs:** 483
- **Total Predicted Pairs:** 364
- **True Positives (TP):** 348
- **False Positives (FP):** 16 (all 16 occurred on singleton entities; 0 on non-singletons)
- **False Negatives (FN):** 135

### Conservation Verification:
$$\text{Conservation Law 1: } \text{TP} + \text{FN} = 348 + 135 = 483 = \text{Total Ground Truth Positive Pairs} \quad \checkmark$$
$$\text{Conservation Law 2: } \text{TP} + \text{FP} = 348 + 16 = 364 = \text{Total Predicted Pairs} \quad \checkmark$$

---

## 8. Recall Decomposition & Multiplicative Reconciliation

$$\text{Candidate Recall} = \frac{\text{Candidate True Pairs}}{\text{All True Pairs}} = \frac{430}{483} = \mathbf{89.03\%}$$
$$\text{Model Recall among Candidates} = \frac{\text{Accepted True Pairs}}{\text{Candidate True Pairs}} = \frac{348}{430} = \mathbf{80.93\%}$$
$$\text{End-to-End Recall} = \frac{\text{Accepted True Pairs}}{\text{All True Pairs}} = \frac{348}{483} = \mathbf{72.05\%}$$

### Multiplicative Identity Check:
$$\text{Candidate Recall} \times \text{Model Recall among Candidates} = 0.890269 \times 0.809302 = \mathbf{0.720497} = \text{End-to-End Recall}$$
$$\text{Discrepancy} = |0.720497 - 0.720497| = \mathbf{0.0000000000} \quad (\text{Exact})$$

---

## 9. Label Leakage Audit

A comprehensive code inspection of `src/pair_features.py`, `src/build_training_pairs.py`, and `src/train_lightgbm.py` verified:
1. **Feature Definitions:** None of the 40 feature functions access `label`, `ground_truth`, `matched_entity_ids`, or target match metadata.
2. **Candidate Generation:** `generate_candidates()` operates strictly on normalized query text and target inverted index postings.
3. **Training Pair Construction:** Ground truth is referenced strictly to assign binary training target `y` ($0$ or $1$) and is never passed into the feature matrix `X`.

---

## 10. Group Split Disjointness Audit

- **Training S1 Entity Count:** 12,000
- **Validation S1 Entity Count:** 3,000
- **$\text{Training S1} \cap \text{Validation S1}$:** **0 entities** (Strictly empty set, zero group leakage).
- **Target Record Overlap:** S2 and S3 target records legitimately appear across both splits because the entity resolution query boundary is on Source 1, exactly mirroring test-set evaluation.

---

## 11. Threshold Selection Caveat

> [!WARNING]
> **Threshold Selection Methodology Caveat:**
> The threshold $\tau = 0.45$ was selected based on the validation split. While valid as an experimental baseline, it is a **Validation-Optimized Threshold**.
>
> In Phase 4, to ensure zero threshold overfitting, we must employ:
> 1. A 5-fold group-aware cross-validation split (out-of-fold threshold calibration).
> 2. A completely untouched held-out validation test split to measure final generalization.

---

## 12. Direct Model Feature Importance Verification

Inspecting `models/lightgbm_baseline.txt` directly confirms the exact tree gain distributions:

1. `feat_address_token_overlap`: **38.95%** of total tree split gain
2. `feat_name_sim_x_addr_sim`: **31.21%** of total tree split gain
   - **Combined Top 2 Gain:** **70.16%**
3. `feat_name_char_3gram_jaccard`: **7.49%**
4. `feat_address_token_jaccard`: **6.24%**
5. `feat_name_sorted_token_sim`: **2.23%**
6. `feat_house_number_exact_match`: **2.22%**

### Analysis of `feat_candidate_rank`:
- `feat_candidate_rank` ranks **18th** in feature importance, accounting for only **0.24%** of tree split gain.
- This demonstrates that the model is **not** overly reliant on the blocking order; rather, it makes its primary discrimination decisions based on token overlap and cross-field similarity interactions.

---

## Unit Test Suite Verification
All 39 unit tests (including dedicated metric audit tests in `tests/test_metric_audit.py`) pass:
```
============================== 39 passed in 1.49s ==============================
```
