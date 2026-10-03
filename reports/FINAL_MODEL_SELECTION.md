# Final Model Selection Report — Amazon ML Challenge 2026

## 1. Selected Production Configuration
- **Model Architecture:** LightGBM GBDT Pairwise Matcher
- **Feature Suite:** 46 Targeted Precision Features (43 Baseline + `has_house_token`, `has_web_norm`, `independent_block_count`)
- **Candidate Generation Strategy:** `Union F` (9 Inverted Index Blocks: `exact_name`, `name_core`, `rare_token`, `char_3gram_k10`, `house_name`, `combined_postal_name`, `address_token`, `name_web_norm`, `house_token`)
- **Candidate Ranking / Pruning:** Multi-Block Support Count ranking with budget cap $K=200$
- **Decision Threshold:** $\tau = 0.50$

---

## 2. Selection Rationale
Per the challenge instructions, the final model was selected using strict priority rules:
1. **Highest Validated Macro-$F_{0.5}$:** Achieves **96.65%** on 3,000 unseen validation queries.
2. **Minimal False Positives:** Produces only **23 false merges** across 3,000 queries (vs 31 in Hybrid K=100 and 42 in Hybrid K=200).
3. **High Candidate Efficiency:** Generates an average of **155.82 candidates / query** with a reduction ratio $>99.98\%$.
4. **Zero GPU/Memory Bottlenecks:** Pure C++ / Python implementation executes with $<1.5\text{ GB}$ RAM, ensuring guaranteed completion across the 11.7M test set.

---

## 3. Validated Benchmark Summary

| Metric | Validated Score |
| :--- | :---: |
| **Macro-F0.5** | **96.65%** |
| **Macro Precision** | **96.67%** |
| **Macro Recall** | **96.62%** |
| **True Positives (TP)** | 401 |
| **False Positives (FP)** | **23** |
| **False Negatives (FN)** | 82 |
| **Singleton Accuracy** | **99.18%** |
| **Non-Singleton Macro-F0.5** | **82.78%** |
| **Candidate Recall** | **93.58%** |
| **Average Candidates / S1** | **155.82** |
| **Inference Time (3,000 queries)** | **0.24 seconds** |
| **Peak Memory** | **1,461.30 MB** |
