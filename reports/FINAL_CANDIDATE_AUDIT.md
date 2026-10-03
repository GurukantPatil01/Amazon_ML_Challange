# Final Candidate Generation Audit Report — Amazon ML Challenge 2026

## 1. Executive Summary
- **Total Test Source 1 Queries:** 1,732,544
- **Total Test Targets (S2 + S3):** 9,969,589
- **Total Candidate Pairs Fed to Matcher:** 34,648,342
- **Reduction Ratio vs Cartesian Product:** 99.999799%
- **Candidate Blocking Strategy:** Union F (9 Inverted Index Blocks)
- **Multi-Block Support Cap:** K = 20 per S1

---

## 2. Cardinality Distribution Statistics

| Metric | Value |
| :--- | :---: |
| **Total S1 Queries Evaluated** | **1,732,544** |
| **Total Candidate Pairs Generated** | **34,648,342** |
| **Average Candidates / S1** | **20.00** |
| **Median Candidates / S1** | **20.0** |
| **95th Percentile (P95)** | **20.0** |
| **99th Percentile (P99)** | **20.0** |
| **Maximum Candidates / S1** | **20** |
| **Singletons Receiving Zero Candidates** | **1 (0.00%)** |
| **Candidate Reduction Ratio** | **0.99999799** |

---

## 3. Compliance and Consistency Verification
- **Every S1 Represented:** 100% (1,732,544 / 1,732,544 rows present)
- **Valid Candidate Prefixes:** All candidate IDs strictly conform to `S2-*` or `S3-*`
- **Zero Self-Matches:** No `S1-*` IDs exist in candidate outputs
- **Zero Duplicates:** All candidate lists are strictly deduplicated and deterministically sorted
- **Predictive Containment:** 100% of all predicted matches are strictly contained within candidate set
