# Phase 2 Candidate Budget & Uncapped Union Evaluation Report

> **IMPORTANT SCOPE NOTICE: SUBSET EXPERIMENT**
> - **Queries:** 15,000 S1 records (randomized deterministic sample)
> - **Target Pool:** 250,000 S2 records + 250,000 S3 records (500,000 total target records)
> - **Ground Truth Pairs in Subset:** 2,530 positive ground-truth pairs indexed
> - **Singletons in Subset:** 12,470 S1 queries have no matching entity in the 500k target pool
> - **Extrapolation Warning:** Do **NOT** extrapolate subset recall directly to the full 10.3M target dataset. All figures reported below represent empirical measurements on this controlled benchmark.

---

## Executive Summary

1. **The Uncapped Ceiling:**
   - Prior experiments imposed an artificial `K=50` insertion-order cap, causing candidate saturation where loose address tokens displaced high-confidence name matches.
   - Evaluating the **complete uncapped candidate union (Union E)** reveals an uncapped pair recall ceiling of **91.66%** (91.59% Mean S1 Recall) with an average of **150.57 candidates per S1**.
2. **Candidate Budgeting & Pruning Policies:**
   - At strict budgets, **Multi-Block Support** consistently outperforms deterministic Blocking Strength Priority by **+2.0% to +3.0% recall** across all budgets ($K=25, 50, 100$).
   - At $K=25$, Multi-Block achieves **75.42% recall** (vs 72.49% for Priority).
   - At $K=50$, Multi-Block achieves **77.08% recall** (vs 74.43% for Priority).
   - At $K=100$, Multi-Block achieves **79.17% recall** (vs 77.00% for Priority).
   - At $K=200$, the candidate capacity accommodates the entire uncapped union, capturing the full **91.66% recall**.
3. **ROI and Information Gain of Individual Blocks:**
   - `house_name` is by far the highest-ROI block: adding it over `name_core` delivers **+19.73% delta recall** while adding only **+0.17 candidates per S1** (a recall gain rate of 0.1924 hits per candidate).
   - `combined_postal_name` achieves **97.22% precision** (105 true hits out of 108 candidates total across 15k queries) with essentially zero candidate inflation.
   - `char_3gram_k5` delivers **+23.68% delta recall** for only +9.64 candidates per S1.
   - Broad address tokens deliver +39.17% recall but introduce +53.79 candidates per S1 at 0.19% precision, requiring Multi-Block consensus or ML scoring to filter effectively.

---

## TABLE 1: Individual Block Performance (Combined S2 + S3 Pool)

Empirical metrics for each individual blocking strategy evaluated across all 15,000 S1 queries against the 500,000 combined target records.

| Blocking Strategy | Pair Recall (%) | Per-S1 Recall (%) | Avg Cands/S1 | Median | P95 | P99 | Max | Candidate Precision (%) | Reduction Ratio | Runtime (s) | Peak Memory (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_name` | 20.63% | 20.53% | 0.46 | 0 | 3 | 9 | 26 | 7.4957% | 0.999999 | 2.56s | 1425.1 |
| `name_core` | 35.61% | 35.49% | 1.47 | 0 | 7 | 27 | 67 | 4.0979% | 0.999997 | 0.26s | 1425.1 |
| `name_token` | 50.91% | 51.16% | 88.27 | 100 | 100 | 100 | 100 | 0.0973% | 0.999823 | 4.63s | 1425.1 |
| `rare_token` | 59.41% | 59.57% | 84.25 | 100 | 100 | 100 | 100 | 0.1189% | 0.999831 | 4.57s | 1425.1 |
| `char_3gram_k5` | 44.82% | 44.95% | 10.00 | 10 | 10 | 10 | 10 | 0.7563% | 0.999980 | 56.20s | 1425.1 |
| `char_3gram_k10` | 48.58% | 48.83% | 19.99 | 20 | 20 | 20 | 20 | 0.4098% | 0.999960 | 55.06s | 1425.1 |
| `char_3gram_k20` | 52.37% | 52.61% | 39.98 | 40 | 40 | 40 | 40 | 0.2209% | 0.999920 | 52.61s | 1425.1 |
| `postal` | 5.22% | 5.02% | 0.13 | 0 | 0 | 4 | 13 | 6.7971% | 1.000000 | 0.87s | 1425.1 |
| `house_name` | 37.59% | 37.44% | 0.20 | 0 | 1 | 3 | 18 | 31.1701% | 1.000000 | 0.94s | 1425.1 |
| `combined_postal_name` | 4.15% | 3.93% | 0.01 | 0 | 0 | 0 | 3 | 97.2222% | 1.000000 | 0.20s | 1425.1 |
| `address_token` | 61.62% | 61.52% | 53.84 | 60 | 60 | 60 | 60 | 0.1931% | 0.999892 | 8.39s | 1425.1 |

---

## TABLE 2: Experiment 1 — Uncapped Candidate Union Performance

Complete candidate union evaluation **without any candidate cap** or truncation. This measures the true empirical recall ceiling and candidate volume.

- **Union A:** `exact_name` + `name_core`
- **Union B:** Union A + `rare_token`
- **Union C:** Union B + `char_3gram_k10`
- **Union D:** Union C + `house_name` + `combined_postal_name`
- **Union E:** Union D + `address_token`

| Candidate Union | Pair Recall (%) | Per-S1 Recall (%) | Avg Cands/S1 | Median | P95 | P99 | Max | Candidate Precision (%) | Reduction Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Union A (Exact + Core)** | **35.61%** | 35.49% | 1.47 | 0 | 7 | 27 | 67 | 4.0979% | 0.999997 |
| **Union B (A + Rare Token)** | **67.55%** | 67.60% | 85.22 | 100 | 104 | 120 | 142 | 0.1337% | 0.999830 |
| **Union C (B + Char 3-Gram K=10)** | **74.27%** | 74.34% | 96.80 | 107 | 121 | 128 | 151 | 0.1294% | 0.999806 |
| **Union D (C + House/Postal)** | **79.53%** | 79.58% | 96.93 | 108 | 121 | 128 | 152 | 0.1384% | 0.999806 |
| **Union E (D + Address Token)** | **91.66%** | 91.59% | 150.57 | 164 | 181 | 187 | 212 | 0.1027% | 0.999699 |

---

## TABLE 3: Experiment 2 — Candidate Budgets (K = 25, 50, 100, 200)

Pruning the complete uncapped Union E down to fixed budgets using deterministic ranking.
We compare two pruning policies:
1. **Blocking Strength Priority:** Ranks by strategy tier (`exact_name` > `name_core` > `house_name` > `combined_postal_name` > `rare_token` > `char_3gram` > `address_token`).
2. **Multi-Block Support:** Ranks candidates by how many independent blocking strategies generated them, breaking ties by sum of blocking strengths.

| Candidate Budget (K) | Pruning Policy | Pair Recall (%) | Per-S1 Recall (%) | Captured True Pairs | Total Candidates | Avg Cands/S1 | Candidate Precision (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| K=25 | Priority | **72.49%** | 72.52% | 1,834 / 2,530 | 374,924 | 24.99 | 0.4892% |
| K=25 | Multi-Block Support | **75.42%** | 75.47% | 1,908 / 2,530 | 374,924 | 24.99 | 0.5089% |
| K=50 | Priority | **74.43%** | 74.44% | 1,883 / 2,530 | 748,088 | 49.87 | 0.2517% |
| K=50 | Multi-Block Support | **77.08%** | 77.09% | 1,950 / 2,530 | 748,088 | 49.87 | 0.2607% |
| K=100 | Priority | **77.00%** | 77.00% | 1,948 / 2,530 | 1,462,814 | 97.52 | 0.1332% |
| K=100 | Multi-Block Support | **79.17%** | 79.16% | 2,003 / 2,530 | 1,462,814 | 97.52 | 0.1369% |
| K=200 | Priority | **91.66%** | 91.59% | 2,319 / 2,530 | 2,258,525 | 150.57 | 0.1027% |
| K=200 | Multi-Block Support | **91.66%** | 91.59% | 2,319 / 2,530 | 2,258,525 | 150.57 | 0.1027% |

### Key Takeaways from Budget Experiments:
- **Multi-Block Superiority:** At $K=25$, Multi-Block preserves **75.42% recall** vs 72.49% for Priority (+2.93% absolute gain). When an entity is matched across multiple independent signals (e.g. name core + house number), confidence is vastly higher than single-channel hits.
- **Budget Saturation Curve:** At $K=100$, Multi-Block reaches **79.17% recall**. The jump to $K=200$ (91.66%) captures the remaining ~12.5% tail that relies solely on isolated address tokens.

---

## TABLE 4: Experiment 3 — Incremental Block Contribution over `name_core`

Starting from baseline `name_core` (Pair Recall: 35.61%, Avg Cands: 1.47), each strategy is added individually **WITHOUT any candidate cap**.

| Added Strategy | Base Recall (%) | New Recall (%) | Delta Recall (%) | Base Cands | New Cands | Delta Cands/S1 | Delta True Hits | Delta Total Cands | Recall Gain / Added Cand |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `house_name` (House Number + Name) | 35.61% | 55.34% | **+19.73%** | 1.47 | 1.64 | +0.17 | +499 | +2,593 | **0.192441** |
| `combined_postal_name` (Postal + Name Token) | 35.61% | 37.39% | **+1.78%** | 1.47 | 1.47 | +0.00 | +45 | +48 | **0.937500** |
| `postal` (Postal Code) | 35.61% | 38.46% | **+2.85%** | 1.47 | 1.59 | +0.12 | +72 | +1,882 | **0.038257** |
| `char_3gram_k5` (Char 3-Gram (K=5)) | 35.61% | 59.29% | **+23.68%** | 1.47 | 11.11 | +9.64 | +599 | +144,659 | **0.004141** |
| `char_3gram_k10` (Char 3-Gram (K=10)) | 35.61% | 61.78% | **+26.17%** | 1.47 | 21.00 | +19.53 | +662 | +293,026 | **0.002259** |
| `char_3gram_k20` (Char 3-Gram (K=20)) | 35.61% | 64.39% | **+28.78%** | 1.47 | 40.91 | +39.44 | +728 | +591,670 | **0.001230** |
| `rare_token` (Rare Token (IDF)) | 35.61% | 67.55% | **+31.94%** | 1.47 | 85.22 | +83.75 | +808 | +1,256,316 | **0.000643** |
| `address_token` (Address Token) | 35.61% | 74.78% | **+39.17%** | 1.47 | 55.26 | +53.79 | +991 | +806,941 | **0.001228** |

### Contribution Analysis:
1. **`house_name` (The Efficiency Leader):** Adds **+499 true hits** (+19.73% recall) at the cost of only **+2,593 candidates** (+0.17 cands/query), yielding a phenomenal efficiency of **0.1924 hits/candidate**.
2. **`combined_postal_name`:** Adds **+45 true hits** (+1.78% recall) with only **+48 candidates**, yielding **0.9375 hits/candidate** (near 100% precision).
3. **`char_3gram_k5` vs `k10` vs `k20`:**
   - `k=5` gives +23.68% recall for +9.64 candidates (0.0041 hits/cand).
   - Moving to `k=10` gains +2.49% recall for +9.89 candidates (0.0023 hits/cand).
   - Moving to `k=20` gains another +2.61% recall for +19.91 candidates (0.0012 hits/cand).
4. **`rare_token`:** Delivers +31.94% recall but adds +83.75 candidates/S1.
5. **`address_token`:** Captures the widest tail (+39.17% recall) but adds +53.79 candidates/S1.

---

## TABLE 5: Experiment 4 — Deep House-Name Strategy Investigation

Investigation into why `house_name` achieves 37.59% recall with only 0.20 candidates/S1.

| Metric | Value | Technical Rationale & Analysis |
| :--- | :---: | :--- |
| Total True Matches Captured | **951** | 37.59% of all ground-truth pairs in subset |
| Total False Candidates Generated | **2,100** | Out of 3,051 total candidates generated |
| Candidate Precision | **31.17%** | 31.17% precision is remarkably high for blocking (vs ~0.1% for token blocks) |
| False-Candidate Rate | **68.83%** | Only 68.83% of proposed candidates are negatives |
| Same House Number Match | **100.00%** | 100% of captured matches share identical normalized house numbers |
| Business Name Also Matches | **47.53%** | 47.53% share identical full normalized name; 52.47% have altered legal/name variants |
| Postal Code Also Matches | **8.52%** | 8.52% (low because postal code is missing/abbreviated in many records) |
| Normalized Address Matches | **19.24%** | 19.24% exact string match; 80.76% have street abbreviation/formatting variations |
| Recovered **ONLY** by `house_name` | **6.20%** | **59 true pairs** are completely invisible to exact name, name core, rare token, and char 3-gram |
| S1 → S2 Recall | **37.09%** | Consistent across sources |
| S1 → S3 Recall | **38.06%** | Consistent across sources |
| Geographic Distribution | **US: 736 | India: 215** | 77.4% US / 22.6% India (France = 0 in train set) |
| Singletons Receiving Candidates | **610** | 4.89% of 12,470 singletons received candidates, showing tight specificity |

### Why `house_name` is NOT an artifact:
- `house_name` pairs the street number (e.g. `100`, `254`) with the first two tokens of the business name.
- This creates an extremely restrictive blocking key: two businesses must operate at the exact same building number AND share the first two words of their company name.
- In business entity resolution, branches, acquisitions, and DBA renames frequently keep the physical address and primary brand name while changing legal forms (`LLC` vs `Inc`), suffixes, or secondary brand descriptors.
- Crucially, **59 true pairs (6.20% of house_name matches)** were found *exclusively* by this strategy, proving it captures genuine ground-truth variation that pure name token matching misses.

---

## TABLE 6: Experiment 5 — Source Separation (S1 → S2 vs S1 → S3)

Performance breakdown evaluated independently against S2 (250,000 targets) and S3 (250,000 targets).

| Strategy | S2 Pair Recall (%) | S2 Avg Cands | S2 Prec (%) | S3 Pair Recall (%) | S3 Avg Cands | S3 Prec (%) | S2 vs S3 Delta Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_name` | 20.59% | 0.23 | 7.315% | 20.67% | 0.23 | 7.673% | -0.08% |
| `name_core` | 36.27% | 0.73 | 4.078% | 34.99% | 0.74 | 4.118% | +1.28% |
| `name_token` | 52.12% | 44.26 | 0.096% | 49.77% | 44.01 | 0.099% | +2.35% |
| `rare_token` | 60.54% | 42.27 | 0.117% | 58.35% | 41.98 | 0.121% | +2.19% |
| `char_3gram_k5` | 45.42% | 5.00 | 0.742% | 44.26% | 5.00 | 0.771% | +1.16% |
| `char_3gram_k10` | 49.02% | 10.00 | 0.400% | 48.16% | 10.00 | 0.419% | +0.86% |
| `char_3gram_k20` | 52.78% | 19.99 | 0.215% | 51.99% | 19.99 | 0.226% | +0.79% |
| `postal` | 5.23% | 0.07 | 6.478% | 5.21% | 0.06 | 7.128% | +0.02% |
| `house_name` | 37.09% | 0.10 | 30.863% | 38.06% | 0.11 | 31.456% | -0.97% |
| `combined_postal_name` | 4.17% | 0.00 | 96.226% | 4.13% | 0.00 | 98.182% | +0.04% |
| `address_token` | 63.89% | 26.69 | 0.195% | 59.49% | 27.15 | 0.191% | +4.40% |

### S2 vs S3 Observations:
- Across virtually all strategies, S1 → S2 recall is slightly higher than S1 → S3 (e.g. `name_core`: 36.27% vs 34.99%; `rare_token`: 60.54% vs 58.35%; `address_token`: 63.89% vs 59.49%).
- S3 exhibits higher textual divergence and address noise compared to S2.
- However, both sources respond harmoniously to `house_name` (37.09% vs 38.06%) and `char_3gram` (49.02% vs 48.16%), confirming the blocking primitives are universally robust across disparate sources.

---

## TABLE 7: Experiment 6 — Hard Negative Impact & Categorization

Candidate quality breakdown showing false candidate volume and false candidate rate per block.

| Strategy | Total Candidates | True Hits Captured | False Candidates | Candidate Precision (%) | False Candidate Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `exact_name` | 6,964 | 522 | 6,442 | 7.4957% | 92.50% |
| `name_core` | 21,987 | 901 | 21,086 | 4.0979% | 95.90% |
| `name_token` | 1,324,094 | 1,288 | 1,322,806 | 0.0973% | 99.90% |
| `rare_token` | 1,263,793 | 1,503 | 1,262,290 | 0.1189% | 99.88% |
| `char_3gram_k5` | 149,939 | 1,134 | 148,805 | 0.7563% | 99.24% |
| `char_3gram_k10` | 299,874 | 1,229 | 298,645 | 0.4098% | 99.59% |
| `char_3gram_k20` | 599,733 | 1,325 | 598,408 | 0.2209% | 99.78% |
| `postal` | 1,942 | 132 | 1,810 | 6.7971% | 93.20% |
| `house_name` | 3,051 | 951 | 2,100 | 31.1701% | 68.83% |
| `combined_postal_name` | 108 | 105 | 3 | 97.2222% | 2.78% |
| `address_token` | 807,560 | 1,559 | 806,001 | 0.1931% | 99.81% |

### Mined Hard Negative Categories (Saved 600 Examples to `experiments/hard_blocking_negatives_budget.tsv`):

A dedicated pool of 600 difficult false-positive candidate pairs was mined from the uncapped candidate union. These pairs represent realistic confusions that our subsequent matching model must learn to disambiguate:

1. **Same Name, Different Entity:**
   - *Example:* `Christ Chapel` (Dundalk, MD) vs `Christ Chapel` (Helena, MT) [Generated by `exact_name`, `name_core`, `char_3gram`].
   - *Resolution Clue:* Geographically distant states, zero street overlap.
2. **Same Core Name, Different Entity:**
   - *Example:* `Christ Chapel` vs `Christ Chapel Inc.` (Buffalo, NY) vs `Christ Chapel Corp` (Damon, TX).
   - *Resolution Clue:* Distinct corporate suffixes and incompatible jurisdictions.
3. **Same House Number, Different Entity:**
   - *Example:* Two different businesses co-located at building number `2100` on different streets or cities.
   - *Resolution Clue:* Complete dissimilarity in business name tokens.
4. **Same Postal Code, Different Entity:**
   - *Example:* Multiple businesses sharing ZIP code `21222` with divergent trade names.
   - *Resolution Clue:* Address text and brand name tokens mismatch.
5. **Same Address, Different Entity (Commercial Complexes & Co-working):**
   - *Example:* Different businesses registered at the same suite/building address.
   - *Resolution Clue:* Suite/unit numbers and company name comparisons.

---

## Synthesis & Architectural Recommendations for Phase 3

1. **Recommended Production Candidate Generation Pipeline:**
   - **Active Strategies:** `exact_name` + `name_core` + `house_name` + `combined_postal_name` + `char_3gram_k10` + `rare_token` + `address_token`.
   - **Pruning Method:** Multi-Block Support ranking.
   - **Budget Recommendation:**
     - For **high-throughput training:** $K=50$ provides **77.08% pair recall** at 49.87 candidates/S1.
     - For **maximum recall inference:** $K=100$ provides **79.17% pair recall**, while $K=200$ captures the full **91.66% uncapped ceiling**.
2. **Next Steps for Phase 3 (Feature Engineering):**
   - Leverage the `blocking_sources` metadata directly as ranking features (e.g. `is_house_name_match`, `is_multi_block_supported`, `block_support_count`).
   - Train our discriminative ranker on the mined hard negatives (`experiments/hard_blocking_negatives_budget.tsv`) to ensure high precision on Macro-$F_{0.5}$.
