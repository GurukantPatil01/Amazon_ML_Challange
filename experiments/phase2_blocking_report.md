# Phase 2 Experiment Report: High-Recall, Low-Cardinality Candidate Generation

**Date:** 2026-09-27  
**Status:** PASS  

---

## 1. Executive Summary & Objective

In large-scale entity resolution, pairwise Cartesian comparison is strictly prohibited:
- 2.2M Source 1 records $\times$ 10.3M Source 2/3 records = **~22.7 Trillion pairs**.

The objective of Phase 2 is to design, implement, and benchmark a **multi-strategy blocking engine** that cuts the search space to a tiny, high-quality candidate set per Source 1 entity while preserving maximal true positive recall.

### Key Results Summary
- **Evaluated**: 15,000 real S1 queries against an indexed pool of 250,000 target S2 records using ground truth.
- **Exact Name vs Name Core**: Stripping legal entity suffixes (`name_core`) increases recall from **20.59% to 36.27%** (+15.68% absolute gain) while adding only **0.50 candidates/query**.
- **Rare Token Blocking**: Utilizing inverse document frequency (IDF) token selection boosts recall to **60.54%**.
- **House Number + Name Token**: Delivers **37.09% recall** with a microscopic **0.10 candidates/query** average (reduction ratio: 0.999999).
- **Multi-Strategy Candidate Union (Union D)**: Achieves **74.26% recall** at an average of **43.95 candidates per query** (P95: 50.0, reduction ratio: **0.999824**), eliminating **99.982%** of unviable pairs.

---

## 2. Token Frequency Analysis & Selectivity Controls

Empirical token frequency distribution measured across the training corpus revealed severe long-tail skew:

| Token Type | Top Frequent Tokens | Max Document Frequency | Danger / Risk | Frequency Control |
|---|---|---|---|---|
| **Name Legal Suffixes** | `limited` (23.9%), `private` (19.8%), `llc` (16.0%), `inc` (10.9%) | 24,000 per 100k records | Massive candidate explosion if indexed as naive tokens | Filtered via configurable stop-suffix dictionary & `max_df_ratio=0.005` |
| **Name Stopwords** | `and` (7.6%), `ltd` (6.6%), `pvt` (5.3%), `of` (2.0%) | 7,600 per 100k records | False merges across unrelated businesses | Filtered by `min_token_length=3` & frequency pruning |
| **Address Thoroughfares** | `rd` (22.1%), `no` (20.3%), `st` (14.7%), `dr` (10.4%), `ave` (8.8%) | 22,000 per 100k records | Would link all businesses on any 'Road' or 'Street' | Address token index restricted to `max_df_ratio=0.002` and composite blocks |
| **City / State Names** | `delhi` (15.0%), `maharashtra` (8.7%), `nagar` (7.5%), `tx` (6.0%) | 15,000 per 100k records | Regional over-clustering | Never used as standalone blocking keys |

---

## 3. Individual Blocking Strategy Ablation Table

Empirical evaluation on real training dataset (15,000 S1 queries vs 250,000 S2 pool):

| Strategy ID & Name | Candidate Pair Recall | Mean S1 Recall | Avg Cands / S1 | Median | P95 | P99 | Max Cands | Reduction Ratio | Candidate Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **exact_name** | 20.59% | 20.59% | 0.23 | 0.0 | 1.0 | 5.0 | 15 | 0.999999 | 0.073149 |
| **name_core** | 36.27% | 36.16% | 0.73 | 0.0 | 4.0 | 14.0 | 38 | 0.999997 | 0.040779 |
| **name_token** | 52.12% | 52.15% | 44.26 | 50.0 | 50.0 | 50.0 | 50 | 0.999823 | 0.000961 |
| **rare_token** | 60.54% | 60.68% | 42.27 | 50.0 | 50.0 | 50.0 | 50 | 0.999831 | 0.001169 |
| **char_3gram_k5** | 45.83% | 45.74% | 5.00 | 5.0 | 5.0 | 5.0 | 5 | 0.999980 | 0.007483 |
| **char_3gram_k10** | 49.75% | 49.62% | 10.00 | 10.0 | 10.0 | 10.0 | 10 | 0.999960 | 0.004062 |
| **char_3gram_k20** | 53.43% | 53.25% | 19.99 | 20.0 | 20.0 | 20.0 | 20 | 0.999920 | 0.002181 |
| **postal** | 5.23% | 5.06% | 0.07 | 0.0 | 0.0 | 2.0 | 8 | 1.000000 | 0.064777 |
| **house_name** | 37.09% | 37.09% | 0.10 | 0.0 | 1.0 | 2.0 | 13 | 1.000000 | 0.308634 |
| **address_token** | 63.89% | 63.80% | 26.69 | 30.0 | 30.0 | 30.0 | 30 | 0.999893 | 0.001953 |
| **combined_postal_name** | 4.17% | 4.01% | 0.00 | 0.0 | 0.0 | 0.0 | 2 | 1.000000 | 0.962264 |

---

## 4. Multi-Strategy Candidate Unions

We evaluated four progressive candidate union configurations (deduplicated and capped at 50 candidates per S1):
- **Union A**: `exact_name` + `name_core` (High precision, minimal candidates)
- **Union B**: Union A + `rare_token` (Captures legal suffix variations and rare brand names)
- **Union C**: Union B + `char_3gram_k10` (Recovers spelling typos, punctuation, and transliterations)
- **Union D**: Union C + `house_name` + `combined_postal_name` (Recovers entities with noisy names sharing physical address premises)

| Candidate Union Configuration | Pair Recall | Mean S1 Recall | Avg Cands / S1 | Median | P95 | P99 | Reduction Ratio | Candidate Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Union A (Exact + Core)** | 36.27% | 36.16% | 0.73 | 0.0 | 4.0 | 14.0 | 0.999997 | 0.040779 |
| **Union B (A + Rare Token)** | 66.99% | 67.05% | 42.30 | 50.0 | 50.0 | 50.0 | 0.999831 | 0.001292 |
| **Union C (B + Char 3-Gram K=10)** | 69.44% | 69.54% | 43.94 | 50.0 | 50.0 | 50.0 | 0.999824 | 0.001290 |
| **Union D (C + House/Postal)** | 74.26% | 74.26% | 43.95 | 50.0 | 50.0 | 50.0 | 0.999824 | 0.001379 |

---

## 5. Hard-Negative Analysis

500 real hard negatives were identified and saved to [`experiments/hard_blocking_negatives.tsv`](file:///Users/gurukantpatil/Desktop/Hackathon/Amazon_ML/amazon-er/experiments/hard_blocking_negatives.tsv).

### Key Hard Negative Patterns Discovered:
1. **Identical Name / Different Address Entities**:
   - S1: `Christ Chapel` at `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD`
   - Candidate: `CHRIST CHAPEL INC.` at `73 LASALLE AVE, BUFFALO, NY` (Ground Truth: NOT A MATCH)
2. **National Brand Franchises / Distinct Geographies**:
   - S1: `Helios` at `66 Edgewood Street, Bridgeport, CT`
   - Candidate: `Helios Inc` at `#695 PARK LN, MONROE, WA` (Ground Truth: NOT A MATCH)
3. **Insight for Phase 3 & 4 (Features & Model)**:
   - Exact or high name similarity alone is insufficient. The matching model must heavily weight address numerical concordances (`house_number`, postal code, city/state tokens) to avoid false merges, which are penalized 2x under Macro-$F_{0.5}$.

---

## 6. Full Data Scaling Projections vs Subset Results

| Metric | Measured Subset (15k S1 vs 250k S2) | Projected Full Training Set (2.2M S1 vs 10.3M S2/S3) | Implementation Safeguard |
|---|---|---|---|
| **Target Index Build Time** | 6.04 seconds (for 250k S2) | ~2.5 - 3.5 minutes (for 10.3M S2/S3) | Built once in parallel for S2 and S3 using zipped string arrays |
| **Query Throughput** | ~5,000 - 8,000 queries/sec | ~4.5 - 6.5 minutes for full 2.2M S1 queries | Chunked generator streaming |
| **Total Candidate Volume** | ~659,000 candidate pairs | ~95 - 110M deduplicated candidate pairs | Streamed directly to disk/cache |
| **Memory Footprint** | ~180 MB | ~1.8 - 2.4 GB peak RAM | Integer postings indices, no Cartesian matrix |

---

## 7. Reusable API Specification

The module [`src/blocking.py`](file:///Users/gurukantpatil/Desktop/Hackathon/Amazon_ML/amazon-er/src/blocking.py) exposes the competition-grade API:

```python
from src.blocking import generate_candidates

candidates_df = generate_candidates(
    source1=s1_df,
    source2=s2_df,
    source3=s3_df,
    strategies=['exact_name', 'name_core', 'rare_token', 'char_3gram_k10', 'house_name'],
    max_candidates_per_s1=50,
)
```

Columns returned: `['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocking_sources']`

---

## 8. Unit Tests & Quality Verification

All **25 unit tests** in `tests/test_blocking.py` and `tests/test_normalize.py` passed in **0.30 seconds**.