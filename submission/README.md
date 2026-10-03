# Amazon ML Challenge 2026 — Business Entity Resolution

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-green.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Model-LightGBM%20%2B%20Union%20F-orange.svg)](https://lightgbm.readthedocs.io/)

A scalable, competition-grade Business Entity Resolution pipeline engineered for the Amazon ML Challenge 2026. The system links canonical reference businesses in Source 1 against noisy, heterogeneous partner directories in Source 2 and Source 3.

---

## 1. System Overview & Problem Statement

Commercial entity resolution requires matching reference businesses against noisy records with optical character recognition (OCR) distortions, domain and web formatting, legal suffix variations, co-located addresses, and open-set international jurisdictions (e.g. France, India, United States).

### Challenge Objective
For every reference query $q \in \text{Source 1}$:
- Predict all matching records in $\text{Source 2}$ and $\text{Source 3}$.
- Source 1 entities may have zero matches (singletons), exactly one match, or multiple matches across both secondary sources.
- Self-matches ($\text{S1} \to \text{S1}$) are strictly prohibited.
- Evaluated via per-entity **Macro-$F_{0.5}$**, placing $2\times$ greater emphasis on precision while strictly penalizing false merges on singletons ($F_{0.5} = 1.0$ if correctly empty, $0.0$ if any false merge).

---

## 2. Technical Architecture

The production architecture consists of four modular layers:

```
Source 1 (Query)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Text & Address Normalization                        │
│    - NFKC Unicode canonical decomposition              │
│    - Legal entity suffix stripping (extracts core)     │
│    - Postal code & building/house token parsing        │
│    - URL / domain normalization (removes .com, .in)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. High-Recall Candidate Blocking (Union F)            │
│    - Exact Name & Name Core Blocks                     │
│    - Rare Token Inverted Index                         │
│    - Character 3-Gram MinHash (K=10 cap)               │
│    - House Number + Name Token Block                   │
│    - Combined Postal + Name Block                      │
│    - Selective Address Token Block                     │
│    - Web-Normalized Collapsed Block                    │
│    - House Token Compound Block                        │
│    - Multi-Block Support Ranking & Candidate Pruning   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Pairwise Feature Engineering (43 Features)          │
│    - String Metrics: Levenshtein, Jaro-Winkler         │
│    - Token Metrics: Jaccard, Containment, 3-gram       │
│    - Address Metrics: House match, Postal match        │
│    - Interactions: Name x Address similarities         │
│    - Provenance: Support count, block strength sum     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Decision & Matcher Layer                            │
│    - LightGBM Gradient Boosted Decision Trees          │
│    - Precision-Calibrated Threshold: tau = 0.50        │
│    - Deterministic Output Formatting & Validation      │
└────────────────────────────────────────────────────────┘
```

---

## 3. Directory Layout

```text
.
├── output/
│   ├── matching_results.tsv       # Final predicted links (scored on leaderboard)
│   └── candidate_pairs.tsv        # Blocking candidate pool (audited for efficiency)
│
├── dataset/
│   ├── train/                     # Training sets and ground truth
│   └── test/                      # test_source1.tsv, test_source2.tsv, test_source3.tsv
│
├── models/
│   └── lightgbm_baseline.txt      # Trained LightGBM booster artifact
│
├── src/
│   ├── normalize.py               # Text, legal suffix, and address normalization
│   ├── block_index.py             # High-performance InvertedIndex data structure
│   ├── blocking_phase4.py         # Union F candidate generation engine
│   ├── pair_features.py           # Vectorized batch pairwise feature extractor
│   ├── train_lightgbm.py          # Group-aware training and threshold tuning
│   ├── decision.py                # F0.5 decision optimization and evaluation
│   ├── run_test_inference.py      # Memory-efficient streaming test inference engine
│   ├── final_submission_audit.py  # Automated integrity and consistency validator
│   └── utils.py                   # System logging and profiling helpers
│
├── methodology/
│   └── Methodology.md             # Comprehensive technical paper
│
├── reports/
│   ├── FINAL_MODEL_SELECTION.md   # Benchmark and model selection justification
│   ├── FINAL_CANDIDATE_AUDIT.md   # Candidate distribution and reduction statistics
│   └── FINAL_SUBMISSION_REPORT.md # Verification contract and execution summary
│
├── utils/
│   └── validate_submission.py     # Official competition submission validator
│
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # Pipeline guide and instructions
└── LICENSE                        # Apache 2.0 open-source license
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- 8 GB RAM minimum

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/GurukantPatil01/Amazon_ML_Challange.git
cd Amazon_ML_Challange

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Training the Model

To train the LightGBM pairwise ranking model on labeled training pairs:

```bash
PYTHONPATH=. python3 src/train_lightgbm.py
```

The script will:
1. Load training records and ground truth labels.
2. Build Union F inverted indices and extract labeled training pairs.
3. Train the LightGBM classifier with early stopping on validation AUC.
4. Tune the decision threshold $\tau$ to maximize Macro-$F_{0.5}$.
5. Persist the trained model artifact to `models/lightgbm_baseline.txt`.

---

## 6. Running Test Inference

To run the complete production inference pipeline across the entire test set (1.73M queries):

```bash
PYTHONPATH=. python3 src/run_test_inference.py
```

### Execution Features
- **Country Partitioning:** Processes France, India, and the United States independently, keeping peak RAM below 2.5 GB.
- **Single-Pass Streaming:** Queries each record once across all inverted index blocks, streaming results directly to disk.
- **Automatic Resume:** Detects partially written outputs and resumes from the exact query boundary.
- **Output Artifacts:** Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

---

## 7. Official Validation & Consistency Verification

Before submitting, run both the official challenge validator and the internal audit:

```bash
# 1. Run official competition submission validator
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 2. Run internal comprehensive integrity audit
python3 src/final_submission_audit.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Verification Criteria
- `matching_results.tsv`: 1,732,544 rows, tab-separated, zero S1 self-matches, valid S2/S3 IDs only.
- `candidate_pairs.tsv`: 1,732,544 rows, matching schema `source1_entity_id\tcandidate_entity_ids`.
- **Strict Predictive Containment:** $100\%$ of predicted matches are contained within the candidate set ($\text{matching} \subseteq \text{candidates}$).

---

## 8. Final Deliverables

The final submission package is assembled in the `submission/` directory:
- `submission/output/`: Contains `matching_results.tsv` and `candidate_pairs.tsv`.
- `submission/code/`: Complete, clean source code for reproduction.
- `submission/methodology/`: Complete technical architecture document (`Methodology.md`).
- `submission/README.md`: Reproduction documentation.
- `submission/LICENSE`: Open source Apache 2.0 license.

The official competition archive is packaged as `Amazon_ML_Challenge_2026_Final_Submission.zip`.

---

## 9. Compliance

- **No External Business Data:** Strictly adheres to competition rules; no external commercial APIs, geocoding lookups, or external company registries are used.
- **Model Size:** Employs LightGBM and local open-source models ($\le 8\text{B}$ parameters).
- **License:** Apache 2.0 compatible open-source license.
