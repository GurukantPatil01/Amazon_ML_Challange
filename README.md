# Amazon ML Challenge 2026 — Business Entity Resolution

A modular, reproducible, competition-grade entity resolution system tailored for the Amazon ML Challenge 2026.

## 1. Project Purpose

In large-scale commercial platforms, business identity data arrives from multiple independent sources with noisy, incomplete, and inconsistent fields (e.g. legal entity abbreviations, landmark-based addresses, missing postal codes).

The challenge requires resolving entities across three distinct sources:
- **Source 1 (`S1-*`)**: Deduplicated reference source.
- **Source 2 (`S2-*`)**: Noisy secondary source.
- **Source 3 (`S3-*`)**: Noisy tertiary source.

The objective is to find all matching records from Source 2 and Source 3 for each Source 1 entity, while strictly adhering to:
1. **Precision-heavy Metric**: Evaluated via **Macro-$F_{0.5}$** across all Source 1 entities. False merges are penalized $2\times$ more heavily than missed links.
2. **Singletons Handling**: Source 1 entities with 0 true matches score $1.0$ if predicted empty, and $0.0$ if any match is mistakenly assigned.
3. **Blocking Efficiency**: Submissions require both `matching_results.tsv` (final matches) and `candidate_pairs.tsv` (candidate pool). Candidate generation must scale and minimize candidate set size.
4. **Unseen Country Generalization**: Training data contains India and US; test data introduces France as an unseen country. No country-specific hardcoding is permitted.
5. **No External Lookup**: Pure machine learning using only provided challenge data.

---

## 2. Architecture & Pipeline Stages

```
amazon-er/
├── dataset/
│   ├── train/                 # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                  # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── src/
│   ├── __init__.py            # Package root
│   ├── config.py              # Centralized paths, seeds, blocking & model parameters
│   ├── load_data.py           # Robust TSV loaders, schema validators & dataset profiling
│   ├── normalize.py           # Country-agnostic text and address normalization (Phase 1)
│   ├── blocking.py            # Scalable candidate pair generation / indexing (Phase 2)
│   ├── features.py            # Pairwise string, token, TF-IDF similarity features (Phase 3)
│   ├── train.py               # Pairwise matching classifier training (Phase 4)
│   ├── predict.py             # Inference pipeline & official TSV formatting (Phase 5)
│   ├── evaluate.py            # Official Macro-F0.5 metric & singleton scoring
│   ├── decision.py            # Threshold calibration & precision-optimal filtering
│   └── utils.py               # Logging, execution timers, and helpers
├── models/                    # Saved model artifacts
├── output/                    # matching_results.tsv & candidate_pairs.tsv
├── experiments/               # Experiment logs and ablation records
├── logs/                      # Execution and profiling logs
├── requirements.txt           # Minimal pinned dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Version control exclusions
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.12)

### Create Virtual Environment
```bash
cd amazon-er
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. How to Run Phase 0 (Data Ingestion & Profiling)

To execute data verification, schema checks, and complete profiling:

```bash
cd amazon-er
source .venv/bin/activate
python -m src.load_data
```

The script will:
1. Validate presence of all source files in `dataset/train/` and `dataset/test/`.
2. Strictly check expected column schemas (`entity_id`, `business_name`, `business_address`, `country`).
3. Compute row counts, unique entity IDs, duplicate counts, missing value statistics, country distributions, and string length profiles.
4. Output execution times and structured JSON profiling reports to console and `logs/data_loading.log`.

---

## 5. Implementation Status

| Phase | Milestone | Status | Description |
|---|---|---|---|
| **Phase 0** | **Data Loading & Profiling** | **Active** | Verified directory layout, centralized configuration, strict TSV parsing, dataset profiling, and official Macro-F0.5 evaluation metric. |
| Phase 1 | Text & Address Normalization | Planned | Standardize names/addresses across multi-lingual and multi-country formats without hardcoding. |
| Phase 2 | Blocking & Candidate Generation | Planned | Inverted indices, token/n-gram blocking to generate high-recall candidate pairs within budget. |
| Phase 3 | Feature Engineering | Planned | String distance, n-gram TF-IDF cosine, and token overlap features. |
| Phase 4 | Model Training | Planned | Classical GBDT (LightGBM) pairwise classifier with group-aware cross-validation. |
| Phase 5 | Calibration & Submission | Planned | F0.5-optimized thresholding, format validation, and generation of `matching_results.tsv` and `candidate_pairs.tsv`. |
