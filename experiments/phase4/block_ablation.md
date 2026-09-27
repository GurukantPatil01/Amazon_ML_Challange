# Phase 4A Evidence-Backed Blocking Ablation Report

## 1. Locked Phase 3 Baseline Reference
- **Validation S1 Entities:** 3,000
- **Ground-Truth Positive Pairs:** 483
- **Candidate Recall (Union E K=200):** **88.82%** (429 / 483 true pairs captured)
- **Candidate Misses:** **53 pairs**
- **Baseline LightGBM Macro-F0.5:** **95.32%** (Threshold $\tau = 0.45$)
- **Baseline Model Recall among Candidates:** 80.93% (348 / 430)
- **Baseline End-to-End Recall:** 72.05% (348 / 483)
- **Baseline FN:** 135 | **Baseline FP:** 16
- **Singleton Exact Empty Accuracy:** 99.37%

---

## 2. Individual Strategy Ablations

| Strategy | True Pairs Captured | Candidate Recall (%) | Remaining Misses | Avg Cands/S1 | P95 | P99 | Max | Candidate Precision (%) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Phase 3 Baseline (Union E)` | 429 / 483 | **88.82%** | 54 | 150.88 | 180 | 185 | 199 | 0.0948% | 0.00s |
| `name_web_norm` | 214 / 483 | **44.31%** | 269 | 2.48 | 12 | 39 | 89 | 2.8787% | 0.11s |
| `char_3gram_k20` | 247 / 483 | **51.14%** | 236 | 40.00 | 40 | 40 | 40 | 0.2058% | 9.74s |
| `char_3gram_k50` | 266 / 483 | **55.07%** | 217 | 99.99 | 100 | 100 | 100 | 0.0887% | 12.56s |
| `char_3gram_adaptive` | 259 / 483 | **53.62%** | 224 | 69.70 | 70 | 70 | 70 | 0.1239% | 9.28s |
| `house_token` | 246 / 483 | **50.93%** | 237 | 4.42 | 23 | 81 | 100 | 1.8548% | 0.25s |
| `name_core_v2` | 174 / 483 | **36.02%** | 309 | 1.42 | 7 | 26 | 68 | 4.0788% | 0.15s |

---

## 3. Combination Ablations (Unions A through H)

| Combination | Captured Pairs | Candidate Recall (%) | Delta Recall (%) | Incremental True Pairs | Avg Cands/S1 | P95 | P99 | Candidate Precision (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Union A (Phase3 + WebNorm)** | 439 / 483 | **90.89%** | **+2.07%** | **+10** | 151.66 | 182 | 200 | 0.0965% |
| **Union B (Phase3 + AdaptiveChar)** | 431 / 483 | **89.23%** | **+0.41%** | **+2** | 182.85 | 230 | 233 | 0.0786% |
| **Union C (Phase3 + HouseToken)** | 448 / 483 | **92.75%** | **+3.93%** | **+19** | 155.02 | 190 | 228 | 0.0963% |
| **Union D (Phase3 + NameCoreV2)** | 431 / 483 | **89.23%** | **+0.41%** | **+2** | 150.90 | 180 | 186 | 0.0952% |
| **Union E (Phase3 + WebNorm + AdaptiveChar)** | 440 / 483 | **91.10%** | **+2.28%** | **+11** | 183.62 | 230 | 246 | 0.0799% |
| **Union F (Phase3 + WebNorm + HouseToken)** | 452 / 483 | **93.58%** | **+4.76%** | **+23** | 155.80 | 195 | 235 | 0.0967% |
| **Union G (Phase3 + AdaptiveChar + HouseToken)** | 449 / 483 | **92.96%** | **+4.14%** | **+20** | 186.98 | 233 | 266 | 0.0800% |
| **Union H (Phase3 + All Four Strategies)** | 452 / 483 | **93.58%** | **+4.76%** | **+23** | 187.76 | 236 | 270 | 0.0802% |

---

## 4. Candidate Budget Ablation for Top Combination (Union H)