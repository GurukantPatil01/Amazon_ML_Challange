# Business Entity Resolution — Amazon ML Challenge 2026

## Methodology and Technical Architecture

---

### 1. Problem Formulation
The objective of the Amazon ML Challenge 2026 Business Entity Resolution challenge is to link reference business entity records from **Source 1 (S1)** with corresponding entities in noisy heterogeneous partner directories: **Source 2 (S2)** and **Source 3 (S3)**.

Given:
- A query record $q \in S_1$
- Target entity repositories $S_2$ and $S_3$

The system must output a predicted set of matched IDs $\hat{M}(q) \subseteq S_2 \cup S_3$ where:
- A Source 1 entity may link to zero entities (true singletons), exactly one entity, or multiple entities across both targets.
- Self-matches ($S_1 \to S_1$) are prohibited.
- Performance is officially scored via per-entity **Macro-$F_{0.5}$**, placing heavy weight on precision while strictly evaluating empty singleton predictions ($F_{0.5} = 1.0$ if empty correctly, $0.0$ if false merge).

---

### 2. Dataset Characteristics & Noise Profiles
The dataset comprises over 11.7 million records across training and test splits:
- **Source 1:** Clean canonical reference businesses.
- **Source 2 & Source 3:** Crowdsourced and operational partner records characterized by severe real-world noise:
  1. *Optical Character Recognition (OCR) distortions & typos:* Digit-letter substitutions, omitted characters, concatenated words.
  2. *Web & Domain Formatting:* Names formatted with URL prefixes or domain suffixes (`www.`, `.com`, `.in`).
  3. *Cross-Lingual Script Divergence:* Indic scripts (Devanagari, Telugu, Kannada, Bengali, Gujarati, Tamil) co-existing with Latin transliterations.
  4. *Address Noise:* Missing postal codes, rearranged street tokens, shopping mall / commercial plaza co-locations.
  5. *Open-Set Geographies:* High-cardinality country coverage including open-set jurisdictions (e.g., France, UK, US, India).

---

### 3. Text & Entity Normalization
To bridge noise across disparate sources without loss of semantic identity, a deterministic normalization pipeline is applied:
- **Unicode Normalization:** NFKD decomposition followed by ASCII transliteration where applicable, preserving Indic code points for multilingual matching.
- **Case Folding & Punctuation:** Canonical whitespace folding, stripping non-alphanumeric punctuation while standardizing ampersands (`&` $\to$ `and`).
- **Domain & Web Sanitization:** Removal of leading `http://`, `https://`, `www.`, and trailing TLDs (`.com`, `.org`, `.net`, `.in`, `.co.uk`).
- **Legal Suffix Standardization:** Identification and stripping of corporate suffixes (`ltd`, `limited`, `pvt`, `inc`, `corp`, `llc`, `gmbh`, `sa`, `sarl`) to extract `name_core`.
- **Address & Structural Parsing:** Deterministic extraction of numeric building/house numbers (`house_number`) and standardized postal code tokens (`postal_code`).

---

### 4. High-Recall Candidate Generation (Union F)
Evaluating a naive Cartesian cross-product ($1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairs) is computationally impossible. We construct a multi-block inverted index candidate generator (**Union F**) that reduces candidate space by **$>99.98\%$** while maintaining **$>93.5\%$** candidate recall:
1. **Exact Name Block (`exact_name`):** Inverted index over fully normalized business names.
2. **Core Name Block (`name_core`):** Inverted index over legal-suffix-stripped business names.
3. **Rare Token Block (`rare_token`):** Tokens with collection frequency $\le 100$ per country partition.
4. **Character 3-Gram Block (`char_3gram_k10`):** MinHash/trigram indexing with size cap $K=10$ to prevent high-frequency fanout.
5. **Address Token Block (`address_token`):** Exact street/premise token match.
6. **House Number + Name Block (`house_name`):** Compound block matching exact building number plus core name.
7. **Postal + Name Block (`combined_postal_name`):** Compound block matching postal code plus first name token.
8. **Web-Normalized Name Block (`name_web_norm`):** Domain-stripped web entity match.
9. **House Token Block (`house_token`):** Inverted index matching house numbers within the same postal/country boundary.

Candidates from all nine blocks are unioned and ranked using **multi-block support count** (number of independent blocks agreeing on the candidate edge).

---

### 5. Multilingual Dense Embedding Retrieval
To recover the remaining blocking blind spots (specifically non-Latin Indic scripts and severe OCR corruptions), we deploy a compact sentence embedding model:
- **Model Identifier:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Parameter Count:** ~117.7M parameters ($\le$ 8B challenge limit)
- **Embedding Dimension:** 384 dense float32 dimensions
- **License:** Apache 2.0 (fully open source, local offline inference, zero external API usage)
- **Representations:**
  - *Name Embedding:* Dense representation of normalized business name.
  - *Name + Address Embedding:* Joint representation `name_norm | address_norm`.
- **ANN Search:** FAISS `IndexFlatIP` on unit-normalized vectors performing exact cosine inner-product retrieval.
- **Contribution:** Dense embeddings bridge script divergence, elevating cross-lingual cosine similarity from $0.07$ (classical Levenshtein) to $0.65 - 0.86$, successfully recovering over 54% of classical blocking misses at $K=100$.

---

### 6. Pairwise Feature Engineering
For every candidate pair $(S_1, \text{Target})$, a 46-to-54 dimensional feature vector is extracted:
1. **Name Similarity Signals:**
   - Levenshtein normalized similarity
   - Jaro-Winkler similarity
   - Token-level Jaccard & containment similarity
   - First-token and last-token Levenshtein similarity
   - Shared token count and minimum token containment ratio
2. **Address & Geo Signals:**
   - Address Levenshtein and token overlap
   - Numeric token containment (matching unit numbers, suite numbers)
   - Exact house number match flag
   - Exact postal code match flag
   - Country match flag (with robust open-set country handling)
3. **Compound & Interaction Signals:**
   - `house_and_name_interaction`: House match $\times$ name token overlap
   - `postal_and_name_interaction`: Postal match $\times$ name token overlap
4. **Block Provenance Features:**
   - Total classical block count
   - Independent block family count (Name, Address, House/Postal, Web)
   - Specific block membership flags (`has_house_token`, `has_web_norm`)
5. **Dense Embedding Features (when hybrid active):**
   - Name cosine similarity
   - Joint name+address cosine similarity
   - Normalized ANN rank $1.0 / (1.0 + \log(1 + \min(\text{rank}, 200)))$
   - ANN top-25 indicator flag
   - Interaction: Name cosine $\times$ name Levenshtein
   - Interaction: Name cosine $\times$ address Levenshtein
   - Interaction: Joint cosine $\times$ independent block count

---

### 7. Pairwise Scoring Model
- **Algorithm:** LightGBM Gradient Boosted Decision Trees (GBDT).
- **Objective:** Binary cross-entropy with early stopping on validation AUC/binary logloss.
- **Group-Aware Training Partition:** Stratified group split by Source 1 entity (12,000 train queries, 3,000 validation queries, seed 42) ensuring zero entity leakage between splits.
- **Hard Negative Mining:**
  - Classical hard negatives: Co-located distinct businesses sharing identical addresses/house numbers but different names; multi-branch chains sharing identical names but distinct cities.
  - Embedding hard negatives: Non-matching pairs exhibiting high cosine similarity ($\ge 0.70$) in ANN top-$K$ retrieval.

---

### 8. Decision Layer & Metric Optimization
- **Official Objective:** Per-entity Macro-$F_{0.5}$ metric:
  $$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- **Precision Bias:** The $\beta = 0.5$ parameter penalizes false merges twice as severely as false dismissals.
- **Optimal Decision Threshold:** Calibrated at $\tau = 0.50$ via systematic threshold sweeps $[0.20, 0.80]$.
- **Singleton Handling:** Queries with candidate scores below threshold $\tau$ output an empty prediction string, guaranteeing $F_{0.5} = 1.0$ on true singletons.
- **Multi-Match Support:** Queries with multiple candidate scores exceeding $\tau$ output comma-separated IDs, preserving one-to-many linkages.

---

### 9. Validation Benchmarks
Across the 3,000 unseen validation S1 queries (483 ground truth pairs, 2,550 singletons):
- **Candidate Recall:** **93.58%** (Union F) to **96.48%** (Union F + Embedding K=100)
- **Macro-$F_{0.5}$:** **96.65%**
- **Precision:** 96.67% | **Recall:** 96.62%
- **Confusion Matrix:** 401 True Positives, 23 False Positives, 82 False Negatives
- **Singleton Accuracy:** **99.18%**
- **Non-Singleton Macro-$F_{0.5}$:** **82.78%**

---

### 10. Scalability & Production Efficiency
- **Sub-Linear Processing:** Multi-block indexing limits candidate cardinality to ~155 candidates per query.
- **Vectorized Inference:** Fast C++ feature extraction via RapidFuzz and LightGBM native C API scores 10,000 candidates in $<0.2$ seconds.
- **Memory Footprint:** Peak RAM remains strictly bounded ($<1.8\text{ GB}$), avoiding Cartesian memory blowups.
- **Full Test Set Readiness:** Full test set processing operates in deterministic sequential chunks with constant memory utilization.

---

### 11. Challenge Compliance Certification
- **No External Entity Data:** Zero external business directories, Wikipedia, Google Places, or proprietary databases used.
- **No Commercial APIs:** Zero calls to external geocoding, translation, or matching APIs.
- **Offline Self-Contained Inference:** All models run locally on consumer hardware within the $\le$ 8B parameter threshold.
- **Zero Label Leakage:** Strict separation between candidate generation, feature engineering, and ground-truth evaluation.
