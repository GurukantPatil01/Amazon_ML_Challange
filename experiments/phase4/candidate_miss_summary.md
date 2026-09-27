# Phase 4 Candidate Miss Diagnostic Summary

- **Total Ground-Truth Positive Pairs in Validation Split:** 483
- **Total Pairs Missed at Candidate Generation:** **53 pairs** (10.97% of ground truth)
- **Target Source Breakdown:** Source 2 = 27 | Source 3 = 26
- **Geographic Distribution:** {'india': 29, 'us': 24}

---

## 1. Primary Name-Signal Characteristics of Missed Pairs

| Name Characteristic | Missed Pair Count | Percentage of Misses (%) | Opportunity for Recovery |
| :--- | :---: | :---: | :--- |
| High Name Similarity (Levenshtein >= 0.70) | **21** | 39.6% | Recoverable via higher char-ngram depth or adaptive char retrieval |
| Moderate Similarity (0.40 <= Levenshtein < 0.70) | **12** | 22.6% | Word-prefix, initials, or token overlap combinations |
| Low Similarity (Levenshtein < 0.40) | **20** | 37.7% | Extreme brand renames; requires address-assisted or house-name variants |
| Same First Token | **13** | 24.5% | Word prefix / first token signature |
| Same Initials / Acronym | **9** | 17.0% | Company acronym / initial signature |

---

## 2. Address-Signal Characteristics of Missed Pairs

| Address Characteristic | Missed Pair Count | Percentage of Misses (%) | Opportunity for Recovery |
| :--- | :---: | :---: | :--- |
| Same House/Street Number | **22** | 41.5% | House number + single word / initial block |
| High Address Token Overlap ($\ge 0.50$) | **46** | 86.8% | Co-located street / address-assisted tokens |
| Missing Address in Query/Target | **3** | 5.7% | Pure name matching required (cannot use address blocks) |
| Postal Code Match | **2** | 3.8% | Postal prefix combinations |

---

## 3. Representative Candidate Miss Examples

| S1 Query Name | Target Match Name | S1 Address | Target Address | Levenshtein Sim | Primary Failure Reason |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `Pediatric Dentistry Pioneer Ca` | `Pediatric Dentistry Pioneer Ca` | `Alton, 20 Marian Heights Drive` | `##20 Marian Heights Drive, # 7` | 0.74 | High Name Similarity but token variation / legal truncation |
| `EY Tradevision Pvt Ltd` | `EY Trádevision Pvt Ltd` | `16, Shri Chaitanaya State Bank` | `##16, Shri Chaitanaya State Ba` | 0.95 | High Name Similarity but token variation / legal truncation |
| `Premier Constructions Private ` | `প্রিমিয়ার কনস্ট্রাকশনস প্রাইভ` | `10, Tottie Lane Kolkata, Kolka` | `10, Kolkata, Howrah, পশ্চিমবঙ্` | 0.08 | Extreme Name Dissimilarity / Alias / DBA | Same Building, Divergent Name Prefix |
| `Fortune Precision High Inc` | `Fortuneprecisionhigh.Com` | `1659 N Temple Street, Unit B21` | `1659 1/2 N Temple St, Unit B21` | 0.81 | High Name Similarity but token variation / legal truncation | Same Building, Divergent Name Prefix |
| `Paone Industries` | `Pindustries.Com` | `809 Pioneer Road, Mesquite, TX` | `809 Pioneer Road, Mesquite, Te` | 0.44 | Same Building, Divergent Name Prefix |
| `Great Galaxy Healthcare Privat` | `ಗ್ರೇಟ್ ಗ್ಯಾಲಕ್ಸಿ ಹೆಲ್ತ್‌ಕೇರ್ ಪ` | `No 1091, 12Th Main, 2Nd Stage ` | `DOOR NO 1091, BANGALORE, Karna` | 0.10 | Extreme Name Dissimilarity / Alias / DBA | Same Building, Divergent Name Prefix |
| `Unique Services Private Limite` | `ইউনিক সার্ভিসেস প্রাইভেট লিমিট` | `47, Matheshwartala Road, Kolka` | `47, CALCUTTA, HOWRAH, West Ben` | 0.06 | Extreme Name Dissimilarity / Alias / DBA | Same Building, Divergent Name Prefix |
| `Valley Coalition` | `The Valley Coalition` | `9628 Laurel Lane, Fl 1, Scotts` | `9628. Laurel Lane, PMB 1695, S` | 0.80 | High Name Similarity but token variation / legal truncation | Same Building, Divergent Name Prefix |
| `Bay Charities LP` | `Fayekor t/a Bay Charities LP` | `362 Scott Street, Hubbard, OH` | `362 Scott Street, Hubbard, OH` | 0.57 | Same Building, Divergent Name Prefix |
| `Mills & Le LLC` | `millsle.com` | `343 Lincoln Avenue, Lebanon, K` | `343-347 LINCOLN AVENUE, LBEANO` | 0.50 | General Tail / Multi-Token Divergence |
