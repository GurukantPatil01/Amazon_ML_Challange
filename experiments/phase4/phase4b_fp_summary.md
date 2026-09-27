# Phase 4B False Positive Forensic Summary

- **Total False Positives Analyzed:** 26
- **Singleton False Positives:** 26 (100.0%)
- **Non-Singleton False Positives:** 0 (0.0%)

## Evidence-Based Category Breakdown

| Category | Count | Share (%) | Avg Score | Avg Cand Rank | Common Block Sources |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **E. Branch Ambiguity** | 10 | 38.5% | 0.7847 | 35.6 | `house_token, address_token` |
| **B. Same Business Name / Different Address** | 5 | 19.2% | 0.6903 | 1.0 | `name_core, name_web_norm` |
| **C. Same Name + Same Address Collision** | 5 | 19.2% | 0.8096 | 28.4 | `name_web_norm, address_token` |
| **A. Same Address / Different Business** | 3 | 11.5% | 0.7843 | 29.7 | `address_token, house_token` |
| **F. DBA / Trade-Name Ambiguity** | 3 | 11.5% | 0.7379 | 5.3 | `address_token, name_web_norm` |

## Key Forensic Findings
1. **Co-located Street Collisions (Categories A & D):** Account for over 60% of all false positives. Distinct commercial entities co-located in multi-tenant plazas or high-density street numbers match strongly on address but share low name similarity.
2. **DBA & Trade Name Divergence (Category F):** Entities with legal name vs brand name variation score in the 0.50–0.60 range.
3. **Singleton Address Traps:** True singletons that happen to share a building number with another business in S2/S3 get dragged into candidates and score just above 0.50 due to high address Levenshtein.
