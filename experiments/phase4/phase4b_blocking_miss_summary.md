# Phase 4B Remaining Candidate Misses Analysis (31 Pairs)

Union F successfully reduced candidate misses from 53 down to **31** pairs.

## Forensic Categorization of the 31 Remaining Misses

1. **Multilingual Indic Script Divergence (15 pairs / 48.4%):**
   - S1 record is written in English Latin script, while the matching S2/S3 record is written in an Indic script (Devanagari, Bengali, Telugu, Kannada, Gujarati, or Punjabi).
   - Examples: `Dynamic Products Private Limited` vs `ಡೈನಾಮಿಕ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`, `Real Tech Private Limited` vs `रियल टेक प्राइवेट लिमिटेड`.
   - Classical character and token blocking fail because Unicode code points share 0% overlap without cross-lingual transliteration or semantic embeddings.

2. **Missing Target Addresses (3 pairs / 9.7%):**
   - S2/S3 target record has a completely missing address (`business_address == ''`).
   - Examples: `Maid Book Store`, `Jones Rocky Enterprise Inc`, `Pediatric Dental Atlantic Center Inc`.
   - Because the target has no address, all address-based and postal-based blocks yield 0 matches. Character mutations in the name prevent exact name blocking.

3. **Severe Optical / OCR Character Mutations (7 pairs / 22.6%):**
   - Multi-character OCR digit-for-letter substitutions and severe typos: `Standard Education Concfultaas`, `D0ng Gnlo of Incorporated`, `Posasr [Inc]`.
   - Displaced by candidate budget or frequency caps.

4. **Complete DBA / Trade Aliases (6 pairs / 19.4%):**
   - Complete brand alias without common tokens: `Hurtado Broadband` vs `Miraquo`, `WN Associated Private Limited` vs `K0rsynkor`.

## Can Classical Blocking Recover These 31 Misses?
- **No.** Further expanding classical character or token blocking to catch these 31 misses causes severe candidate explosion (+500 cands/S1) and uncontrolled false-positive inflation.
- These remaining misses are textbook **semantic, cross-script, and alias matching problems** that require multilingual embeddings or phonetic transliteration.
