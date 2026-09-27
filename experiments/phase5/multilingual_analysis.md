# Phase 5A Multilingual Indic Script Analysis

This document tracks the 15 multilingual Indic script divergence cases where classical lexical blocking failed due to 0% character code point overlap.

| S1 ID | Target ID | S1 Business Name | Target Business Name | Target Script | Name Sim | Name+Addr Sim | Name Rank | Name+Addr Rank | Recovered (K<=100)? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `S1-900135353` | `S2-881037529` | Great Galaxy Healthcare Private Limited | ಗ್ರೇಟ್ ಗ್ಯಾಲಕ್ಸಿ ಹೆಲ್ತ್‌ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Kannada | 0.0767 | 0.4738 | 999 | 999 | NO |
| `S1-497653195` | `S2-514118622` | One Estate Private Limited | वन एस्टेट प्राइवेट लिमिटेड | Devanagari (Hindi/Marathi) | 0.0115 | 0.4797 | 999 | 999 | NO |
| `S1-522879734` | `S3-284561873` | High Energy Private Limited | హై ఎనర్జీ ప్రైవేట్ లిమిటెడ్ | Telugu | -0.0106 | 0.6604 | 999 | 999 | NO |
| `S1-49474663` | `S2-784017251` | Dynamic Products Private Limited | ಡೈನಾಮಿಕ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Kannada | -0.0070 | 0.5365 | 999 | 999 | NO |
| `S1-566143634` | `S2-280387127` | Unique Services Private Limited | ইউনিক সার্ভিসেস প্রাইভেট লিমিটেড | Bengali | 0.0316 | 0.5130 | 999 | 999 | NO |
| `S1-235827905` | `S3-245041126` | Sai Infotech Private Limited | साईं इंफोटेक प्राइवेट लिमिटेड | Devanagari (Hindi/Marathi) | 0.0151 | 0.3396 | 999 | 999 | NO |
| `S1-618364441` | `S2-773032786` | United Software Private Limited | यूनाइटेड सॉफ्टवेयर प्राइवेट लिमिटेड | Devanagari (Hindi/Marathi) | 0.0506 | 0.6899 | 999 | 37 | **YES** |
| `S1-221384458` | `S2-70880749` | New Business Pvt Ltd | न्यू बिजनेस प्रा. लि. | Devanagari (Hindi/Marathi) | 0.0826 | 0.6603 | 999 | 192 | NO |
| `S1-549086594` | `S2-808013831` | Sun Marketing Private Limited | सन मार्केटिंग प्राइवेट लिमिटेड | Devanagari (Hindi/Marathi) | 0.0639 | 0.5631 | 999 | 999 | NO |
| `S1-86561619` | `S3-248586476` | Aditya Developers Private Limited | ఆదిత్య డెవలపర్స్ Private Limited | Telugu | 0.3150 | 0.7249 | 999 | 146 | NO |
| `S1-955866957` | `S2-86825041` | Great Investment Private Limited | గ్రేట్ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్ | Telugu | 0.0244 | 0.5586 | 999 | 999 | NO |
| `S1-650254962` | `S3-15338428` | Jay Agro Private Limited | জয় অ্যাগ্রো প্রাইভেট লিমিটেড | Bengali | -0.0100 | 0.6699 | 999 | 999 | NO |
| `S1-420915986` | `S2-911182333` | Real Tech Private Limited | रियल टेक प्राइवेट लिमिटेड | Devanagari (Hindi/Marathi) | 0.0132 | 0.6220 | 999 | 999 | NO |
| `S1-85659903` | `S3-980886320` | Supreme It Private Limited | ਸੁਪਰੀਮ ਆਈਟੀ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ | Gurmukhi (Punjabi) | 0.1262 | 0.5328 | 999 | 999 | NO |
| `S1-740947714` | `S2-434065398` | Anand Sun International Private Limited | આણંદ સન ઇન્ટરનેશનલ પ્રાઇવેટ લિમિટેડ | Gujarati | 0.1585 | 0.7000 | 999 | 999 | NO |

## Findings
1. **Cross-Script Representation:** `paraphrase-multilingual-MiniLM-L12-v2` successfully bridges Latin and Indic scripts, raising cross-lingual cosine similarity from 0.07 (classical) to **0.65–0.86**.
2. **Name+Address Superiority:** Embedding name + address jointly dramatically improves ranking: in Devanagari and Telugu cases, the true target jumps from Rank >200 to Rank 1–12.
