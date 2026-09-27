"""Generates the comprehensive Phase 2 Candidate Budget and Uncapped Union Markdown Report.

Reads data from experiments/phase2_budget_data.json and formats Tables 1 through 7
with rigorous technical analysis and methodology documentation.
"""

import json
from pathlib import Path

DATA_PATH = Path("experiments/phase2_budget_data.json")
REPORT_PATH = Path("experiments/phase2_budget_report.md")


def format_report():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    t1 = data["table1_individual"]
    t2 = data["table2_uncapped_unions"]
    t3 = data["table3_budgets"]
    t4 = data["table4_incremental"]
    t5 = data["table5_house_investigation"]
    t6 = data["table6_sources"]
    t7 = data["table7_neg_stats"]

    md = []

    md.append("# Phase 2 Candidate Budget & Uncapped Union Evaluation Report")
    md.append("")
    md.append("> **IMPORTANT SCOPE NOTICE: SUBSET EXPERIMENT**")
    md.append("> - **Queries:** 15,000 S1 records (randomized deterministic sample)")
    md.append("> - **Target Pool:** 250,000 S2 records + 250,000 S3 records (500,000 total target records)")
    md.append("> - **Ground Truth Pairs in Subset:** 2,530 positive ground-truth pairs indexed")
    md.append("> - **Singletons in Subset:** 12,470 S1 queries have no matching entity in the 500k target pool")
    md.append("> - **Extrapolation Warning:** Do **NOT** extrapolate subset recall directly to the full 10.3M target dataset. All figures reported below represent empirical measurements on this controlled benchmark.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append("1. **The Uncapped Ceiling:**")
    md.append("   - Prior experiments imposed an artificial `K=50` insertion-order cap, causing candidate saturation where loose address tokens displaced high-confidence name matches.")
    md.append("   - Evaluating the **complete uncapped candidate union (Union E)** reveals an uncapped pair recall ceiling of **91.66%** (91.59% Mean S1 Recall) with an average of **150.57 candidates per S1**.")
    md.append("2. **Candidate Budgeting & Pruning Policies:**")
    md.append("   - At strict budgets, **Multi-Block Support** consistently outperforms deterministic Blocking Strength Priority by **+2.0% to +3.0% recall** across all budgets ($K=25, 50, 100$).")
    md.append("   - At $K=25$, Multi-Block achieves **75.42% recall** (vs 72.49% for Priority).")
    md.append("   - At $K=50$, Multi-Block achieves **77.08% recall** (vs 74.43% for Priority).")
    md.append("   - At $K=100$, Multi-Block achieves **79.17% recall** (vs 77.00% for Priority).")
    md.append("   - At $K=200$, the candidate capacity accommodates the entire uncapped union, capturing the full **91.66% recall**.")
    md.append("3. **ROI and Information Gain of Individual Blocks:**")
    md.append("   - `house_name` is by far the highest-ROI block: adding it over `name_core` delivers **+19.73% delta recall** while adding only **+0.17 candidates per S1** (a recall gain rate of 0.1924 hits per candidate).")
    md.append("   - `combined_postal_name` achieves **97.22% precision** (105 true hits out of 108 candidates total across 15k queries) with essentially zero candidate inflation.")
    md.append("   - `char_3gram_k5` delivers **+23.68% delta recall** for only +9.64 candidates per S1.")
    md.append("   - Broad address tokens deliver +39.17% recall but introduce +53.79 candidates per S1 at 0.19% precision, requiring Multi-Block consensus or ML scoring to filter effectively.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 1: Individual Block Performance (Combined S2 + S3 Pool)")
    md.append("")
    md.append("Empirical metrics for each individual blocking strategy evaluated across all 15,000 S1 queries against the 500,000 combined target records.")
    md.append("")
    md.append("| Blocking Strategy | Pair Recall (%) | Per-S1 Recall (%) | Avg Cands/S1 | Median | P95 | P99 | Max | Candidate Precision (%) | Reduction Ratio | Runtime (s) | Peak Memory (MB) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    strat_order = [
        "exact_name", "name_core", "name_token", "rare_token",
        "char_3gram_k5", "char_3gram_k10", "char_3gram_k20",
        "postal", "house_name", "combined_postal_name", "address_token"
    ]

    for strat in strat_order:
        if strat in t1:
            m = t1[strat]
            md.append(
                f"| `{strat}` | {m['candidate_pair_recall']*100:.2f}% | {m['mean_s1_recall']*100:.2f}% | "
                f"{m['avg_candidates_per_s1']:.2f} | {m['median_candidates']:.0f} | {m['p95_candidates']:.0f} | "
                f"{m['p99_candidates']:.0f} | {m['max_candidates']} | {m['candidate_precision']*100:.4f}% | "
                f"{m['reduction_ratio']:.6f} | {m['runtime_sec']:.2f}s | {m['peak_memory_mb']:.1f} |"
            )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 2: Experiment 1 — Uncapped Candidate Union Performance")
    md.append("")
    md.append("Complete candidate union evaluation **without any candidate cap** or truncation. This measures the true empirical recall ceiling and candidate volume.")
    md.append("")
    md.append("- **Union A:** `exact_name` + `name_core`")
    md.append("- **Union B:** Union A + `rare_token`")
    md.append("- **Union C:** Union B + `char_3gram_k10`")
    md.append("- **Union D:** Union C + `house_name` + `combined_postal_name`")
    md.append("- **Union E:** Union D + `address_token`")
    md.append("")
    md.append("| Candidate Union | Pair Recall (%) | Per-S1 Recall (%) | Avg Cands/S1 | Median | P95 | P99 | Max | Candidate Precision (%) | Reduction Ratio |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for uname, m in t2.items():
        md.append(
            f"| **{uname}** | **{m['candidate_pair_recall']*100:.2f}%** | {m['mean_s1_recall']*100:.2f}% | "
            f"{m['avg_candidates_per_s1']:.2f} | {m['median_candidates']:.0f} | {m['p95_candidates']:.0f} | "
            f"{m['p99_candidates']:.0f} | {m['max_candidates']} | {m['candidate_precision']*100:.4f}% | "
            f"{m['reduction_ratio']:.6f} |"
        )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 3: Experiment 2 — Candidate Budgets (K = 25, 50, 100, 200)")
    md.append("")
    md.append("Pruning the complete uncapped Union E down to fixed budgets using deterministic ranking.")
    md.append("We compare two pruning policies:")
    md.append("1. **Blocking Strength Priority:** Ranks by strategy tier (`exact_name` > `name_core` > `house_name` > `combined_postal_name` > `rare_token` > `char_3gram` > `address_token`).")
    md.append("2. **Multi-Block Support:** Ranks candidates by how many independent blocking strategies generated them, breaking ties by sum of blocking strengths.")
    md.append("")
    md.append("| Candidate Budget (K) | Pruning Policy | Pair Recall (%) | Per-S1 Recall (%) | Captured True Pairs | Total Candidates | Avg Cands/S1 | Candidate Precision (%) |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    budget_keys = [
        ("K=25", "Priority", "K=25_Priority"),
        ("K=25", "Multi-Block Support", "K=25_MultiBlock"),
        ("K=50", "Priority", "K=50_Priority"),
        ("K=50", "Multi-Block Support", "K=50_MultiBlock"),
        ("K=100", "Priority", "K=100_Priority"),
        ("K=100", "Multi-Block Support", "K=100_MultiBlock"),
        ("K=200", "Priority", "K=200_Priority"),
        ("K=200", "Multi-Block Support", "K=200_MultiBlock"),
    ]

    for k_label, policy_label, bkey in budget_keys:
        if bkey in t3:
            m = t3[bkey]
            md.append(
                f"| {k_label} | {policy_label} | **{m['candidate_pair_recall']*100:.2f}%** | "
                f"{m['mean_s1_recall']*100:.2f}% | {m['captured_true_pairs']:,} / {m['total_true_pairs']:,} | "
                f"{m['total_candidates_generated']:,} | {m['avg_candidates_per_s1']:.2f} | {m['candidate_precision']*100:.4f}% |"
            )

    md.append("")
    md.append("### Key Takeaways from Budget Experiments:")
    md.append("- **Multi-Block Superiority:** At $K=25$, Multi-Block preserves **75.42% recall** vs 72.49% for Priority (+2.93% absolute gain). When an entity is matched across multiple independent signals (e.g. name core + house number), confidence is vastly higher than single-channel hits.")
    md.append("- **Budget Saturation Curve:** At $K=100$, Multi-Block reaches **79.17% recall**. The jump to $K=200$ (91.66%) captures the remaining ~12.5% tail that relies solely on isolated address tokens.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 4: Experiment 3 — Incremental Block Contribution over `name_core`")
    md.append("")
    md.append("Starting from baseline `name_core` (Pair Recall: 35.61%, Avg Cands: 1.47), each strategy is added individually **WITHOUT any candidate cap**.")
    md.append("")
    md.append("| Added Strategy | Base Recall (%) | New Recall (%) | Delta Recall (%) | Base Cands | New Cands | Delta Cands/S1 | Delta True Hits | Delta Total Cands | Recall Gain / Added Cand |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    inc_order = [
        "house_name", "combined_postal_name", "postal",
        "char_3gram_k5", "char_3gram_k10", "char_3gram_k20",
        "rare_token", "address_token"
    ]

    for strat in inc_order:
        if strat in t4:
            m = t4[strat]
            md.append(
                f"| `{strat}` ({m['block_label']}) | {m['base_recall']*100:.2f}% | {m['new_recall']*100:.2f}% | "
                f"**{m['delta_recall']*100:+.2f}%** | {m['base_candidates']:.2f} | {m['new_candidates']:.2f} | "
                f"{m['delta_candidates']:+.2f} | +{m['delta_true_hits']} | +{m['delta_total_candidates']:,} | "
                f"**{m['recall_gain_per_candidate']:.6f}** |"
            )

    md.append("")
    md.append("### Contribution Analysis:")
    md.append("1. **`house_name` (The Efficiency Leader):** Adds **+499 true hits** (+19.73% recall) at the cost of only **+2,593 candidates** (+0.17 cands/query), yielding a phenomenal efficiency of **0.1924 hits/candidate**.")
    md.append("2. **`combined_postal_name`:** Adds **+45 true hits** (+1.78% recall) with only **+48 candidates**, yielding **0.9375 hits/candidate** (near 100% precision).")
    md.append("3. **`char_3gram_k5` vs `k10` vs `k20`:**")
    md.append("   - `k=5` gives +23.68% recall for +9.64 candidates (0.0041 hits/cand).")
    md.append("   - Moving to `k=10` gains +2.49% recall for +9.89 candidates (0.0023 hits/cand).")
    md.append("   - Moving to `k=20` gains another +2.61% recall for +19.91 candidates (0.0012 hits/cand).")
    md.append("4. **`rare_token`:** Delivers +31.94% recall but adds +83.75 candidates/S1.")
    md.append("5. **`address_token`:** Captures the widest tail (+39.17% recall) but adds +53.79 candidates/S1.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 5: Experiment 4 — Deep House-Name Strategy Investigation")
    md.append("")
    md.append("Investigation into why `house_name` achieves 37.59% recall with only 0.20 candidates/S1.")
    md.append("")
    md.append("| Metric | Value | Technical Rationale & Analysis |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| Total True Matches Captured | **{t5['total_true_matches_captured']:,}** | 37.59% of all ground-truth pairs in subset |")
    md.append(f"| Total False Candidates Generated | **{t5['total_false_candidates_generated']:,}** | Out of 3,051 total candidates generated |")
    md.append(f"| Candidate Precision | **{t5['candidate_precision']*100:.2f}%** | 31.17% precision is remarkably high for blocking (vs ~0.1% for token blocks) |")
    md.append(f"| False-Candidate Rate | **{t5['false_candidate_rate']*100:.2f}%** | Only 68.83% of proposed candidates are negatives |")
    md.append(f"| Same House Number Match | **{t5['pct_exact_same_house_number']:.2f}%** | 100% of captured matches share identical normalized house numbers |")
    md.append(f"| Business Name Also Matches | **{t5['pct_name_also_matches']:.2f}%** | 47.53% share identical full normalized name; 52.47% have altered legal/name variants |")
    md.append(f"| Postal Code Also Matches | **{t5['pct_postal_code_also_matches']:.2f}%** | 8.52% (low because postal code is missing/abbreviated in many records) |")
    md.append(f"| Normalized Address Matches | **{t5['pct_address_also_matches']:.2f}%** | 19.24% exact string match; 80.76% have street abbreviation/formatting variations |")
    md.append(f"| Recovered **ONLY** by `house_name` | **{t5['pct_recovered_ONLY_by_house_name']:.2f}%** | **{t5['unique_hits_count']} true pairs** are completely invisible to exact name, name core, rare token, and char 3-gram |")
    md.append(f"| S1 → S2 Recall | **{t5['s2_recall']*100:.2f}%** | Consistent across sources |")
    md.append(f"| S1 → S3 Recall | **{t5['s3_recall']*100:.2f}%** | Consistent across sources |")
    md.append(f"| Geographic Distribution | **US: {t5['us_hits']} | India: {t5['india_hits']}** | 77.4% US / 22.6% India (France = 0 in train set) |")
    md.append(f"| Singletons Receiving Candidates | **{t5['singletons_receiving_house_cands']:,}** | 4.89% of 12,470 singletons received candidates, showing tight specificity |")
    md.append("")
    md.append("### Why `house_name` is NOT an artifact:")
    md.append("- `house_name` pairs the street number (e.g. `100`, `254`) with the first two tokens of the business name.")
    md.append("- This creates an extremely restrictive blocking key: two businesses must operate at the exact same building number AND share the first two words of their company name.")
    md.append("- In business entity resolution, branches, acquisitions, and DBA renames frequently keep the physical address and primary brand name while changing legal forms (`LLC` vs `Inc`), suffixes, or secondary brand descriptors.")
    md.append("- Crucially, **59 true pairs (6.20% of house_name matches)** were found *exclusively* by this strategy, proving it captures genuine ground-truth variation that pure name token matching misses.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 6: Experiment 5 — Source Separation (S1 → S2 vs S1 → S3)")
    md.append("")
    md.append("Performance breakdown evaluated independently against S2 (250,000 targets) and S3 (250,000 targets).")
    md.append("")
    md.append("| Strategy | S2 Pair Recall (%) | S2 Avg Cands | S2 Prec (%) | S3 Pair Recall (%) | S3 Avg Cands | S3 Prec (%) | S2 vs S3 Delta Recall |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for strat in strat_order:
        if strat in t6:
            m = t6[strat]
            delta = (m["s2_recall"] - m["s3_recall"]) * 100
            md.append(
                f"| `{strat}` | {m['s2_recall']*100:.2f}% | {m['s2_avg_cands']:.2f} | {m['s2_precision']*100:.3f}% | "
                f"{m['s3_recall']*100:.2f}% | {m['s3_avg_cands']:.2f} | {m['s3_precision']*100:.3f}% | {delta:+.2f}% |"
            )

    md.append("")
    md.append("### S2 vs S3 Observations:")
    md.append("- Across virtually all strategies, S1 → S2 recall is slightly higher than S1 → S3 (e.g. `name_core`: 36.27% vs 34.99%; `rare_token`: 60.54% vs 58.35%; `address_token`: 63.89% vs 59.49%).")
    md.append("- S3 exhibits higher textual divergence and address noise compared to S2.")
    md.append("- However, both sources respond harmoniously to `house_name` (37.09% vs 38.06%) and `char_3gram` (49.02% vs 48.16%), confirming the blocking primitives are universally robust across disparate sources.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TABLE 7: Experiment 6 — Hard Negative Impact & Categorization")
    md.append("")
    md.append("Candidate quality breakdown showing false candidate volume and false candidate rate per block.")
    md.append("")
    md.append("| Strategy | Total Candidates | True Hits Captured | False Candidates | Candidate Precision (%) | False Candidate Rate (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for strat in strat_order:
        if strat in t7:
            m = t7[strat]
            md.append(
                f"| `{strat}` | {m['total_candidates']:,} | {m['true_candidates']:,} | "
                f"{m['false_candidates']:,} | {m['candidate_precision']*100:.4f}% | {m['false_candidate_rate']*100:.2f}% |"
            )

    md.append("")
    md.append("### Mined Hard Negative Categories (Saved 600 Examples to `experiments/hard_blocking_negatives_budget.tsv`):")
    md.append("")
    md.append("A dedicated pool of 600 difficult false-positive candidate pairs was mined from the uncapped candidate union. These pairs represent realistic confusions that our subsequent matching model must learn to disambiguate:")
    md.append("")
    md.append("1. **Same Name, Different Entity:**")
    md.append("   - *Example:* `Christ Chapel` (Dundalk, MD) vs `Christ Chapel` (Helena, MT) [Generated by `exact_name`, `name_core`, `char_3gram`].")
    md.append("   - *Resolution Clue:* Geographically distant states, zero street overlap.")
    md.append("2. **Same Core Name, Different Entity:**")
    md.append("   - *Example:* `Christ Chapel` vs `Christ Chapel Inc.` (Buffalo, NY) vs `Christ Chapel Corp` (Damon, TX).")
    md.append("   - *Resolution Clue:* Distinct corporate suffixes and incompatible jurisdictions.")
    md.append("3. **Same House Number, Different Entity:**")
    md.append("   - *Example:* Two different businesses co-located at building number `2100` on different streets or cities.")
    md.append("   - *Resolution Clue:* Complete dissimilarity in business name tokens.")
    md.append("4. **Same Postal Code, Different Entity:**")
    md.append("   - *Example:* Multiple businesses sharing ZIP code `21222` with divergent trade names.")
    md.append("   - *Resolution Clue:* Address text and brand name tokens mismatch.")
    md.append("5. **Same Address, Different Entity (Commercial Complexes & Co-working):**")
    md.append("   - *Example:* Different businesses registered at the same suite/building address.")
    md.append("   - *Resolution Clue:* Suite/unit numbers and company name comparisons.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Synthesis & Architectural Recommendations for Phase 3")
    md.append("")
    md.append("1. **Recommended Production Candidate Generation Pipeline:**")
    md.append("   - **Active Strategies:** `exact_name` + `name_core` + `house_name` + `combined_postal_name` + `char_3gram_k10` + `rare_token` + `address_token`.")
    md.append("   - **Pruning Method:** Multi-Block Support ranking.")
    md.append("   - **Budget Recommendation:**")
    md.append("     - For **high-throughput training:** $K=50$ provides **77.08% pair recall** at 49.87 candidates/S1.")
    md.append("     - For **maximum recall inference:** $K=100$ provides **79.17% pair recall**, while $K=200$ captures the full **91.66% uncapped ceiling**.")
    md.append("2. **Next Steps for Phase 3 (Feature Engineering):**")
    md.append("   - Leverage the `blocking_sources` metadata directly as ranking features (e.g. `is_house_name_match`, `is_multi_block_supported`, `block_support_count`).")
    md.append("   - Train our discriminative ranker on the mined hard negatives (`experiments/hard_blocking_negatives_budget.tsv`) to ensure high precision on Macro-$F_{0.5}$.")
    md.append("")

    content = "\n".join(md)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report successfully generated at: {REPORT_PATH}")


if __name__ == "__main__":
    format_report()
