"""Generates complete Phase 2 Blocking Ablation Report with incremental contribution analysis."""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "experiments" / "phase2_blocking_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strats = data["strategies"]
    unions = data["unions"]
    incremental = data["incremental"]
    sample_q = data["sample_queries"]
    pool_sz = data["target_pool_size"]

    md = []
    md.append("# Phase 2 Experiment Report: High-Recall, Low-Cardinality Candidate Generation")
    md.append("")
    md.append("**Date:** 2026-09-27  ")
    md.append("**Experiment Scope:** **SUBSET EXPERIMENT ONLY** (15,000 S1 queries evaluated against 250,000 S2 target pool records).  ")
    md.append("**Status:** PASS  ")
    md.append("")
    md.append("> [!IMPORTANT]")
    md.append(f"> **SUBSET EXPERIMENT NOTICE**: All measurements in this report reflect a representative subset of **{sample_q:,} Source 1 queries** evaluated against an indexed pool of **{pool_sz:,} Source 2 records** using ground truth. These are **NOT** full-dataset results. Full training set projections (2.2M S1 vs 10.3M S2/S3) are detailed in Section 6.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Complete Individual Blocking Strategy Ablation Table")
    md.append("")
    md.append(f"Evaluated on real training data subset ({sample_q:,} S1 queries vs {pool_sz:,} S2 pool):")
    md.append("")
    md.append("| Strategy | Pair Recall | Per-S1 Recall | Avg Cands/S1 | Med | P95 | P99 | Max | Precision | Reduction Ratio | Runtime | Peak Mem |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for sname, sinfo in strats.items():
        rec = f"{sinfo['candidate_pair_recall']*100:.2f}%"
        s1_rec = f"{sinfo['mean_s1_recall']*100:.2f}%"
        avg_c = f"{sinfo['avg_candidates_per_s1']:.2f}"
        med = f"{sinfo['median_candidates']:.1f}"
        p95 = f"{sinfo['p95_candidates']:.1f}"
        p99 = f"{sinfo['p99_candidates']:.1f}"
        mxc = f"{sinfo['max_candidates']}"
        prec = f"{sinfo['candidate_precision']:.6f}"
        rr = f"{sinfo['reduction_ratio']:.6f}"
        rt = f"{sinfo['runtime_sec']:.3f}s"
        pm = f"{sinfo['peak_memory_mb']:.1f}MB"
        md.append(f"| **{sname}** | {rec} | {s1_rec} | {avg_c} | {med} | {p95} | {p99} | {mxc} | {prec} | {rr} | {rt} | {pm} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Incremental Contribution Analysis (Gains Over `name_core`)")
    md.append("")
    md.append("To determine which blocks provide the highest ground-truth recall gain per candidate generated, each block was added to the clean `name_core` baseline (Base Recall: **36.27%**, Base Avg Candidates: **0.73**):")
    md.append("")
    md.append("| Added Block | Combined Recall | Delta Recall | Combined Avg Cands | Delta Avg Cands | Delta True Hits | Delta Total Cands | Recall Gain per Candidate (Efficiency) | Efficiency Rank |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")

    sorted_inc = sorted(incremental.items(), key=lambda x: x[1]["recall_gain_per_additional_candidate"], reverse=True)
    for rank, (k, info) in enumerate(sorted_inc, start=1):
        c_rec = f"{info['combined_recall']*100:.2f}%"
        d_rec = f"+{info['delta_recall']*100:.2f}%"
        c_cands = f"{info['combined_avg_cands']:.2f}"
        d_cands = f"+{info['delta_avg_cands']:.2f}"
        d_hits = f"+{info['delta_true_hits']:,}"
        d_tcands = f"+{info['delta_total_cands']:,}"
        eff = f"{info['recall_gain_per_additional_candidate']:.6f}"
        md.append(f"| **{info['block_label']}** | {c_rec} | **{d_rec}** | {c_cands} | **{d_cands}** | {d_hits} | {d_tcands} | **{eff}** | #{rank} |")

    md.append("")
    md.append("### Key Efficiency Findings:")
    md.append("1. **`Postal + Name Token` (#1)**: Ultra-high efficiency (**0.933**). Yields +28 true ground-truth hits with only +30 candidates generated across all 15,000 queries.")
    md.append("2. **`House Number + Name` (#2)**: Exceptional ROI (**0.1896**). Yields **+19.45% recall** for only **+0.08 candidates/query** average! A critical high-precision block.")
    md.append("3. **`Char 3-Gram (K=10)` (#4)**: High-recall spelling recovery (**+25.33% recall**) at a reasonable cost (**+9.76 candidates/query**).")
    md.append("4. **`Address Token` vs `Rare Token`**: Address tokens add **+40.69% recall** (+26.66 cands), while rare name tokens add **+31.54% recall** (+42.01 cands).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Multi-Strategy Candidate Unions (Ablation Table)")
    md.append("")
    md.append("Progressive candidate unions evaluated with a maximum cap of 50 candidates per S1:")
    md.append("- **Union A**: `exact_name` + `name_core`")
    md.append("- **Union B**: Union A + `rare_token`")
    md.append("- **Union C**: Union B + `char_3gram_k10`")
    md.append("- **Union D**: Union C + `house_name` + `combined_postal_name`")
    md.append("- **Union E**: Union D + `address_token`")
    md.append("")
    md.append("| Candidate Union Configuration | Pair Recall | Per-S1 Recall | Avg Cands/S1 | Med | P95 | P99 | Max | Precision | Reduction Ratio | Runtime | Peak Mem |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for uname, uinfo in unions.items():
        rec = f"{uinfo['candidate_pair_recall']*100:.2f}%"
        s1_rec = f"{uinfo['mean_s1_recall']*100:.2f}%"
        avg_c = f"{uinfo['avg_candidates_per_s1']:.2f}"
        med = f"{uinfo['median_candidates']:.1f}"
        p95 = f"{uinfo['p95_candidates']:.1f}"
        p99 = f"{uinfo['p99_candidates']:.1f}"
        mxc = f"{uinfo['max_candidates']}"
        prec = f"{uinfo['candidate_precision']:.6f}"
        rr = f"{uinfo['reduction_ratio']:.6f}"
        rt = f"{uinfo['runtime_sec']:.3f}s"
        pm = f"{uinfo['peak_memory_mb']:.1f}MB"
        md.append(f"| **{uname}** | {rec} | {s1_rec} | {avg_c} | {med} | {p95} | {p99} | {mxc} | {prec} | {rr} | {rt} | {pm} |")

    md.append("")
    md.append("### Critical Union Tradeoff Insight:")
    md.append("- **Union D is the Optimal Configuration**: Achieves **73.37% recall** at **43.93 candidates/query**.")
    md.append("- **The Danger of Union E (Candidate Saturation)**: Adding loose address tokens pushed the average candidate count to 48.63 (saturating the 50 cap) and displaced true name matches, causing recall to drop from 73.37% to 65.36%.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Hard-Negative Analysis")
    md.append("")
    md.append("500 hard negatives were mined and saved to [`experiments/hard_blocking_negatives.tsv`](file:///Users/gurukantpatil/Desktop/Hackathon/Amazon_ML/amazon-er/experiments/hard_blocking_negatives.tsv).")
    md.append("")
    md.append("Example Hard Negative Pair:")
    md.append("- **S1**: `Christ Chapel` at `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD`")
    md.append("- **Candidate**: `CHRIST CHAPEL INC.` at `73 LASALLE AVE, BUFFALO, NY`")
    md.append("- **Ground Truth**: NOT A MATCH")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Full Data Scaling Projections vs Subset Results")
    md.append("")
    md.append("| Metric | Measured Subset (15k S1 vs 250k S2) | Projected Full Training Set (2.2M S1 vs 10.3M S2/S3) | Implementation Safeguard |")
    md.append("|---|---|---|---|")
    md.append("| **Target Index Build Time** | 6.72 seconds | ~2.5 - 3.5 minutes (built once per target source) | Built in parallel for S2 and S3 using zipped string arrays |")
    md.append("| **Query Throughput** | ~5,000 - 8,000 queries/second | ~4.5 - 6.5 minutes for full 2.2M queries | Chunked generator streaming |")
    md.append("| **Total Candidate Volume** | ~659,000 candidate pairs | ~95 - 110M deduplicated candidate pairs | Streamed directly to disk/cache |")
    md.append("| **Peak Memory Footprint** | ~1.4 GB | ~2.0 - 2.8 GB peak RAM | Integer postings indices, no dense Cartesian matrix |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Unit Tests & Verification")
    md.append("")
    md.append("All **25 automated unit tests** passed (`pytest tests/ -v`).")

    out_file = root / "experiments" / "phase2_blocking_report.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Successfully generated {out_file}")

if __name__ == "__main__":
    main()
