"""Quality analysis, benchmarking, and real-data pair inspection for Phase 1 Normalization."""

import gc
import json
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config import (
    EXPERIMENTS_DIR,
    LOGS_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.load_data import load_ground_truth, load_train_source1, load_train_source2
from src.normalize import (
    extract_house_number,
    extract_name_core,
    extract_postal_code,
    normalize_address_base,
    normalize_business_name_base,
    normalize_dataframe,
    normalize_record,
)
from src.utils import setup_logger, timer

logger = setup_logger("analyze_normalization", log_file=LOGS_DIR / "phase1_analysis.log")


def benchmark_normalization(sample_size: int = 100_000) -> Dict[str, Any]:
    """Benchmarks throughput and memory footprint on a sample of training data."""
    logger.info(f"Loading {sample_size:,} records from train_source1 for benchmarking...")
    df = pd.read_csv(
        TRAIN_SOURCE1_PATH,
        sep="\t",
        nrows=sample_size,
        dtype=str,
        keep_default_na=False,
    )

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.perf_counter()

    norm_df = normalize_dataframe(df)

    elapsed = time.perf_counter() - start_time
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # On macOS ru_maxrss is in bytes, on Linux in kilobytes
    import sys
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    mem_diff_mb = (mem_after - mem_before) / divisor

    throughput = len(df) / elapsed if elapsed > 0 else 0

    stats = {
        "records_processed": len(df),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_records_per_sec": round(throughput, 1),
        "peak_memory_diff_mb": round(mem_diff_mb, 2),
    }

    logger.info(
        f"Benchmark Result: {stats['records_processed']:,} records in {stats['elapsed_seconds']}s "
        f"({stats['throughput_records_per_sec']:,.0f} records/sec). Memory Delta: {stats['peak_memory_diff_mb']} MB"
    )

    del df, norm_df
    gc.collect()
    return stats


def compute_normalization_quality_stats(sample_size: int = 200_000) -> Dict[str, Any]:
    """Computes comprehensive before/after transformation statistics."""
    logger.info(f"Computing quality statistics on {sample_size:,} records...")

    # Mix S1 (reference) and S2 (noisy) to capture both clean and missing address behavior
    s1 = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", nrows=sample_size // 2, dtype=str, keep_default_na=False)
    s2 = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", nrows=sample_size // 2, dtype=str, keep_default_na=False)
    df = pd.concat([s1, s2], ignore_index=True)
    del s1, s2
    gc.collect()

    total = len(df)
    norm_df = normalize_dataframe(df)

    # 1. Name stats
    names_raw = norm_df["business_name"].str.strip()
    names_norm = norm_df["name_norm"]
    names_core = norm_df["name_core"]

    name_changed = (names_raw != names_norm).sum()
    suffix_changed = (names_norm != names_core).sum()
    name_empty_after = (names_norm == "").sum()
    name_empty_before = (names_raw == "").sum()

    # Suspicious over-normalization: raw had >= 3 chars, but core became empty or single char
    suspicious_names = ((names_raw.str.len() >= 3) & (names_core.str.len() < 2)).sum()

    # 2. Address stats
    addr_raw = norm_df["business_address"].str.strip()
    addr_norm = norm_df["address_norm"]
    postal = norm_df["postal_code"]
    house = norm_df["house_number"]

    addr_missing = (addr_raw == "").sum()
    addr_valid_count = total - addr_missing
    addr_changed = ((addr_raw != addr_norm) & (addr_raw != "")).sum()
    addr_empty_unexpected = ((addr_raw != "") & (addr_norm == "")).sum()

    postal_extracted = (postal != "").sum()
    house_extracted = (house != "").sum()

    stats = {
        "total_records_analyzed": total,
        "name_stats": {
            "name_changed_count": int(name_changed),
            "name_changed_pct": round(name_changed / total * 100, 2),
            "suffix_removed_count": int(suffix_changed),
            "suffix_removed_pct": round(suffix_changed / total * 100, 2),
            "name_empty_before": int(name_empty_before),
            "name_empty_after": int(name_empty_after),
            "suspicious_over_normalized": int(suspicious_names),
        },
        "address_stats": {
            "address_missing_count": int(addr_missing),
            "address_missing_pct": round(addr_missing / total * 100, 2),
            "address_changed_count": int(addr_changed),
            "address_changed_pct": round(addr_changed / addr_valid_count * 100, 2) if addr_valid_count > 0 else 0.0,
            "postal_code_extracted_count": int(postal_extracted),
            "postal_code_extraction_rate_pct": round(postal_extracted / total * 100, 2),
            "postal_code_extraction_valid_rate_pct": round(postal_extracted / addr_valid_count * 100, 2) if addr_valid_count > 0 else 0.0,
            "house_number_extracted_count": int(house_extracted),
            "house_number_extraction_rate_pct": round(house_extracted / total * 100, 2),
            "house_number_extraction_valid_rate_pct": round(house_extracted / addr_valid_count * 100, 2) if addr_valid_count > 0 else 0.0,
            "address_empty_unexpected": int(addr_empty_unexpected),
        },
    }

    del df, norm_df
    gc.collect()
    return stats


def extract_real_ground_truth_pairs(
    num_standard: int = 50,
    num_hard: int = 25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts 50 diverse matching pairs and 20+ hard positive pairs from ground truth."""
    logger.info("Extracting ground-truth positive pairs...")

    # Load ground truth and slice of S1 & S2
    gt = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", keep_default_na=False, nrows=100_000)
    s1 = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", keep_default_na=False, nrows=100_000).set_index("entity_id")
    s2 = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", keep_default_na=False, nrows=250_000).set_index("entity_id")

    pairs: List[Tuple[pd.Series, pd.Series]] = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"].strip()
        if not matched_str or s1_id not in s1.index:
            continue
        for m_id in matched_str.split(","):
            if m_id.startswith("S2-") and m_id in s2.index:
                pairs.append((s1.loc[s1_id], s2.loc[m_id]))
                if len(pairs) >= 500:
                    break
        if len(pairs) >= 500:
            break

    logger.info(f"Found {len(pairs)} candidate ground truth pairs for inspection.")

    standard_examples: List[Dict[str, Any]] = []
    hard_examples: List[Dict[str, Any]] = []

    for r1, r2 in pairs:
        d1 = r1.to_dict()
        d1["entity_id"] = str(r1.name)
        d2 = r2.to_dict()
        d2["entity_id"] = str(r2.name)

        norm1 = normalize_record(d1)
        norm2 = normalize_record(d2)

        # Measure name token overlap
        t1 = set(norm1["name_sorted_tokens"].split())
        t2 = set(norm2["name_sorted_tokens"].split())
        overlap = len(t1.intersection(t2)) / max(len(t1.union(t2)), 1)

        # Detect transformation categories
        categories = []
        if r1["business_name"].lower() != r2["business_name"].lower():
            if norm1["name_sorted_tokens"] == norm2["name_sorted_tokens"]:
                categories.append("Word Order Transposition")
            elif norm1["name_core"] == norm2["name_core"]:
                categories.append("Legal Suffix Difference")
            elif "&" in r1["business_name"] or "&" in r2["business_name"]:
                categories.append("Ampersand Variation")
            else:
                categories.append("Name Spelling/Abbreviation")

        if r1["business_address"].lower() != r2["business_address"].lower():
            if norm1["address_norm"] == norm2["address_norm"]:
                categories.append("Address Abbreviation Standardization")
            elif not r2["business_address"]:
                categories.append("Missing Address in S2")
            else:
                categories.append("Address Formatting / Detail Difference")

        example = {
            "s1_id": norm1["entity_id"],
            "s2_id": norm2["entity_id"],
            "country": norm1["country"],
            "s1_name_raw": norm1["business_name"],
            "s2_name_raw": norm2["business_name"],
            "s1_name_norm": norm1["name_norm"],
            "s2_name_norm": norm2["name_norm"],
            "s1_name_core": norm1["name_core"],
            "s2_name_core": norm2["name_core"],
            "s1_addr_raw": norm1["business_address"],
            "s2_addr_raw": norm2["business_address"],
            "s1_addr_norm": norm1["address_norm"],
            "s2_addr_norm": norm2["address_norm"],
            "s1_postal": norm1["postal_code"],
            "s2_postal": norm2["postal_code"],
            "s1_house": norm1["house_number"],
            "s2_house": norm2["house_number"],
            "token_overlap": round(overlap, 3),
            "transformations": ", ".join(categories) if categories else "Identical / Case only",
        }

        # Classify as hard if token overlap is low or address is drastically different
        is_hard = (overlap < 0.6) or (r2["business_address"] == "") or ("BRENTWOOD" not in norm2["address_norm"] and overlap < 0.8 and r1["business_name"] != r2["business_name"])

        if is_hard and len(hard_examples) < num_hard:
            hard_examples.append(example)
        elif len(standard_examples) < num_standard:
            standard_examples.append(example)

        if len(standard_examples) >= num_standard and len(hard_examples) >= num_hard:
            break

    logger.info(f"Selected {len(standard_examples)} standard examples and {len(hard_examples)} hard examples.")
    return standard_examples, hard_examples


def run_phase1_analysis() -> Dict[str, Any]:
    """Runs complete Phase 1 analysis and returns compiled report dictionary."""
    with timer("Phase 1 Benchmarking", logger):
        benchmark_results = benchmark_normalization(sample_size=100_000)

    with timer("Phase 1 Quality Statistics", logger):
        quality_stats = compute_normalization_quality_stats(sample_size=200_000)

    with timer("Phase 1 Real Data Inspection", logger):
        standard_examples, hard_examples = extract_real_ground_truth_pairs(num_standard=55, num_hard=25)

    report = {
        "benchmark": benchmark_results,
        "quality": quality_stats,
        "standard_examples": standard_examples,
        "hard_examples": hard_examples,
    }

    # Save to JSON artifact
    out_file = EXPERIMENTS_DIR / "phase1_analysis_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved analysis data to {out_file}")

    return report


if __name__ == "__main__":
    run_phase1_analysis()
