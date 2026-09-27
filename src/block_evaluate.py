"""Comprehensive evaluation engine for entity resolution candidate generation (Phase 2).

Measures candidate recall (overall and per-S1), candidate cardinality distributions
(Mean, Median, P95, P99, Max), reduction ratios, singleton impacts, and mines hard negatives.
"""

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENTS_DIR,
    LOGS_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.blocking import (
    AVAILABLE_STRATEGIES,
    CandidateGenerator,
    run_strategy_on_s1,
)
from src.normalize import normalize_dataframe
from src.utils import setup_logger, timer

logger = setup_logger("block_evaluate", log_file=LOGS_DIR / "blocking_eval.log")


def evaluate_candidate_set(
    candidates_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, Set[str]],
    total_s1_count: int,
    total_target_pool_size: int,
    target_prefix: str = "S2-",
) -> Dict[str, Any]:
    """Calculates all mandatory blocking evaluation metrics for a candidate set against ground truth.

    Args:
        candidates_dict: Mapping from s1_id to set of candidate entity IDs.
        ground_truth_dict: Mapping from s1_id to set of true matched entity IDs.
        total_s1_count: Total number of evaluated S1 queries.
        total_target_pool_size: Number of records in the candidate pool.
        target_prefix: Prefix filter for target source ('S2-' or 'S3-').

    Returns:
        Structured dictionary of evaluation metrics.
    """
    total_true_pairs = 0
    captured_true_pairs = 0
    s1_recalls: List[float] = []

    true_singleton_count = 0
    true_singletons_with_candidates = 0

    # Candidate count distribution
    cand_counts = [len(candidates_dict.get(s1_id, set())) for s1_id in ground_truth_dict]
    if not cand_counts:
        cand_counts = [0]

    for s1_id, true_all in ground_truth_dict.items():
        # Filter true matches to target source if prefix is specified
        true_matches = {m for m in true_all if m.startswith(target_prefix)} if target_prefix else set(true_all)
        cands = candidates_dict.get(s1_id, set())

        is_singleton = (len(true_matches) == 0)
        if is_singleton:
            true_singleton_count += 1
            if len(cands) > 0:
                true_singletons_with_candidates += 1
        else:
            total_true_pairs += len(true_matches)
            hits = len(true_matches.intersection(cands))
            captured_true_pairs += hits
            s1_recalls.append(hits / len(true_matches))

    # Aggregate metrics
    pair_recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    mean_s1_recall = float(np.mean(s1_recalls)) if s1_recalls else 1.0

    total_candidates_generated = sum(cand_counts)
    cartesian_space = total_s1_count * total_target_pool_size
    reduction_ratio = (
        1.0 - (total_candidates_generated / cartesian_space) if cartesian_space > 0 else 1.0
    )

    candidate_precision = (
        (captured_true_pairs / total_candidates_generated) if total_candidates_generated > 0 else 0.0
    )

    return {
        "candidate_pair_recall": round(pair_recall, 4),
        "mean_s1_recall": round(mean_s1_recall, 4),
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "total_candidates_generated": total_candidates_generated,
        "avg_candidates_per_s1": round(float(np.mean(cand_counts)), 2),
        "median_candidates": float(np.median(cand_counts)),
        "p95_candidates": float(np.percentile(cand_counts, 95)),
        "p99_candidates": float(np.percentile(cand_counts, 99)),
        "max_candidates": int(np.max(cand_counts)),
        "reduction_ratio": round(reduction_ratio, 6),
        "candidate_precision": round(candidate_precision, 6),
        "true_singleton_count": true_singleton_count,
        "singletons_receiving_candidates": true_singletons_with_candidates,
    }


def mine_hard_negatives(
    candidate_pairs: List[Tuple[str, str]],
    ground_truth_dict: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    output_path: Path,
    max_hard_negatives: int = 500,
) -> pd.DataFrame:
    """Identifies candidates with identical/near-identical names that are NOT true matches.

    Saves hard negatives to output TSV for downstream model training.
    """
    logger.info("Mining hard blocking negatives...")
    s1_map = s1_df.set_index("entity_id")
    s2_map = s2_df.set_index("entity_id")

    hard_negatives = []
    for s1_id, cid in candidate_pairs:
        true_set = ground_truth_dict.get(s1_id, set())
        if cid not in true_set and s1_id in s1_map.index and cid in s2_map.index:
            r1 = s1_map.loc[s1_id]
            r2 = s2_map.loc[cid]

            n1 = r1["name_norm"]
            n2 = r2["name_norm"]

            # Check if name is exact or very high similarity
            if n1 == n2 or (n1 and n2 and r1["name_core"] == r2["name_core"]):
                hard_negatives.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "country": r1["country_norm"],
                    "s1_name": r1["business_name"],
                    "candidate_name": r2["business_name"],
                    "s1_address": r1["business_address"],
                    "candidate_address": r2["business_address"],
                    "negative_reason": "Identical Name / Different Address Entity",
                })
                if len(hard_negatives) >= max_hard_negatives:
                    break

    df_neg = pd.DataFrame(hard_negatives)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_neg.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(df_neg):,} hard blocking negatives to {output_path}")
    return df_neg


def run_blocking_ablation(
    sample_queries: int = 15_000,
    target_sample_size: int = 250_000,
) -> Dict[str, Any]:
    """Runs rigorous multi-strategy blocking ablation against ground truth.

    Evaluates individual strategies, multi-strategy unions (A, B, C, D),
    and mines hard negatives.
    """
    logger.info("=" * 60)
    logger.info(f"STARTING PHASE 2 BLOCKING ABLATION ({sample_queries:,} queries vs {target_sample_size:,} pool)")
    logger.info("=" * 60)

    # 1. Load Ground Truth
    with timer("Loading ground truth", logger):
        gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", keep_default_na=False)
        gt_dict: Dict[str, Set[str]] = {}
        s1_ids = gt_df["source1_entity_id"].tolist()
        matched_strs = gt_df["matched_entity_ids"].tolist()
        for s1_id, m_str in zip(s1_ids, matched_strs):
            gt_dict[s1_id] = set(m_str.split(",")) if m_str.strip() else set()
        del gt_df

    # 2. Load and Normalize Query Sample (S1)
    s1_raw = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", nrows=sample_queries, dtype=str, keep_default_na=False)
    s1_norm = normalize_dataframe(s1_raw)
    query_gt = {s1_id: gt_dict.get(s1_id, set()) for s1_id in s1_norm["entity_id"]}

    # 3. Load and Normalize Target Pool (S2)
    s2_raw = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", nrows=target_sample_size, dtype=str, keep_default_na=False)
    s2_norm = normalize_dataframe(s2_raw)
    target_ids_set = set(s2_norm["entity_id"])

    # Ensure ground truth only evaluates true matches present in the indexed pool
    query_gt = {}
    for s1_id in s1_norm["entity_id"]:
        true_all = gt_dict.get(s1_id, set())
        query_gt[s1_id] = {m for m in true_all if m in target_ids_set}

    gen = CandidateGenerator("source2", s2_norm)

    # Individual Strategy Evaluation
    strategies_to_test = [
        "exact_name",
        "name_core",
        "name_token",
        "rare_token",
        "char_3gram_k5",
        "char_3gram_k10",
        "char_3gram_k20",
        "postal",
        "house_name",
        "address_token",
        "combined_postal_name",
    ]

    strategy_results: Dict[str, Any] = {}
    strategy_pair_maps: Dict[str, Dict[Tuple[str, str], Set[str]]] = {}

    import resource
    import sys
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024

    for strat in strategies_to_test:
        t0 = time.perf_counter()
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pairs = run_strategy_on_s1(strat, s1_norm, gen)
        elapsed = time.perf_counter() - t0
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = mem_after / divisor

        strategy_pair_maps[strat] = pairs

        # Map to s1_id -> candidate set
        cands_by_s1: Dict[str, Set[str]] = defaultdict(set)
        for (s1_id, cid) in pairs:
            cands_by_s1[s1_id].add(cid)

        metrics = evaluate_candidate_set(
            cands_by_s1,
            query_gt,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm),
            target_prefix="S2-",
        )
        metrics["runtime_sec"] = round(elapsed, 3)
        metrics["peak_memory_mb"] = round(peak_rss, 2)
        strategy_results[strat] = metrics
        logger.info(
            f"{strat:20} -> Recall: {metrics['candidate_pair_recall']*100:.2f}%, "
            f"Avg Cands: {metrics['avg_candidates_per_s1']}, P95: {metrics['p95_candidates']}, "
            f"Time: {metrics['runtime_sec']}s, Mem: {metrics['peak_memory_mb']}MB"
        )

    # Union Configurations:
    # Union A: exact_name + name_core
    # Union B: Union A + rare_token
    # Union C: Union B + char_3gram_k10
    # Union D: Union C + house_name + combined_postal_name
    # Union E: Union D + address_token (all strategies combined)
    unions = {
        "Union A (Exact + Core)": ["exact_name", "name_core"],
        "Union B (A + Rare Token)": ["exact_name", "name_core", "rare_token"],
        "Union C (B + Char 3-Gram K=10)": ["exact_name", "name_core", "rare_token", "char_3gram_k10"],
        "Union D (C + House/Postal)": [
            "exact_name",
            "name_core",
            "rare_token",
            "char_3gram_k10",
            "house_name",
            "combined_postal_name",
        ],
        "Union E (D + Address Token)": [
            "exact_name",
            "name_core",
            "rare_token",
            "char_3gram_k10",
            "house_name",
            "combined_postal_name",
            "address_token",
        ],
    }

    union_results: Dict[str, Any] = {}
    union_pairs_d: List[Tuple[str, str]] = []

    for uname, ustrats in unions.items():
        t0 = time.perf_counter()
        cands_by_s1 = defaultdict(set)
        for strat in ustrats:
            for (s1_id, cid) in strategy_pair_maps[strat]:
                cands_by_s1[s1_id].add(cid)

        # Apply maximum candidates per entity cap = 50
        capped_cands_by_s1 = {s1: set(list(cands)[:50]) for s1, cands in cands_by_s1.items()}
        elapsed = time.perf_counter() - t0
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor

        metrics = evaluate_candidate_set(
            capped_cands_by_s1,
            query_gt,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm),
            target_prefix="S2-",
        )
        metrics["runtime_sec"] = round(elapsed, 3)
        metrics["peak_memory_mb"] = round(peak_rss, 2)
        union_results[uname] = metrics
        logger.info(
            f"{uname:30} -> Recall: {metrics['candidate_pair_recall']*100:.2f}%, "
            f"Avg Cands: {metrics['avg_candidates_per_s1']}, P95: {metrics['p95_candidates']}, "
            f"Time: {metrics['runtime_sec']}s, Mem: {metrics['peak_memory_mb']}MB"
        )

        if uname == "Union D (C + House/Postal)":
            union_pairs_d = [(s1, cid) for s1, cands in capped_cands_by_s1.items() for cid in cands]

    # Incremental block analysis over name_core baseline
    logger.info("Computing incremental block contributions over name_core...")
    base_core_pairs = strategy_pair_maps["name_core"]
    base_core_cands = defaultdict(set)
    for (s1_id, cid) in base_core_pairs:
        base_core_cands[s1_id].add(cid)
    base_metrics = strategy_results["name_core"]

    blocks_to_test_incremental = [
        ("rare_token", "Rare Token (IDF)"),
        ("char_3gram_k10", "Char 3-Gram (K=10)"),
        ("postal", "Postal Code"),
        ("house_name", "House Number + Name"),
        ("address_token", "Address Token"),
        ("combined_postal_name", "Postal + Name Token"),
    ]

    incremental_results: Dict[str, Any] = {}
    for block_key, block_label in blocks_to_test_incremental:
        combined_cands = defaultdict(set)
        for s1_id, cands in base_core_cands.items():
            combined_cands[s1_id].update(cands)
        for (s1_id, cid) in strategy_pair_maps[block_key]:
            combined_cands[s1_id].add(cid)

        comb_metrics = evaluate_candidate_set(
            combined_cands,
            query_gt,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm),
            target_prefix="S2-",
        )

        delta_recall = comb_metrics["candidate_pair_recall"] - base_metrics["candidate_pair_recall"]
        delta_cands = comb_metrics["avg_candidates_per_s1"] - base_metrics["avg_candidates_per_s1"]
        delta_true_hits = comb_metrics["captured_true_pairs"] - base_metrics["captured_true_pairs"]
        delta_total_cands = comb_metrics["total_candidates_generated"] - base_metrics["total_candidates_generated"]
        efficiency = (delta_true_hits / delta_total_cands) if delta_total_cands > 0 else 0.0

        incremental_results[block_key] = {
            "block_label": block_label,
            "base_recall": base_metrics["candidate_pair_recall"],
            "combined_recall": comb_metrics["candidate_pair_recall"],
            "delta_recall": round(delta_recall, 4),
            "base_avg_cands": base_metrics["avg_candidates_per_s1"],
            "combined_avg_cands": comb_metrics["avg_candidates_per_s1"],
            "delta_avg_cands": round(delta_cands, 2),
            "delta_true_hits": delta_true_hits,
            "delta_total_cands": delta_total_cands,
            "recall_gain_per_additional_candidate": round(efficiency, 6),
        }
        logger.info(
            f"name_core + {block_label:25} -> +{delta_recall*100:+.2f}% recall, "
            f"+{delta_cands:+.2f} cands/S1 (Efficiency: {efficiency:.6f})"
        )

    # Mine hard negatives
    hard_neg_path = EXPERIMENTS_DIR / "hard_blocking_negatives.tsv"
    mine_hard_negatives(union_pairs_d, query_gt, s1_norm, s2_norm, hard_neg_path, max_hard_negatives=500)

    report_data = {
        "experiment_type": "SUBSET EXPERIMENT",
        "sample_queries": sample_queries,
        "target_pool_size": target_sample_size,
        "strategies": strategy_results,
        "unions": union_results,
        "incremental": incremental_results,
    }

    out_json = EXPERIMENTS_DIR / "phase2_blocking_data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        import json
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved Phase 2 evaluation data to {out_json}")

    return report_data


if __name__ == "__main__":
    run_blocking_ablation()
