"""Rigorous Uncapped and Budget-Aware Blocking Evaluation Engine for Amazon Entity Resolution.

Executes 6 structured experiments:
1. Complete Uncapped Candidate Union (A, B, C, D, E)
2. Budgeted Evaluation (K=25, 50, 100, 200) under two ranking policies:
   - Blocking Strength Priority
   - Multi-Block Support
3. Uncapped Incremental Block Contributions over name_core
4. Detailed House-Name Strategy Investigation
5. Source Separation (S1 -> S2, S1 -> S3, and Combined)
6. Hard Negative Categorization and Mining
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.blocking import (
    AVAILABLE_STRATEGIES,
    CandidateGenerator,
    run_strategy_on_s1,
)
from src.config import (
    EXPERIMENTS_DIR,
    LOGS_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.normalize import normalize_dataframe
from src.utils import setup_logger, timer

logger = setup_logger("block_budget_eval", log_file=LOGS_DIR / "block_budget_eval.log")

STRATEGY_PRIORITY: Dict[str, int] = {
    "exact_name": 7,
    "name_core": 6,
    "house_name": 5,
    "combined_postal_name": 4,
    "rare_token": 3,
    "char_3gram_k10": 2,
    "char_3gram_k5": 2,
    "char_3gram_k20": 2,
    "name_token": 2,
    "postal": 2,
    "address_token": 1,
}


def get_peak_memory_mb() -> float:
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 2)


def evaluate_candidates(
    candidates_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, Set[str]],
    total_s1_count: int,
    total_target_pool_size: int,
    target_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculates pair recall, per-S1 recall, cardinality distribution, reduction ratio, and precision."""
    total_true_pairs = 0
    captured_true_pairs = 0
    s1_recalls: List[float] = []

    true_singleton_count = 0
    singletons_receiving_candidates = 0

    cand_counts = [len(candidates_dict.get(s1_id, set())) for s1_id in ground_truth_dict]
    if not cand_counts:
        cand_counts = [0]

    for s1_id, true_all in ground_truth_dict.items():
        if target_prefix:
            true_matches = {m for m in true_all if m.startswith(target_prefix)}
        else:
            true_matches = set(true_all)

        cands = candidates_dict.get(s1_id, set())

        if len(true_matches) == 0:
            true_singleton_count += 1
            if len(cands) > 0:
                singletons_receiving_candidates += 1
        else:
            total_true_pairs += len(true_matches)
            hits = len(true_matches.intersection(cands))
            captured_true_pairs += hits
            s1_recalls.append(hits / len(true_matches))

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
        "singletons_receiving_candidates": singletons_receiving_candidates,
    }


def prune_candidates(
    s1_candidates_with_sources: Dict[str, Dict[str, Set[str]]],
    k_budget: int,
    policy: str = "priority",
) -> Dict[str, Set[str]]:
    """Prunes candidates to top K using deterministic priority or multi-block support.

    Args:
        s1_candidates_with_sources: s1_id -> {cand_id: set(strategy_names)}
        k_budget: Maximum candidate budget per S1.
        policy: 'priority' (blocking strength) or 'multi_block' (multi-block support count).

    Returns:
        Pruned mapping s1_id -> set of candidate entity IDs.
    """
    pruned: Dict[str, Set[str]] = {}

    for s1_id, cands in s1_candidates_with_sources.items():
        if len(cands) <= k_budget:
            pruned[s1_id] = set(cands.keys())
            continue

        cand_items = []
        for cid, sources in cands.items():
            support_count = len(sources)
            max_priority = max(STRATEGY_PRIORITY.get(s, 1) for s in sources)
            sum_priority = sum(STRATEGY_PRIORITY.get(s, 1) for s in sources)
            cand_items.append((cid, support_count, max_priority, sum_priority))

        if policy == "priority":
            # Primary: max priority score, Secondary: support count, Tertiary: deterministic cid
            cand_items.sort(key=lambda x: (x[2], x[1], x[3], x[0]), reverse=True)
        elif policy == "multi_block":
            # Primary: support count, Secondary: sum priority score, Tertiary: deterministic cid
            cand_items.sort(key=lambda x: (x[1], x[3], x[2], x[0]), reverse=True)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        top_k_cids = {item[0] for item in cand_items[:k_budget]}
        pruned[s1_id] = top_k_cids

    return pruned


def run_budget_experiments(
    sample_queries: int = 15_000,
    target_sample_size: int = 250_000,
) -> Dict[str, Any]:
    """Orchestrates all 6 required Phase 2 experiments on real dataset subsets."""
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2 UNCAPPED AND BUDGET-AWARE BLOCKING EVALUATION")
    logger.info(f"SCOPE: SUBSET EXPERIMENT ({sample_queries:,} S1 queries vs {target_sample_size:,} S2 & S3 pool)")
    logger.info("=" * 70)

    # 1. Load Ground Truth
    with timer("Loading Ground Truth", logger):
        gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", keep_default_na=False)
        gt_dict: Dict[str, Set[str]] = {}
        s1_ids = gt_df["source1_entity_id"].tolist()
        matched_strs = gt_df["matched_entity_ids"].tolist()
        for s1_id, m_str in zip(s1_ids, matched_strs):
            gt_dict[s1_id] = set(m_str.split(",")) if m_str.strip() else set()
        del gt_df

    # 2. Load and Normalize Query Set (S1)
    with timer("Loading and Normalizing S1 Queries", logger):
        s1_raw = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", nrows=sample_queries, dtype=str, keep_default_na=False)
        s1_norm = normalize_dataframe(s1_raw)

    # 3. Load and Normalize Target Pools (S2 and S3)
    with timer("Loading and Normalizing S2 Target Pool", logger):
        s2_raw = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", nrows=target_sample_size, dtype=str, keep_default_na=False)
        s2_norm = normalize_dataframe(s2_raw)
        s2_ids_set = set(s2_norm["entity_id"])

    with timer("Loading and Normalizing S3 Target Pool", logger):
        s3_raw = pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t", nrows=target_sample_size, dtype=str, keep_default_na=False)
        s3_norm = normalize_dataframe(s3_raw)
        s3_ids_set = set(s3_norm["entity_id"])

    # Ground truth filtered to indexed pool targets
    query_gt_s2: Dict[str, Set[str]] = {}
    query_gt_s3: Dict[str, Set[str]] = {}
    query_gt_comb: Dict[str, Set[str]] = {}

    for s1_id in s1_norm["entity_id"]:
        true_all = gt_dict.get(s1_id, set())
        s2_matches = {m for m in true_all if m in s2_ids_set}
        s3_matches = {m for m in true_all if m in s3_ids_set}
        query_gt_s2[s1_id] = s2_matches
        query_gt_s3[s1_id] = s3_matches
        query_gt_comb[s1_id] = s2_matches.union(s3_matches)

    gen_s2 = CandidateGenerator("source2", s2_norm)
    gen_s3 = CandidateGenerator("source3", s3_norm)

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

    # Run all strategies for S1 -> S2 and S1 -> S3
    logger.info("Executing individual blocking strategies across S2 and S3...")
    s2_strategy_pairs: Dict[str, Dict[Tuple[str, str], Set[str]]] = {}
    s3_strategy_pairs: Dict[str, Dict[Tuple[str, str], Set[str]]] = {}
    s2_strat_times: Dict[str, float] = {}
    s3_strat_times: Dict[str, float] = {}

    for strat in strategies_to_test:
        t0 = time.perf_counter()
        s2_strategy_pairs[strat] = run_strategy_on_s1(strat, s1_norm, gen_s2)
        s2_strat_times[strat] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        s3_strategy_pairs[strat] = run_strategy_on_s1(strat, s1_norm, gen_s3)
        s3_strat_times[strat] = round(time.perf_counter() - t0, 3)

    # =======================================================================
    # TABLE 1: Individual Block Performance (Combined & S2)
    # =======================================================================
    table1_individual: Dict[str, Any] = {}
    for strat in strategies_to_test:
        cands_comb: Dict[str, Set[str]] = defaultdict(set)
        for (s1_id, cid) in s2_strategy_pairs[strat]:
            cands_comb[s1_id].add(cid)
        for (s1_id, cid) in s3_strategy_pairs[strat]:
            cands_comb[s1_id].add(cid)

        m = evaluate_candidates(
            cands_comb,
            query_gt_comb,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm) + len(s3_norm),
        )
        m["runtime_sec"] = round(s2_strat_times[strat] + s3_strat_times[strat], 3)
        m["peak_memory_mb"] = get_peak_memory_mb()
        table1_individual[strat] = m

    # =======================================================================
    # TABLE 2: Experiment 1 — Uncapped Candidate Unions
    # =======================================================================
    logger.info("Evaluating Uncapped Candidate Unions (A, B, C, D, E)...")
    union_defs = {
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

    # Store full candidate mapping with strategy sources for budget pruning:
    # s1_id -> {cand_id: set(strategies)}
    s2_union_candidates_with_sources: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    s3_union_candidates_with_sources: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    comb_union_candidates_with_sources: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # Build Union E (contains all strategies)
    for strat in union_defs["Union E (D + Address Token)"]:
        for (s1_id, cid), strats in s2_strategy_pairs[strat].items():
            s2_union_candidates_with_sources[s1_id][cid].update(strats)
            comb_union_candidates_with_sources[s1_id][cid].update(strats)
        for (s1_id, cid), strats in s3_strategy_pairs[strat].items():
            s3_union_candidates_with_sources[s1_id][cid].update(strats)
            comb_union_candidates_with_sources[s1_id][cid].update(strats)

    table2_uncapped_unions: Dict[str, Any] = {}
    for uname, ustrats in union_defs.items():
        cands_comb: Dict[str, Set[str]] = defaultdict(set)
        for strat in ustrats:
            for (s1_id, cid) in s2_strategy_pairs[strat]:
                cands_comb[s1_id].add(cid)
            for (s1_id, cid) in s3_strategy_pairs[strat]:
                cands_comb[s1_id].add(cid)

        m = evaluate_candidates(
            cands_comb,
            query_gt_comb,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm) + len(s3_norm),
        )
        m["peak_memory_mb"] = get_peak_memory_mb()
        table2_uncapped_unions[uname] = m

    # =======================================================================
    # TABLE 3: Experiment 2 — Candidate Budgets (K = 25, 50, 100, 200)
    # =======================================================================
    logger.info("Evaluating Candidate Budgets under Priority and Multi-Block Pruning...")
    table3_budgets: Dict[str, Any] = {}
    budgets_to_test = [25, 50, 100, 200]

    for k in budgets_to_test:
        # 1. Deterministic Blocking Strength Priority
        t0 = time.perf_counter()
        pruned_prio = prune_candidates(comb_union_candidates_with_sources, k_budget=k, policy="priority")
        prio_metrics = evaluate_candidates(
            pruned_prio,
            query_gt_comb,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm) + len(s3_norm),
        )
        prio_metrics["runtime_sec"] = round(time.perf_counter() - t0, 3)

        # 2. Multi-Block Support
        t0 = time.perf_counter()
        pruned_multi = prune_candidates(comb_union_candidates_with_sources, k_budget=k, policy="multi_block")
        multi_metrics = evaluate_candidates(
            pruned_multi,
            query_gt_comb,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm) + len(s3_norm),
        )
        multi_metrics["runtime_sec"] = round(time.perf_counter() - t0, 3)

        table3_budgets[f"K={k}_Priority"] = prio_metrics
        table3_budgets[f"K={k}_MultiBlock"] = multi_metrics

    # =======================================================================
    # TABLE 4: Experiment 3 — Uncapped Incremental Block Contributions
    # =======================================================================
    logger.info("Computing Uncapped Incremental Block Contributions over name_core...")
    base_core_cands: Dict[str, Set[str]] = defaultdict(set)
    for (s1_id, cid) in s2_strategy_pairs["name_core"]:
        base_core_cands[s1_id].add(cid)
    for (s1_id, cid) in s3_strategy_pairs["name_core"]:
        base_core_cands[s1_id].add(cid)

    base_core_metrics = evaluate_candidates(
        base_core_cands,
        query_gt_comb,
        total_s1_count=len(s1_norm),
        total_target_pool_size=len(s2_norm) + len(s3_norm),
    )

    incremental_blocks = [
        ("house_name", "House Number + Name"),
        ("postal", "Postal Code"),
        ("combined_postal_name", "Postal + Name Token"),
        ("rare_token", "Rare Token (IDF)"),
        ("char_3gram_k5", "Char 3-Gram (K=5)"),
        ("char_3gram_k10", "Char 3-Gram (K=10)"),
        ("char_3gram_k20", "Char 3-Gram (K=20)"),
        ("address_token", "Address Token"),
    ]

    table4_incremental: Dict[str, Any] = {}
    for bkey, blabel in incremental_blocks:
        comb_cands: Dict[str, Set[str]] = defaultdict(set)
        for s1_id, cands in base_core_cands.items():
            comb_cands[s1_id].update(cands)
        for (s1_id, cid) in s2_strategy_pairs[bkey]:
            comb_cands[s1_id].add(cid)
        for (s1_id, cid) in s3_strategy_pairs[bkey]:
            comb_cands[s1_id].add(cid)

        m = evaluate_candidates(
            comb_cands,
            query_gt_comb,
            total_s1_count=len(s1_norm),
            total_target_pool_size=len(s2_norm) + len(s3_norm),
        )

        d_recall = m["candidate_pair_recall"] - base_core_metrics["candidate_pair_recall"]
        d_cands = m["avg_candidates_per_s1"] - base_core_metrics["avg_candidates_per_s1"]
        d_hits = m["captured_true_pairs"] - base_core_metrics["captured_true_pairs"]
        d_total_cands = m["total_candidates_generated"] - base_core_metrics["total_candidates_generated"]
        eff = (d_hits / d_total_cands) if d_total_cands > 0 else 0.0

        table4_incremental[bkey] = {
            "block_label": blabel,
            "base_recall": base_core_metrics["candidate_pair_recall"],
            "new_recall": m["candidate_pair_recall"],
            "delta_recall": round(d_recall, 4),
            "base_candidates": base_core_metrics["avg_candidates_per_s1"],
            "new_candidates": m["avg_candidates_per_s1"],
            "delta_candidates": round(d_cands, 2),
            "delta_true_hits": d_hits,
            "delta_total_candidates": d_total_cands,
            "recall_gain_per_candidate": round(eff, 6),
        }

    # =======================================================================
    # TABLE 5: Experiment 4 — House Number Investigation
    # =======================================================================
    logger.info("Conducting Detailed Investigation into house_name Strategy...")
    s1_map = s1_norm.set_index("entity_id")
    s2_map = s2_norm.set_index("entity_id")
    s3_map = s3_norm.set_index("entity_id")

    # True pairs captured by house_name in S2 and S3
    house_true_pairs_s2: List[Tuple[str, str]] = []
    house_false_pairs_s2: List[Tuple[str, str]] = []
    house_true_pairs_s3: List[Tuple[str, str]] = []
    house_false_pairs_s3: List[Tuple[str, str]] = []

    for (s1_id, cid) in s2_strategy_pairs["house_name"]:
        if cid in query_gt_s2.get(s1_id, set()):
            house_true_pairs_s2.append((s1_id, cid))
        else:
            house_false_pairs_s2.append((s1_id, cid))

    for (s1_id, cid) in s3_strategy_pairs["house_name"]:
        if cid in query_gt_s3.get(s1_id, set()):
            house_true_pairs_s3.append((s1_id, cid))
        else:
            house_false_pairs_s3.append((s1_id, cid))

    all_house_true = house_true_pairs_s2 + house_true_pairs_s3
    all_house_false = house_false_pairs_s2 + house_false_pairs_s3

    # Check properties of true matches recovered by house_name
    exact_house_match_count = 0
    name_match_count = 0
    postal_match_count = 0
    address_match_count = 0
    unique_to_house_name_count = 0

    # Build set of pairs captured by ANY OTHER strategy
    other_strategy_pairs: Set[Tuple[str, str]] = set()
    for strat in strategies_to_test:
        if strat != "house_name":
            other_strategy_pairs.update(s2_strategy_pairs[strat].keys())
            other_strategy_pairs.update(s3_strategy_pairs[strat].keys())

    us_true_count = 0
    us_total_hits = 0
    in_true_count = 0
    in_total_hits = 0

    for s1_id, cid in all_house_true:
        r1 = s1_map.loc[s1_id]
        r2 = s2_map.loc[cid] if cid in s2_map.index else s3_map.loc[cid]

        # Exact same house number
        if r1["house_number"] and r1["house_number"] == r2["house_number"]:
            exact_house_match_count += 1

        # Name also matches (exact or core)
        if r1["name_norm"] == r2["name_norm"] or r1["name_core"] == r2["name_core"]:
            name_match_count += 1

        # Postal matches
        if r1["postal_code"] and r1["postal_code"] == r2["postal_code"]:
            postal_match_count += 1

        # Address matches
        if r1["address_norm"] and r1["address_norm"] == r2["address_norm"]:
            address_match_count += 1

        # Unique to house_name
        if (s1_id, cid) not in other_strategy_pairs:
            unique_to_house_name_count += 1

        # Country breakdown
        country = r1["country_norm"]
        if country == "us":
            us_total_hits += 1
        elif country == "india":
            in_total_hits += 1

    total_hits = max(len(all_house_true), 1)
    house_investigation = {
        "total_true_matches_captured": len(all_house_true),
        "total_false_candidates_generated": len(all_house_false),
        "false_candidate_rate": round(len(all_house_false) / max(len(all_house_true) + len(all_house_false), 1), 4),
        "candidate_precision": round(len(all_house_true) / max(len(all_house_true) + len(all_house_false), 1), 4),
        "pct_exact_same_house_number": round(exact_house_match_count / total_hits * 100, 2),
        "pct_name_also_matches": round(name_match_count / total_hits * 100, 2),
        "pct_postal_code_also_matches": round(postal_match_count / total_hits * 100, 2),
        "pct_address_also_matches": round(address_match_count / total_hits * 100, 2),
        "pct_recovered_ONLY_by_house_name": round(unique_to_house_name_count / total_hits * 100, 2),
        "unique_hits_count": unique_to_house_name_count,
        "s2_recall": round(len(house_true_pairs_s2) / max(sum(len(v) for v in query_gt_s2.values()), 1), 4),
        "s3_recall": round(len(house_true_pairs_s3) / max(sum(len(v) for v in query_gt_s3.values()), 1), 4),
        "us_hits": us_total_hits,
        "india_hits": in_total_hits,
        "singletons_receiving_house_cands": sum(1 for s1_id in s1_norm["entity_id"] if len(query_gt_comb.get(s1_id, set())) == 0 and len([cid for sid, cid in s2_strategy_pairs["house_name"] if sid == s1_id]) > 0),
    }

    # =======================================================================
    # TABLE 6: Experiment 5 — S2 vs S3 Separate Performance
    # =======================================================================
    logger.info("Computing S1 -> S2 and S1 -> S3 Separation Metrics...")
    table6_sources: Dict[str, Any] = {}
    for strat in strategies_to_test:
        cands_s2: Dict[str, Set[str]] = defaultdict(set)
        for (s1_id, cid) in s2_strategy_pairs[strat]:
            cands_s2[s1_id].add(cid)
        m_s2 = evaluate_candidates(cands_s2, query_gt_s2, total_s1_count=len(s1_norm), total_target_pool_size=len(s2_norm))

        cands_s3: Dict[str, Set[str]] = defaultdict(set)
        for (s1_id, cid) in s3_strategy_pairs[strat]:
            cands_s3[s1_id].add(cid)
        m_s3 = evaluate_candidates(cands_s3, query_gt_s3, total_s1_count=len(s1_norm), total_target_pool_size=len(s3_norm))

        table6_sources[strat] = {
            "s2_recall": m_s2["candidate_pair_recall"],
            "s2_avg_cands": m_s2["avg_candidates_per_s1"],
            "s2_precision": m_s2["candidate_precision"],
            "s3_recall": m_s3["candidate_pair_recall"],
            "s3_avg_cands": m_s3["avg_candidates_per_s1"],
            "s3_precision": m_s3["candidate_precision"],
        }

    # =======================================================================
    # TABLE 7: Experiment 6 — Hard Negative Impact & Mining
    # =======================================================================
    logger.info("Computing Hard Negative Statistics across Blocks and Mining Negatives...")
    table7_neg_stats: Dict[str, Any] = {}
    for strat in strategies_to_test:
        tot_cands = table1_individual[strat]["total_candidates_generated"]
        true_cands = table1_individual[strat]["captured_true_pairs"]
        false_cands = tot_cands - true_cands
        prec = table1_individual[strat]["candidate_precision"]
        table7_neg_stats[strat] = {
            "total_candidates": tot_cands,
            "true_candidates": true_cands,
            "false_candidates": false_cands,
            "candidate_precision": prec,
            "false_candidate_rate": round(false_cands / max(tot_cands, 1), 6),
        }

    # Mine 500+ diverse hard negatives with specific categories
    hard_negatives: List[Dict[str, Any]] = []
    # Collect candidate pairs from all strategies
    all_pairs = list(comb_union_candidates_with_sources.items())

    for s1_id, cand_map in all_pairs:
        true_matches = query_gt_comb.get(s1_id, set())
        for cid, sources in cand_map.items():
            if cid not in true_matches and s1_id in s1_map.index:
                r1 = s1_map.loc[s1_id]
                r2 = s2_map.loc[cid] if cid in s2_map.index else (s3_map.loc[cid] if cid in s3_map.index else None)
                if r2 is None:
                    continue

                category = "Dissimilar Candidate"
                if r1["name_norm"] == r2["name_norm"]:
                    category = "Same Name, Different Entity"
                elif r1["name_core"] == r2["name_core"]:
                    category = "Same Core Name, Different Entity"
                elif r1["house_number"] and r1["house_number"] == r2["house_number"] and r1["name_core"] != r2["name_core"]:
                    category = "Same House Number, Different Entity"
                elif r1["postal_code"] and r1["postal_code"] == r2["postal_code"] and r1["name_core"] != r2["name_core"]:
                    category = "Same Postal Code, Different Entity"
                elif r1["address_norm"] and r1["address_norm"] == r2["address_norm"]:
                    category = "Same Address, Different Entity"

                if category != "Dissimilar Candidate":
                    hard_negatives.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "country": r1["country_norm"],
                        "s1_name": r1["business_name"],
                        "candidate_name": r2["business_name"],
                        "s1_address": r1["business_address"],
                        "candidate_address": r2["business_address"],
                        "negative_category": category,
                        "blocking_sources": ",".join(sorted(sources)),
                    })
                    if len(hard_negatives) >= 600:
                        break
        if len(hard_negatives) >= 600:
            break

    df_hard_neg = pd.DataFrame(hard_negatives)
    neg_out_path = EXPERIMENTS_DIR / "hard_blocking_negatives_budget.tsv"
    df_hard_neg.to_csv(neg_out_path, sep="\t", index=False)
    logger.info(f"Saved {len(df_hard_neg):,} categorized hard negatives to {neg_out_path}")

    # Compile entire experiment payload
    payload = {
        "experiment_type": "SUBSET EXPERIMENT",
        "sample_queries": sample_queries,
        "target_pool_size": target_sample_size,
        "table1_individual": table1_individual,
        "table2_uncapped_unions": table2_uncapped_unions,
        "table3_budgets": table3_budgets,
        "table4_incremental": table4_incremental,
        "table5_house_investigation": house_investigation,
        "table6_sources": table6_sources,
        "table7_neg_stats": table7_neg_stats,
    }

    payload_path = EXPERIMENTS_DIR / "phase2_budget_data.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved complete Phase 2 Budget evaluation data to {payload_path}")

    return payload


if __name__ == "__main__":
    run_budget_experiments()
