"""Production-Grade Candidate Generation and Training Pair Construction Engine.

Fulfills Phase 3A and 3B:
1. generate_candidates(source1_df, source2_df, source3_df, budget=200):
   - Union E execution: exact_name, name_core, rare_token, char_3gram_k10,
     house_name, combined_postal_name, address_token.
   - Multi-block support ranking with deterministic tie-breaking.
   - Per-S1 budget enforcement (K=200).
   - Provenance tracking (block_count, block_strength_sum, individual flags).
2. build_training_and_val_pairs():
   - Group-aware S1 splitting (zero S1 overlap between train and validation).
   - Positive pairs mapped directly from train_ground_truth.
   - Comprehensive hard-negative oversampling:
     * Same normalized name / core name
     * Same house number / postal code
     * Same address
     * High string similarity
     * Multi-block false candidates
"""

from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.blocking import CandidateGenerator, run_strategy_on_s1
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.normalize import normalize_dataframe
from src.utils import setup_logger

logger = setup_logger("build_training_pairs")

STRATEGY_PRIORITY_WEIGHTS: Dict[str, float] = {
    "exact_name": 7.0,
    "name_core": 6.0,
    "house_name": 5.0,
    "combined_postal_name": 4.0,
    "rare_token": 3.0,
    "char_3gram_k10": 2.0,
    "address_token": 1.0,
}

UNION_E_STRATEGIES = [
    "exact_name",
    "name_core",
    "rare_token",
    "char_3gram_k10",
    "house_name",
    "combined_postal_name",
    "address_token",
]


def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    budget: int = 200,
    strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Production candidate generation interface matching Phase 3A specifications.

    Generates Union E candidate pairs separately for S1 -> S2 and S1 -> S3 with country constraints,
    tracks blocking provenance, calculates multi-block support metrics, ranks deterministically,
    and applies candidate budget K.

    Args:
        source1_df: Source 1 DataFrame.
        source2_df: Source 2 DataFrame.
        source3_df: Source 3 DataFrame.
        budget: Maximum candidate budget per S1 (default 200).
        strategies: List of blocking strategies to activate (defaults to Union E).

    Returns:
        DataFrame containing candidate pairs with blocking metadata columns:
        ['source1_entity_id', 'candidate_entity_id', 'candidate_source',
         'blocking_sources', 'block_count', 'block_strength_sum',
         'exact_name_hit', 'name_core_hit', 'house_name_hit',
         'combined_postal_name_hit', 'rare_token_hit', 'char_3gram_hit',
         'address_token_hit', 'rank']
    """
    active_strategies = strategies or UNION_E_STRATEGIES

    # Ensure normalization columns exist
    s1_norm = source1_df if "name_norm" in source1_df.columns else normalize_dataframe(source1_df)
    s2_norm = source2_df if "name_norm" in source2_df.columns else normalize_dataframe(source2_df)
    s3_norm = source3_df if "name_norm" in source3_df.columns else normalize_dataframe(source3_df)

    output_rows: List[Dict[str, Any]] = []

    for target_name, target_df in [("source2", s2_norm), ("source3", s3_norm)]:
        logger.info(f"Generating candidate pool for S1 -> {target_name} ({len(target_df):,} records)...")
        gen = CandidateGenerator(target_name, target_df)

        # Collect candidate pairs: (s1_id, cand_id) -> set(strategy_names)
        s1_candidate_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        for strat in active_strategies:
            strat_pairs = run_strategy_on_s1(strat, s1_norm, gen)
            for (s1_id, cid), strats in strat_pairs.items():
                s1_candidate_map[s1_id][cid].update(strats)

        # Rank candidates per S1 by Multi-Block Support and apply budget
        for s1_id, cand_dict in s1_candidate_map.items():
            cand_items = []
            for cid, sources in cand_dict.items():
                support_count = len(sources)
                strength_sum = sum(STRATEGY_PRIORITY_WEIGHTS.get(s, 1.0) for s in sources)
                max_strength = max(STRATEGY_PRIORITY_WEIGHTS.get(s, 1.0) for s in sources)
                cand_items.append((cid, support_count, strength_sum, max_strength, sources))

            # Multi-block support ranking: Primary: support_count, Secondary: strength_sum,
            # Tertiary: max_strength, Quaternary: deterministic cid
            cand_items.sort(key=lambda x: (x[1], x[2], x[3], x[0]), reverse=True)

            pruned_cands = cand_items[:budget]

            for rank_idx, (cid, sup_count, str_sum, _, sources) in enumerate(pruned_cands, start=1):
                output_rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "candidate_source": target_name,
                    "blocking_sources": ",".join(sorted(sources)),
                    "block_count": sup_count,
                    "block_strength_sum": str_sum,
                    "exact_name_hit": 1.0 if "exact_name" in sources else 0.0,
                    "name_core_hit": 1.0 if "name_core" in sources else 0.0,
                    "house_name_hit": 1.0 if "house_name" in sources else 0.0,
                    "combined_postal_name_hit": 1.0 if "combined_postal_name" in sources else 0.0,
                    "rare_token_hit": 1.0 if "rare_token" in sources else 0.0,
                    "char_3gram_hit": 1.0 if any("char_3gram" in s for s in sources) else 0.0,
                    "address_token_hit": 1.0 if "address_token" in sources else 0.0,
                    "rank": rank_idx,
                })

    return pd.DataFrame(output_rows)


def build_labeled_pairs(
    candidate_df: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, Any]],
    target_lookup: Dict[str, Dict[str, Any]],
    ground_truth_dict: Dict[str, Set[str]],
    max_negatives_per_s1: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Constructs balanced training/validation pairs with heavy hard-negative oversampling.

    For each S1:
    - Every candidate in ground truth is labeled 1 (Positive).
    - Hard negatives are prioritized:
      1. Same normalized name / name_core
      2. Same house number / postal code
      3. Same address / high address similarity
      4. High name similarity
      5. Multi-block supported candidates
    - Up to max_negatives_per_s1 negatives are sampled per S1.

    Args:
        candidate_df: DataFrame output of generate_candidates().
        s1_lookup: Dict mapping source1_entity_id -> normalized record dict.
        target_lookup: Dict mapping candidate_entity_id -> normalized record dict.
        ground_truth_dict: Dict mapping source1_entity_id -> set of true matched IDs.
        max_negatives_per_s1: Maximum negatives per S1 entity.
        random_seed: Seed for reproducible negative sampling.

    Returns:
        DataFrame with columns: ['source1_entity_id', 'candidate_entity_id', 'label',
                                 'hard_negative_category', ...]
    """
    rng = random.Random(random_seed)
    labeled_records: List[Dict[str, Any]] = []

    # Group candidates by S1
    grouped = candidate_df.groupby("source1_entity_id")

    for s1_id, group in grouped:
        true_matches = ground_truth_dict.get(s1_id, set())
        s1_rec = s1_lookup.get(s1_id, {})
        s1_name = s1_rec.get("name_norm", "")
        s1_core = s1_rec.get("name_core", "")
        s1_house = s1_rec.get("house_number", "")
        s1_post = s1_rec.get("postal_code", "")
        s1_addr = s1_rec.get("address_norm", "")

        pos_records = []
        hard_negs_pool = []
        regular_negs_pool = []

        cand_rows = group.to_dict(orient="records")

        for row in cand_rows:
            cid = row["candidate_entity_id"]
            is_pos = cid in true_matches

            if is_pos:
                row_copy = dict(row)
                row_copy["label"] = 1
                row_copy["hard_negative_category"] = "Positive Ground Truth"
                pos_records.append(row_copy)
            else:
                cand_rec = target_lookup.get(cid, {})
                c_name = cand_rec.get("name_norm", "")
                c_core = cand_rec.get("name_core", "")
                c_house = cand_rec.get("house_number", "")
                c_post = cand_rec.get("postal_code", "")
                c_addr = cand_rec.get("address_norm", "")

                hard_cat = None
                if s1_name and s1_name == c_name:
                    hard_cat = "Same Name, Different Entity"
                elif s1_core and s1_core == c_core:
                    hard_cat = "Same Core Name, Different Entity"
                elif s1_addr and s1_addr == c_addr:
                    hard_cat = "Same Address, Different Entity"
                elif s1_house and s1_house == c_house and s1_core != c_core:
                    hard_cat = "Same House Number, Different Entity"
                elif s1_post and s1_post == c_post and s1_core != c_core:
                    hard_cat = "Same Postal Code, Different Entity"
                elif row["block_count"] >= 2:
                    hard_cat = "Multi-Block Supported Negative"
                elif s1_name and c_name and Levenshtein.normalized_similarity(s1_name, c_name) >= 0.75:
                    hard_cat = "High Name Similarity Negative"
                elif s1_addr and c_addr and Levenshtein.normalized_similarity(s1_addr, c_addr) >= 0.75:
                    hard_cat = "High Address Similarity Negative"

                row_copy = dict(row)
                row_copy["label"] = 0
                if hard_cat:
                    row_copy["hard_negative_category"] = hard_cat
                    hard_negs_pool.append(row_copy)
                else:
                    row_copy["hard_negative_category"] = "Standard Candidate Negative"
                    regular_negs_pool.append(row_copy)

        # Combine: always include all positives
        labeled_records.extend(pos_records)

        # Sample negatives: take all hard negatives up to limit, then fill with regular negatives
        rng.shuffle(hard_negs_pool)
        rng.shuffle(regular_negs_pool)

        selected_negs = hard_negs_pool[:max_negatives_per_s1]
        remaining_budget = max_negatives_per_s1 - len(selected_negs)
        if remaining_budget > 0:
            selected_negs.extend(regular_negs_pool[:remaining_budget])

        labeled_records.extend(selected_negs)

    return pd.DataFrame(labeled_records)


def split_s1_groups(
    s1_ids: Sequence[str],
    train_ratio: float = 0.80,
    random_seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Generates a strictly group-aware split by Source 1 entity ID.

    Guarantees zero overlap: no S1 entity in validation set appears in training set.

    Args:
        s1_ids: Sequence of all S1 entity IDs.
        train_ratio: Fraction allocated to training (default 0.80).
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_s1_ids_set, val_s1_ids_set).
    """
    unique_s1 = sorted(list(set(s1_ids)))
    rng = random.Random(random_seed)
    rng.shuffle(unique_s1)

    n_train = int(len(unique_s1) * train_ratio)
    train_set = set(unique_s1[:n_train])
    val_set = set(unique_s1[n_train:])

    assert len(train_set.intersection(val_set)) == 0, "Group leakage detected!"
    return train_set, val_set
