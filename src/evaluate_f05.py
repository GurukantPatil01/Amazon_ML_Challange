"""Macro-F0.5 Evaluation and Candidate-Ceiling Diagnostic Module.

Performs strict competition-grade entity-level evaluation and decomposes error sources:
1. Candidate Recall Ceiling (blocking layer limit)
2. Model Recall among Candidates (ranking layer limit)
3. End-to-End Recall
4. Missed pair categorizations (text similarity, missing address, country, source)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.evaluate import compute_entity_f05


def analyze_candidate_ceiling(
    ground_truth: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
    model_predictions: Dict[str, Set[str]],
    s1_lookup: Dict[str, Dict[str, Any]],
    target_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes the strict three-tier recall decomposition and diagnoses missed pairs.

    1. Candidate Recall = (Ground-truth pairs in Candidate Set) / (Total Ground-truth pairs)
    2. Model Recall = (Ground-truth pairs in Predictions) / (Ground-truth pairs in Candidate Set)
    3. End-to-End Recall = (Ground-truth pairs in Predictions) / (Total Ground-truth pairs)

    Args:
        ground_truth: Mapping source1_id -> set of true matched IDs.
        candidate_dict: Mapping source1_id -> set of candidate entity IDs.
        model_predictions: Mapping source1_id -> set of predicted matched IDs.
        s1_lookup: Dict mapping source1_id -> normalized record dict.
        target_lookup: Dict mapping candidate_id -> normalized record dict.

    Returns:
        Dictionary with recall breakdown and categorized missed pairs.
    """
    total_gt_pairs = 0
    candidate_captured_pairs = 0
    model_captured_pairs = 0

    missed_at_blocking: List[Dict[str, Any]] = []
    missed_at_model: List[Dict[str, Any]] = []

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue

        cands = candidate_dict.get(s1_id, set())
        preds = model_predictions.get(s1_id, set())

        s1_rec = s1_lookup.get(s1_id, {})
        s1_name = s1_rec.get("name_norm", "")
        s1_addr = s1_rec.get("address_norm", "")
        country = s1_rec.get("country_norm", "")

        for cid in true_matches:
            total_gt_pairs += 1
            cand_rec = target_lookup.get(cid, {})
            c_name = cand_rec.get("name_norm", "")
            c_addr = cand_rec.get("address_norm", "")
            target_source = "source2" if "S2" in cid else "source3"

            name_sim = float(Levenshtein.normalized_similarity(s1_name, c_name)) if (s1_name and c_name) else 0.0
            addr_sim = float(Levenshtein.normalized_similarity(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0
            addr_missing = (not s1_addr) or (not c_addr)

            pair_info = {
                "source1_id": s1_id,
                "candidate_id": cid,
                "country": country,
                "target_source": target_source,
                "name_similarity": round(name_sim, 4),
                "address_similarity": round(addr_sim, 4),
                "address_missing": addr_missing,
                "s1_name": s1_rec.get("business_name", ""),
                "target_name": cand_rec.get("business_name", ""),
            }

            if cid in cands:
                candidate_captured_pairs += 1
                if cid in preds:
                    model_captured_pairs += 1
                else:
                    missed_at_model.append(pair_info)
            else:
                missed_at_blocking.append(pair_info)

    cand_recall = (candidate_captured_pairs / total_gt_pairs) if total_gt_pairs > 0 else 1.0
    model_recall_among_cands = (
        (model_captured_pairs / candidate_captured_pairs) if candidate_captured_pairs > 0 else 1.0
    )
    end_to_end_recall = (model_captured_pairs / total_gt_pairs) if total_gt_pairs > 0 else 1.0

    # Categorize blocking misses
    blocking_miss_breakdown = {
        "total_missed_at_blocking": len(missed_at_blocking),
        "missing_address_count": sum(1 for m in missed_at_blocking if m["address_missing"]),
        "high_name_sim_missed": sum(1 for m in missed_at_blocking if m["name_similarity"] >= 0.70),
        "low_name_sim_missed": sum(1 for m in missed_at_blocking if m["name_similarity"] < 0.40),
        "s2_misses": sum(1 for m in missed_at_blocking if m["target_source"] == "source2"),
        "s3_misses": sum(1 for m in missed_at_blocking if m["target_source"] == "source3"),
    }

    return {
        "total_ground_truth_pairs": total_gt_pairs,
        "candidate_captured_pairs": candidate_captured_pairs,
        "model_captured_pairs": model_captured_pairs,
        "candidate_recall_ceiling": round(cand_recall, 4),
        "model_recall_among_candidates": round(model_recall_among_cands, 4),
        "end_to_end_recall": round(end_to_end_recall, 4),
        "blocking_miss_breakdown": blocking_miss_breakdown,
        "missed_at_blocking_samples": missed_at_blocking[:20],
        "missed_at_model_samples": missed_at_model[:20],
    }
