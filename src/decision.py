"""Macro-F0.5 Decision Engine, Threshold Calibration, and Score-Gap Analyzer.

Optimized strictly for the Amazon ML Challenge 2026 Macro-F0.5 metric:
- Macro-averaged F_0.5 penalizes false positives 2x heavier than false negatives (beta = 0.5)
- True singletons (0 matches) score 1.0 ONLY if predicted empty, and 0.0 upon any false positive
- Naturally supports zero, one, or multiple matches
- Score-gap margin calibration for disambiguating ambiguous close candidate scores
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.evaluate import compute_entity_f05


def predict_matches(
    candidate_predictions: Sequence[Dict[str, Any]],
    threshold: float = 0.50,
    score_gap_threshold: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """Generates predicted candidate sets per S1 entity at a given threshold.

    Allows zero, one, or multiple matches per S1 entity.

    Args:
        candidate_predictions: List of dicts with 'source1_entity_id',
                               'candidate_entity_id', and 'score'.
        threshold: Decision probability threshold.
        score_gap_threshold: Optional minimum score gap between the top candidate
                             and subsequent candidates to permit multi-match.

    Returns:
        Dict mapping source1_entity_id -> set of predicted candidate entity IDs.
    """
    # Group candidates by S1
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for p in candidate_predictions:
        grouped[p["source1_entity_id"]].append((p["candidate_entity_id"], float(p["score"])))

    predictions: Dict[str, Set[str]] = {}

    for s1_id, cand_scores in grouped.items():
        # Sort descending by score
        cand_scores.sort(key=lambda x: x[1], reverse=True)

        # Baseline threshold filtering
        passing = [cid for cid, score in cand_scores if score >= threshold]

        if not passing:
            predictions[s1_id] = set()
            continue

        if score_gap_threshold is not None and len(passing) > 1:
            # Apply score gap rule: keep top-1, and keep additional candidates only if
            # (top_score - cand_score) <= score_gap_threshold
            top_score = cand_scores[0][1]
            gap_filtered = [passing[0]]
            for cid in passing[1:]:
                # find its score
                c_score = next(s for c, s in cand_scores if c == cid)
                if (top_score - c_score) <= score_gap_threshold:
                    gap_filtered.append(cid)
            predictions[s1_id] = set(gap_filtered)
        else:
            predictions[s1_id] = set(passing)

    return predictions


def evaluate_decision_metrics(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Calculates comprehensive macro and micro evaluation metrics for entity resolution.

    Strictly conforms to official competition evaluation rules.

    Args:
        ground_truth: Mapping source1_id -> set of ground truth entity IDs.
        predictions: Mapping source1_id -> set of predicted matched entity IDs.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    total_pred_matches = 0
    total_true_matches = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    singleton_count = 0
    singleton_correct_empty = 0
    singleton_false_match = 0

    single_match_s1_count = 0
    single_match_s1_correct = 0

    multi_match_s1_count = 0
    multi_match_s1_correct = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        score = compute_entity_f05(true_set, pred_set)
        f05_scores.append(score)

        pred_len = len(pred_set)
        true_len = len(true_set)
        tp = len(true_set.intersection(pred_set))
        fp = pred_len - tp
        fn = true_len - tp

        total_pred_matches += pred_len
        total_true_matches += true_len
        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn

        # Entity-level precision and recall
        if pred_len > 0:
            precisions.append(tp / pred_len)
        elif true_len == 0:
            precisions.append(1.0)  # Correctly predicted empty singleton
        else:
            precisions.append(0.0)

        if true_len > 0:
            recalls.append(tp / true_len)
        elif pred_len == 0:
            recalls.append(1.0)  # Correctly predicted empty singleton
        else:
            recalls.append(0.0)

        # Entity category breakdowns
        if true_len == 0:
            singleton_count += 1
            if pred_len == 0:
                singleton_correct_empty += 1
            else:
                singleton_false_match += 1
        elif true_len == 1:
            single_match_s1_count += 1
            if pred_set == true_set:
                single_match_s1_correct += 1
        else:
            multi_match_s1_count += 1
            if pred_set == true_set:
                multi_match_s1_correct += 1

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0

    micro_prec = (total_true_positives / total_pred_matches) if total_pred_matches > 0 else 0.0
    micro_rec = (total_true_positives / total_true_matches) if total_true_matches > 0 else 0.0

    singleton_acc = (singleton_correct_empty / singleton_count) if singleton_count > 0 else 1.0
    singleton_fp_rate = (singleton_false_match / singleton_count) if singleton_count > 0 else 0.0

    single_match_acc = (single_match_s1_correct / single_match_s1_count) if single_match_s1_count > 0 else 0.0
    multi_match_acc = (multi_match_s1_correct / multi_match_s1_count) if multi_match_s1_count > 0 else 0.0

    return {
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "total_predicted_matches": total_pred_matches,
        "total_true_matches": total_true_matches,
        "total_true_positives": total_true_positives,
        "false_merges_count": total_false_positives,
        "missed_true_pairs_count": total_false_negatives,
        "singleton_count": singleton_count,
        "singleton_exact_empty_acc": round(singleton_acc, 4),
        "singleton_false_match_rate": round(singleton_fp_rate, 4),
        "single_match_s1_accuracy": round(single_match_acc, 4),
        "multi_match_s1_accuracy": round(multi_match_acc, 4),
    }


def sweep_thresholds(
    candidate_predictions: Sequence[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Executes a complete threshold sweep from 0.05 to 0.95 and returns a comparison table."""
    test_thresholds = thresholds or [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
    results: List[Dict[str, Any]] = []

    for thresh in test_thresholds:
        preds = predict_matches(candidate_predictions, threshold=thresh)
        metrics = evaluate_decision_metrics(ground_truth, preds)
        metrics["threshold"] = thresh
        results.append(metrics)

    df_sweep = pd.DataFrame(results)
    # Reorder columns with threshold and macro_f05 first
    cols = ["threshold", "macro_f05", "macro_precision", "macro_recall",
            "micro_precision", "micro_recall", "total_predicted_matches",
            "false_merges_count", "missed_true_pairs_count",
            "singleton_false_match_rate", "singleton_exact_empty_acc",
            "single_match_s1_accuracy", "multi_match_s1_accuracy"]
    remaining = [c for c in df_sweep.columns if c not in cols]
    return df_sweep[cols + remaining]
