"""Official Evaluation Metric Module for Amazon ML Challenge 2026.

Metric: Macro-averaged F_beta (beta = 0.5) per Source 1 entity.
Singletons (entities with 0 true matches) are strictly handled:
- Predicted empty for a true singleton: score = 1.0
- Predicted non-empty for a true singleton: score = 0.0
- Predicted empty for a non-singleton: score = 0.0
- Otherwise: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

from typing import Dict, List, Set, Union
import pandas as pd


def compute_entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str],
) -> float:
    """Computes F_0.5 score for a single Source 1 entity.

    Args:
        true_ids: Set of ground-truth matched entity IDs.
        pred_ids: Set of predicted matched entity IDs.

    Returns:
        F_0.5 score in range [0.0, 1.0].
    """
    # Case 1: Singleton (true matches is empty)
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    # Case 2: Non-singleton, but predicted empty
    if len(pred_ids) == 0:
        return 0.0

    # Case 3: Both non-empty
    true_positives = len(true_ids.intersection(pred_ids))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_ids)
    recall = true_positives / len(true_ids)

    # Beta = 0.5: beta^2 = 0.25, 1 + beta^2 = 1.25
    beta_sq = 0.25
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Computes macro-averaged F_0.5 score across all Source 1 entities.

    Args:
        ground_truth: Mapping of source1_entity_id to set of true matched IDs.
        predictions: Mapping of source1_entity_id to set of predicted matched IDs.

    Returns:
        Dictionary with macro_f05, mean_precision, mean_recall, and singleton_accuracy.
    """
    scores: List[float] = []
    singleton_scores: List[float] = []
    non_singleton_scores: List[float] = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    non_singleton_f05 = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "total_evaluated_entities": len(scores),
    }
